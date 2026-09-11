"""Learned matchup model: P(deck A beats deck B), in PyTorch.

Architecture
  card token  = Linear([ learned card embedding | normalised stats from cards.csv ])
                + form embedding (base / evo / hero) + side embedding (mine / theirs)
  encoder     = Transformer self-attention over all 16 tokens (no positional
                encoding: a deck is a *set* of (card, form) tokens -- slot position
                is meaningless, only the form matters)
  head        = f(A, B) = MLP([mean of A's tokens | mean of B's tokens])
  logit(A, B) = f(A, B) - f(B, A)      <- exactly antisymmetric, so
                P(A beats B) + P(B beats A) == 1 by construction.

Training is `fit` (BCE on soft labels, early stopping on validation log-loss).
Because the model is antisymmetric, swapping sides and flipping the label is a
no-op, so no side-swap augmentation is needed. Card-token dropout (randomly
masking a card) is the regulariser that makes it generalise to unseen decks.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from optimizer import config
from optimizer.models import CardPool, Deck

FORM_BASE, FORM_EVO, FORM_HERO = 0, 1, 2
N_FORMS = 3
DECK_SIZE = config.DECK_SIZE

# --------------------------------------------------------------------------- #
# Card features from cards.csv                                                 #
# --------------------------------------------------------------------------- #
_NUMERIC = ["elixir", "hitpoints", "damage", "damage_per_second", "attack_period",
            "range", "radius", "lifetime", "crown_tower_damage", "special_damage",
            "evo_cycles", "evo_overall_cost"]
_TYPES = ["troop", "building", "spell"]
_RARITIES = ["common", "rare", "epic", "legendary", "champion"]
_WIN_TIERS = ["primary", "secondary", "conditional"]
_SPELL_SIZES = ["small", "medium", "large"]


def build_card_features(pool: CardPool) -> tuple[list[int], Tensor]:
    """(card ids sorted, feature matrix [n_cards, F]) for every card in the pool.

    Numeric stats are z-scored over the pool with None -> 0 plus a missing flag
    per column; categorical fields become one-hots; booleans pass through.
    """
    cards = sorted(pool.cards, key=lambda c: c.id)
    norms = []
    for f in _NUMERIC:
        present = [getattr(c, f) for c in cards if getattr(c, f) is not None]
        mean = sum(present) / len(present) if present else 0.0
        var = sum((v - mean) ** 2 for v in present) / len(present) if present else 0.0
        norms.append((mean, math.sqrt(var) or 1.0))

    rows = []
    for c in cards:
        row: list[float] = []
        for f, (mean, std) in zip(_NUMERIC, norms):
            v = getattr(c, f)
            row.append((v - mean) / std if v is not None else 0.0)
            row.append(0.0 if v is not None else 1.0)
        row += [1.0 if c.type == t else 0.0 for t in _TYPES]
        row += [1.0 if c.rarity.lower() == r else 0.0 for r in _RARITIES]
        row += [1.0 if c.win_condition == w else 0.0 for w in _WIN_TIERS]
        row += [1.0 if c.spell_size == s else 0.0 for s in _SPELL_SIZES]
        row += [float(c.air), float(c.has_evolution), float(c.is_champion_hero),
                float(c.is_champion)]
        rows.append(row)
    return [c.id for c in cards], torch.tensor(rows, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def _unpack(deck) -> tuple[Iterable[int], set[int], set[int]]:
    """(card ids, evolved ids, hero ids) from a Deck, a MetaDeck, or a 3-tuple."""
    if isinstance(deck, Deck):
        return [c.id for c in deck.cards], set(deck.evolved), set(deck.hero)
    if isinstance(deck, tuple) and len(deck) == 3:
        return deck[0], set(deck[1]), set(deck[2])
    evo = getattr(deck, "evo", None)
    if evo is None:
        evo = getattr(deck, "evolved", ())
    return deck.cards, set(evo), set(getattr(deck, "hero", ()))


class MatchupModel(nn.Module):
    def __init__(
        self,
        card_ids: list[int],
        card_feats: Tensor,
        champion_ids: Iterable[int] = (),
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        card_dropout: float = 0.1,
        use_stats: bool = True,
        has_hero_data: bool = False,
    ):
        super().__init__()
        self.hparams = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                            d_ff=d_ff, dropout=dropout, card_dropout=card_dropout,
                            use_stats=use_stats, has_hero_data=has_hero_data)
        self.card_ids = [int(c) for c in card_ids]
        self.card_index = {cid: i for i, cid in enumerate(self.card_ids)}
        self.champion_ids = frozenset(int(c) for c in champion_ids)
        self.has_hero_data = has_hero_data
        self.card_dropout = card_dropout
        self.meta: dict = {}  # free-form info stored with the checkpoint

        n_cards = len(self.card_ids)
        self.mask_index = n_cards  # extra vocab row for a dropped-out card
        feats = torch.as_tensor(card_feats, dtype=torch.float32)
        if feats.shape[0] != n_cards:
            raise ValueError("card_feats must have one row per card id")
        feats = torch.cat([feats, torch.zeros(1, feats.shape[1])])  # MASK -> zeros
        if not use_stats:
            feats = torch.zeros_like(feats)
        self.register_buffer("card_feats", feats)

        self.card_emb = nn.Embedding(n_cards + 1, d_model)
        self.token_proj = nn.Linear(d_model + feats.shape[1], d_model)
        self.form_emb = nn.Embedding(N_FORMS, d_model)
        self.side_emb = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    @classmethod
    def from_pool(cls, pool: CardPool, **hparams) -> "MatchupModel":
        ids, feats = build_card_features(pool)
        return cls(ids, feats, pool.champion_set, **hparams)

    @property
    def device(self) -> torch.device:
        return self.card_emb.weight.device

    # -- encoding decks -> index tensors ------------------------------------- #
    def encode_decks(
        self, decks: Iterable, device=None, *, unknown: str = "raise"
    ) -> tuple[Tensor, Tensor]:
        """Decks -> (card indices [N, 8], forms [N, 8]).

        Champions are always encoded as base form (their hero-ness is part of
        the card itself). Hero form on other cards is only used when the model
        was trained with hero data; otherwise it is mapped to base so the GA
        can't exploit an untrained embedding.

        A card id the model wasn't trained on (released after training) raises
        by default; with unknown="mask" it becomes the MASK token -- the neutral
        "some card" the model learned to cope with through card dropout.
        """
        idx_rows, form_rows = [], []
        for deck in decks:
            cards, evo, hero = _unpack(deck)
            cards = list(cards)
            if len(cards) != DECK_SIZE:
                raise ValueError(f"deck must have {DECK_SIZE} cards, got {len(cards)}")
            if unknown == "mask":
                idx_rows.append([self.card_index.get(c, self.mask_index) for c in cards])
            else:
                try:
                    idx_rows.append([self.card_index[c] for c in cards])
                except KeyError as exc:
                    raise ValueError(f"card id {exc.args[0]} is unknown to the model") from None
            forms = []
            for c in cards:
                if c in evo:
                    forms.append(FORM_EVO)
                elif c in hero and self.has_hero_data and c not in self.champion_ids:
                    forms.append(FORM_HERO)
                else:
                    forms.append(FORM_BASE)
            form_rows.append(forms)
        device = device or self.device
        return (torch.tensor(idx_rows, dtype=torch.long, device=device),
                torch.tensor(form_rows, dtype=torch.long, device=device))

    # -- forward -------------------------------------------------------------- #
    def _tokens(self, idx: Tensor, form: Tensor, side: int) -> Tensor:
        if self.training and self.card_dropout > 0:
            drop = torch.rand(idx.shape, device=idx.device) < self.card_dropout
            idx = idx.masked_fill(drop, self.mask_index)
        x = torch.cat([self.card_emb(idx), self.card_feats[idx]], dim=-1)
        return self.token_proj(x) + self.form_emb(form) + self.side_emb.weight[side]

    def forward(self, a_idx: Tensor, a_form: Tensor, b_idx: Tensor, b_form: Tensor) -> Tensor:
        """Log-odds that each A beats its B. Shapes: all [N, 8] -> [N]."""
        n = a_idx.shape[0]
        # Both orientations in one batch: rows [0, n) are (A mine, B theirs),
        # rows [n, 2n) are (B mine, A theirs). Subtracting makes it antisymmetric.
        mine = self._tokens(torch.cat([a_idx, b_idx]), torch.cat([a_form, b_form]), 0)
        theirs = self._tokens(torch.cat([b_idx, a_idx]), torch.cat([b_form, a_form]), 1)
        h = self.norm(self.encoder(torch.cat([mine, theirs], dim=1)))  # [2n, 16, d]
        z_mine, z_theirs = h[:, :DECK_SIZE].mean(1), h[:, DECK_SIZE:].mean(1)
        f = self.head(torch.cat([z_mine, z_theirs], dim=-1)).squeeze(-1)  # [2n]
        return f[:n] - f[n:]

    @torch.inference_mode()
    def predict_proba(self, a_idx, a_form, b_idx, b_form) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self(a_idx, a_form, b_idx, b_form))
        finally:
            self.train(was_training)

    # -- persistence ---------------------------------------------------------- #
    def save(self, path=config.MODEL_PATH, meta: dict | None = None) -> None:
        self.meta = dict(self.meta, **(meta or {}))
        torch.save(
            {
                "hparams": self.hparams,
                "card_ids": self.card_ids,
                "champion_ids": sorted(self.champion_ids),
                "card_feats": self.card_feats[:-1].detach().cpu(),
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path=config.MODEL_PATH, device=None) -> "MatchupModel":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(ck["card_ids"], ck["card_feats"], ck["champion_ids"], **ck["hparams"])
        model.load_state_dict(ck["state_dict"])
        model.meta = dict(ck.get("meta", {}))
        return model.to(device or "cpu").eval()


# --------------------------------------------------------------------------- #
# Training helpers                                                             #
# --------------------------------------------------------------------------- #
def encode_battles(model: MatchupModel, rows, device=None) -> tuple[Tensor, ...]:
    """BattleRows -> (a_idx, a_form, b_idx, b_form, y) on `device`."""
    device = device or model.device
    a_idx, a_form = model.encode_decks((r.a for r in rows), device)
    b_idx, b_form = model.encode_decks((r.b for r in rows), device)
    y = torch.tensor([r.result for r in rows], dtype=torch.float32, device=device)
    return a_idx, a_form, b_idx, b_form, y


@torch.inference_mode()
def evaluate(model: MatchupModel, data: tuple[Tensor, ...], batch_size: int = 4096) -> dict:
    """Log-loss, accuracy (draws excluded) and Brier score on a tensor tuple."""
    a_idx, a_form, b_idx, b_form, y = data
    if not len(y):
        return {"n": 0, "logloss": float("nan"), "accuracy": float("nan"), "brier": float("nan")}
    was_training = model.training
    model.eval()
    logits = torch.cat([
        model(a_idx[s:s + batch_size], a_form[s:s + batch_size],
              b_idx[s:s + batch_size], b_form[s:s + batch_size])
        for s in range(0, len(y), batch_size)
    ])
    model.train(was_training)
    p = torch.sigmoid(logits)
    decided = y != 0.5
    acc = ((p[decided] > 0.5).float() == y[decided]).float().mean().item() if decided.any() else float("nan")
    return {
        "n": int(len(y)),
        "logloss": F.binary_cross_entropy_with_logits(logits, y).item(),
        "accuracy": acc,
        "brier": ((p - y) ** 2).mean().item(),
    }


def fit(
    model: MatchupModel,
    train: tuple[Tensor, ...],
    val: tuple[Tensor, ...] | None = None,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 1024,
    patience: int = 5,
    seed: int = 0,
    amp: bool | None = None,
    log: Callable[[str], None] | None = print,
) -> list[dict]:
    """Train in place with AdamW + cosine LR; keeps the best validation
    checkpoint (by log-loss) and stops after `patience` epochs without
    improvement. Returns the per-epoch history.

    On CUDA, `amp` (default on) runs the forward/backward in bfloat16 autocast
    and allows TF32 matmuls -- roughly 2x faster, no loss scaling needed."""
    torch.manual_seed(seed)
    a_idx, a_form, b_idx, b_form, y = train
    n = len(y)
    on_cuda = y.device.type == "cuda"
    if amp is None:
        amp = on_cuda and torch.cuda.is_bf16_supported()
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp and on_cuda
        else contextlib.nullcontext()
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    gen = torch.Generator(device="cpu").manual_seed(seed)

    history: list[dict] = []
    best_loss, best_state, since_best = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, generator=gen).to(y.device)
        total = torch.zeros((), device=y.device)
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            with autocast:
                logits = model(a_idx[b], a_form[b], b_idx[b], b_form[b])
            loss = F.binary_cross_entropy_with_logits(logits.float(), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.detach() * len(b)  # no per-step GPU sync
        sched.step()
        total = total.item()

        record = {"epoch": epoch, "train_logloss": total / max(1, n), "seconds": time.time() - t0}
        if val is not None and len(val[-1]):
            ev = evaluate(model, val)
            record.update(val_logloss=ev["logloss"], val_accuracy=ev["accuracy"])
            monitored = ev["logloss"]
        else:
            monitored = record["train_logloss"]
        history.append(record)
        if log:
            msg = f"epoch {epoch:3d} | train {record['train_logloss']:.4f}"
            if "val_logloss" in record:
                msg += f" | val {record['val_logloss']:.4f} (acc {record['val_accuracy']:.3f})"
            log(msg + f" | {record['seconds']:.1f}s")

        if monitored < best_loss - 1e-5:
            best_loss, since_best = monitored, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                if log:
                    log(f"early stop: no improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history
