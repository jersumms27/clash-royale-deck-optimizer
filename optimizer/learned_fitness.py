"""GA fitness from the learned matchup model: expected win rate vs the meta.

fitness(deck) = sum_i weight_i * P(deck beats meta_deck_i)

`LearnedFitness` is callable like heuristic.score, and also exposes
`score_batch` so the GA can score a whole generation in one GPU pass.
"""

from __future__ import annotations

import contextlib

import torch

from optimizer import config
from optimizer.cr_api import load_card_pool
from optimizer.meta import MetaDeck, load_meta_decks
from optimizer.model import MatchupModel
from optimizer.models import Deck


class LearnedFitness:
    def __init__(
        self,
        model_path=None,
        meta_path=None,
        *,
        meta: list[MetaDeck] | None = None,
        device: str | torch.device | None = None,
        chunk_pairs: int = 8192,
    ):
        """`meta` overrides the file: pass your own MetaDecks (e.g. a single
        target deck) to optimise against them instead of the ladder meta."""
        model_path = model_path or config.MODEL_PATH
        meta_path = meta_path or config.META_DECKS_CSV
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = MatchupModel.load(model_path, self.device)
        if meta is None:
            meta = load_meta_decks(meta_path, known_ids=set(self.model.card_ids))
            if not meta:
                raise RuntimeError(
                    f"{meta_path} has no usable meta decks (see data/README.md)."
                )
        elif not meta:
            raise ValueError("meta must contain at least one deck")
        self.meta = list(meta)
        self.meta_idx, self.meta_form = self.model.encode_decks(self.meta, self.device)
        self.weights = torch.tensor(
            [m.weight for m in self.meta], dtype=torch.float32, device=self.device
        )
        self.weights /= self.weights.sum()
        self.chunk_pairs = chunk_pairs
        # Cards released after the model was trained are scored as MASK tokens
        # (a neutral "some card"); retrain to give them real embeddings.
        self.unknown_cards = sorted(
            set(load_card_pool().by_id) - set(self.model.card_ids)
        )

    # -- scoring --------------------------------------------------------------- #
    def encode(self, decks):
        """Decks -> model index tensors; cards newer than the model are masked."""
        return self.model.encode_decks(decks, self.device, unknown="mask")

    def score_batch(self, decks: list[Deck]) -> list[float]:
        """Expected win rate of each deck against the weighted meta."""
        if not decks:
            return []
        a_idx, a_form = self.encode(decks)
        n, m = len(decks), len(self.meta)
        # Every (candidate, meta) pair: candidates repeated m times, meta tiled n times.
        pa_idx, pa_form = a_idx.repeat_interleave(m, 0), a_form.repeat_interleave(m, 0)
        pb_idx, pb_form = self.meta_idx.repeat(n, 1), self.meta_form.repeat(n, 1)

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        probs = []
        with torch.inference_mode(), autocast:
            for s in range(0, n * m, self.chunk_pairs):
                e = s + self.chunk_pairs
                logits = self.model(pa_idx[s:e], pa_form[s:e], pb_idx[s:e], pb_form[s:e])
                probs.append(torch.sigmoid(logits.float()))
        win = torch.cat(probs).view(n, m) @ self.weights
        return win.tolist()

    def __call__(self, deck: Deck) -> float:
        return self.score_batch([deck])[0]

    def matchups(self, deck: Deck) -> list[tuple[float, float]]:
        """[(weight, P(deck beats meta_i))] for one deck, in meta order."""
        a_idx, a_form = self.encode([deck])
        m = len(self.meta)
        with torch.inference_mode():
            p = torch.sigmoid(
                self.model(a_idx.repeat(m, 1), a_form.repeat(m, 1), self.meta_idx, self.meta_form)
            )
        return list(zip(self.weights.tolist(), p.tolist()))

    def info(self) -> dict:
        return {
            "device": str(self.device),
            "meta_decks": len(self.meta),
            "has_hero_data": self.model.has_hero_data,
            "unknown_cards": len(self.unknown_cards),
            **{k: v for k, v in self.model.meta.items()
               if isinstance(v, (int, float, str, bool))},
        }
