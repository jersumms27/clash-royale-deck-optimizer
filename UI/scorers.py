"""Scorer registry for the web UI -- the things that can rate a deck.

Two scorers are known:

  heuristic  optimizer/heuristic.py's score(): the hand-tuned baseline.
  learned    the learned matchup model: P(deck A beats deck B), with GA fitness
             = usage-weighted expected win rate against the meta deck set.

This file is the ONLY place the UI touches a fitness function. It never imports
torch or a checkpoint itself; it imports optimizer/matchup.py (the model's
plain-function face; see LEARNED_MODULE_CANDIDATES) and duck-types what it
finds. If the import fails or is_available() says no (module not written yet,
torch missing, checkpoint missing) the learned scorer is listed as unavailable
*with the reason*, and the UI falls back to the heuristic. Nothing in
optimizer/ is modified.

Protocol for the learned module -- only `score` is required:

    score(deck: Deck) -> float
        GA fitness: expected win rate in [0, 1] vs the meta decks.
        (aliases: fitness, expected_win_rate)

    predict(deck_a: Deck, deck_b: Deck) -> float
        P(deck_a beats deck_b). Powers the Matchups tab.
        (aliases: win_probability, p_win, predict_matchup)

    meta_decks() -> iterable of
            Deck | (Deck, weight) | (Deck, weight, name)
            | {"deck": Deck, "weight": w, "name": n} | object with those attrs
        The meta set the fitness averages over (weights need not sum to 1;
        the UI normalizes). May also be a module-level META_DECKS list.
        (aliases: load_meta_decks)

    matchups(deck: Deck) -> iterable of
            (Deck, weight, p_win[, name]) | {"deck", "weight", "p_win", "name"}
        Optional. When absent the UI derives it from predict() + meta_decks().

    is_available() -> bool | (bool, reason)
        Optional. Lets the module report "no checkpoint yet" without raising.

    info() -> dict
        Optional. Checkpoint / device summary shown under the scorer in the UI
        (device, meta_decks, has_hero_data, validation metrics, ...). Fetched
        lazily so page load never waits on torch.

    KIND = "winrate" | "score"
        Optional. How the UI formats fitness ("winrate" -> 58.3 %,
        "score" -> 0.5830). Defaults to "winrate" for the learned scorer.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from optimizer.models import Deck

# Modules to try, in order, for the learned matchup model. First import wins.
# (optimizer/model.py is the network itself, not a scorer -- don't list it.)
LEARNED_MODULE_CANDIDATES = ("optimizer.matchup",)

# Accepted attribute names for each role (first hit wins).
_ALIASES = {
    "score": ("score", "fitness", "expected_win_rate"),
    "predict": ("predict", "win_probability", "p_win", "predict_matchup"),
    "meta": ("meta_decks", "load_meta_decks", "META_DECKS"),
    "matchups": ("matchups", "matchup_breakdown"),
    "info": ("info", "model_info"),
}


@dataclass(frozen=True)
class MetaDeck:
    deck: Deck
    weight: float
    name: str


@dataclass(frozen=True)
class Matchup:
    meta: MetaDeck
    p_win: float  # P(candidate beats meta.deck)


@dataclass
class Scorer:
    id: str
    label: str
    kind: str  # "winrate" | "score"
    description: str
    available: bool = True
    reason: str = ""  # why it's unavailable
    score: Callable[[Deck], float] | None = None
    predict: Callable[[Deck, Deck], float] | None = None
    meta: Callable[[], list[MetaDeck]] | None = None
    matchups: Callable[[Deck], list[Matchup]] | None = None
    info: Callable[[], dict] | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "description": self.description,
            "available": self.available,
            "reason": self.reason,
            "supports": {
                "score": self.score is not None,
                "predict": self.predict is not None,
                "meta": self.meta is not None,
                "matchups": self.matchups is not None,
                "info": self.info is not None,
            },
        }


# --------------------------------------------------------------------------- #
# Normalizers: accept the loose shapes listed in the module docstring          #
# --------------------------------------------------------------------------- #
def _pick(obj: Any, key: str, default=None):
    """Read `key` from a dict or an attribute-bearing object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_meta_deck(item: Any, index: int) -> MetaDeck:
    default_name = f"Meta deck {index + 1}"
    if isinstance(item, Deck):
        return MetaDeck(item, 1.0, default_name)
    if isinstance(item, (tuple, list)):
        deck = item[0]
        weight = float(item[1]) if len(item) > 1 else 1.0
        name = str(item[2]) if len(item) > 2 and item[2] else default_name
        return MetaDeck(deck, weight, name)
    deck = _pick(item, "deck")
    if not isinstance(deck, Deck):
        raise TypeError(f"meta deck #{index + 1}: can't find a Deck in {type(item).__name__}")
    weight = _pick(item, "weight", 1.0)
    weight = 1.0 if weight is None else float(weight)
    return MetaDeck(deck, weight, str(_pick(item, "name") or default_name))


def _as_matchup(item: Any, index: int) -> Matchup:
    if isinstance(item, (tuple, list)):
        deck, weight = item[0], float(item[1]) if len(item) > 1 else 1.0
        p = float(item[2]) if len(item) > 2 else float("nan")
        name = str(item[3]) if len(item) > 3 and item[3] else f"Meta deck {index + 1}"
        return Matchup(MetaDeck(deck, weight, name), p)
    meta = _as_meta_deck(item, index)
    p = _pick(item, "p_win", None)
    if p is None:
        p = _pick(item, "p", _pick(item, "prob", float("nan")))
    return Matchup(meta, float(p))


def _first_attr(module: Any, role: str):
    for name in _ALIASES[role]:
        fn = getattr(module, name, None)
        if fn is not None:
            return fn
    return None


# --------------------------------------------------------------------------- #
# Builders                                                                      #
# --------------------------------------------------------------------------- #
def _build_heuristic() -> Scorer:
    scorer = Scorer(
        id="heuristic",
        label="Heuristic (baseline)",
        kind="score",
        description=(
            "Hand-tuned weighted average of elixir, air troops, buildings, "
            "spells, HP, DPS and win conditions (optimizer/heuristic.py)."
        ),
    )
    try:
        from optimizer.heuristic import score  # noqa: WPS433 (runtime import on purpose)
    except Exception as exc:  # heuristic.py removed or broken
        scorer.available = False
        scorer.reason = f"{type(exc).__name__}: {exc}"
        return scorer
    scorer.score = score
    return scorer


def _build_learned() -> Scorer:
    scorer = Scorer(
        id="learned",
        label="Learned matchup model",
        kind="winrate",
        description=(
            "Neural matchup model trained on real 1v1 battles. Fitness is the "
            "usage-weighted expected win rate against the current meta decks."
        ),
    )

    module = None
    errors: list[str] = []
    for name in LEARNED_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(name)
            break
        except ModuleNotFoundError as exc:
            # Distinguish "module not written yet" from "its import of torch failed".
            if exc.name in (name, name.rsplit(".", 1)[-1]):
                errors.append(f"{name}: not found")
            else:
                errors.append(f"{name}: {exc}")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if module is None:
        scorer.available = False
        missing_only = all(e.endswith(": not found") for e in errors)
        scorer.reason = (
            "No matchup-model module yet (looked for "
            + ", ".join(LEARNED_MODULE_CANDIDATES)
            + "). Train the model and expose score(deck)."
            if missing_only
            else "Couldn't import the model: " + "; ".join(e for e in errors if not e.endswith(": not found"))
        )
        return scorer

    # Optional self-report: is_available() -> bool | (bool, reason)
    probe = getattr(module, "is_available", None)
    if callable(probe):
        try:
            result = probe()
        except Exception as exc:
            result = (False, f"{type(exc).__name__}: {exc}")
        ok, why = (result if isinstance(result, tuple) else (bool(result), ""))
        if not ok:
            scorer.available = False
            scorer.reason = str(why) or f"{module.__name__}.is_available() returned False"
            return scorer

    kind = getattr(module, "KIND", None)
    if kind in ("winrate", "score"):
        scorer.kind = kind

    score = _first_attr(module, "score")
    if not callable(score):
        scorer.available = False
        scorer.reason = f"{module.__name__} has no score(deck) function"
        return scorer
    scorer.score = score

    predict = _first_attr(module, "predict")
    if callable(predict):
        scorer.predict = predict

    meta_src = _first_attr(module, "meta")
    if meta_src is not None:
        def load_meta() -> list[MetaDeck]:
            raw = meta_src() if callable(meta_src) else meta_src
            return [_as_meta_deck(item, i) for i, item in enumerate(raw or [])]
        scorer.meta = load_meta

    info = _first_attr(module, "info")
    if callable(info):
        scorer.info = info

    provided = _first_attr(module, "matchups")
    if callable(provided):
        def matchups(deck: Deck) -> list[Matchup]:
            return [_as_matchup(item, i) for i, item in enumerate(provided(deck) or [])]
        scorer.matchups = matchups
    elif scorer.predict is not None and scorer.meta is not None:
        def matchups(deck: Deck) -> list[Matchup]:
            return [Matchup(m, float(scorer.predict(deck, m.deck))) for m in scorer.meta()]
        scorer.matchups = matchups

    return scorer


# --------------------------------------------------------------------------- #
# Registry                                                                      #
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_registry: list[Scorer] | None = None


def list_scorers(rescan: bool = False) -> list[Scorer]:
    """All known scorers (available or not), learned model first."""
    global _registry
    with _lock:
        if _registry is None or rescan:
            _registry = [_build_learned(), _build_heuristic()]
        return list(_registry)


def get_scorer(scorer_id: str | None) -> Scorer | None:
    """Scorer by id; None -> the default (learned if available, else heuristic)."""
    scorers = list_scorers()
    if scorer_id:
        for s in scorers:
            if s.id == scorer_id:
                return s
        return None
    return default_scorer()


def default_scorer() -> Scorer | None:
    for s in list_scorers():
        if s.available:
            return s
    return None


def normalized_shares(items: Iterable[MetaDeck]) -> list[float]:
    """Usage weights scaled to sum to 1 (uniform if they're all zero)."""
    weights = [max(0.0, m.weight) for m in items]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights) if weights else []
    return [w / total for w in weights]
