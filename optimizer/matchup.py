"""The learned matchup model as plain module-level functions.

This is the face of the model for callers that just want "rate this deck" --
the web UI's scorer registry (UI/scorers.py) imports it by name and duck-types
these functions:

    is_available() -> (bool, reason)
    score(deck)            GA fitness: expected win rate vs the meta, in [0, 1]
    predict(a, b)          P(deck a beats deck b)
    meta_decks()           [(Deck, weight, label), ...]
    matchups(deck)         [(Deck, weight, P(deck beats it), label), ...]
    KIND = "winrate"

`score.score_batch(decks)` is also attached so GeneticAlgorithm scores a whole
generation in one GPU pass. Everything is built lazily from fitness.make_fitness,
so importing this module never loads torch or the checkpoint by itself.
"""

from __future__ import annotations

from optimizer import config, fitness as _fitness
from optimizer.cr_api import load_card_pool
from optimizer.meta import MetaDeck, deck_from_names, meta_to_deck
from optimizer.models import Deck

KIND = "winrate"

_pool_cache: dict = {}


def is_available() -> tuple[bool, str]:
    return _fitness.model_available()


def _lf():
    """The cached LearnedFitness (rebuilt when the checkpoint changes)."""
    return _fitness.make_fitness("model")


def _pool():
    if "pool" not in _pool_cache:
        _pool_cache["pool"] = load_card_pool()
    return _pool_cache["pool"]


def score(deck: Deck) -> float:
    """Usage-weighted expected win rate of `deck` against the meta decks."""
    return _lf()(deck)


def score_batch(decks: list[Deck]) -> list[float]:
    return _lf().score_batch(decks)


score.score_batch = score_batch  # lets ga.GeneticAlgorithm batch a generation


def predict(a: Deck, b: Deck) -> float:
    """P(a beats b). Exactly 1 - predict(b, a)."""
    lf = _lf()
    a_idx, a_form = lf.encode([a])
    b_idx, b_form = lf.encode([b])
    return float(lf.model.predict_proba(a_idx, a_form, b_idx, b_form)[0])


def meta_decks() -> list[tuple[Deck, float, str]]:
    lf, pool = _lf(), _pool()
    return [(meta_to_deck(m, pool), m.weight, m.label) for m in lf.meta]


def matchups(deck: Deck) -> list[tuple[Deck, float, float, str]]:
    lf, pool = _lf(), _pool()
    return [
        (meta_to_deck(m, pool), weight, p_win, m.label)
        for m, (weight, p_win) in zip(lf.meta, lf.matchups(deck))
    ]


def info() -> dict:
    """Checkpoint / device summary (validation log-loss, #meta decks, ...)."""
    return _lf().info()


def fitness_against(*targets, weights=None):
    """A fitness function that optimises against specific decks instead of the
    ladder meta -- e.g. "the best deck against this one opponent". Targets may
    be Decks, MetaDecks, or strings for meta.deck_from_names ("Golem*, ...").
    Pass it to GeneticAlgorithm; it batches on the GPU like the normal fitness."""
    from optimizer.learned_fitness import LearnedFitness  # lazy: imports torch

    if not targets:
        raise ValueError("give at least one target deck")
    pool = _pool()
    metas = []
    for i, t in enumerate(targets):
        w = 1.0 if weights is None else float(weights[i])
        if isinstance(t, str):
            metas.append(deck_from_names(t, pool, weight=w))
        elif isinstance(t, Deck):
            metas.append(MetaDeck(
                tuple(c.id for c in t.cards), frozenset(t.evolved),
                frozenset(c.id for c in t.cards if c.id in t.hero and not c.is_champion),
                w, ""))
        else:  # MetaDeck
            metas.append(MetaDeck(t.cards, t.evo, t.hero, w, t.label))
    return LearnedFitness(config.MODEL_PATH, config.META_DECKS_CSV, meta=metas)
