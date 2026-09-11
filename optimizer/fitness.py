"""Pick the fitness function by name: "heuristic" (heuristic.py) or "model"
(learned_fitness.py). Used by main.py and the web UI. Torch is only imported
when the model is actually requested, so the heuristic path stays stdlib-only.
"""

from __future__ import annotations

import importlib.util
import os

from optimizer import config, heuristic

CHOICES = ("heuristic", "model")

_cache: dict = {"key": None, "fitness": None}


def model_available() -> tuple[bool, str]:
    """(True, "") if the learned fitness can be built, else (False, why not)."""
    if importlib.util.find_spec("torch") is None:
        return False, "PyTorch is not installed (see environment.yml)."
    if not config.MODEL_PATH.exists():
        return False, (f"No trained model at {config.MODEL_PATH.name} -- "
                       "run notebooks/train_model.ipynb.")
    if not config.META_DECKS_CSV.exists():
        return False, (f"No {config.META_DECKS_CSV.name} -- "
                       "run notebooks/train_model.ipynb.")
    return True, ""


def make_fitness(name: str = config.FITNESS):
    """Return a callable deck -> float. The learned fitness is cached and
    rebuilt only when the model or meta file changes on disk."""
    if name == "heuristic":
        return heuristic.score
    if name != "model":
        raise ValueError(f"unknown fitness {name!r}; choose from {CHOICES}")

    ok, reason = model_available()
    if not ok:
        raise RuntimeError(reason)
    key = (os.path.getmtime(config.MODEL_PATH), os.path.getmtime(config.META_DECKS_CSV))
    if _cache["key"] != key:
        from optimizer.learned_fitness import LearnedFitness  # lazy: imports torch

        _cache["fitness"] = LearnedFitness(config.MODEL_PATH, config.META_DECKS_CSV)
        _cache["key"] = key
    return _cache["fitness"]


def describe(name: str) -> dict:
    """Small summary for UIs: which fitness, and how to read its numbers."""
    if name == "model":
        fn = make_fitness("model")
        return {"kind": "model", "label": "expected win rate", **fn.info()}
    return {"kind": "heuristic", "label": "heuristic score"}
