"""Tests for the learned matchup fitness. Run from the project root:

    python -m unittest discover -s tests -v

Everything runs on the CPU with a tiny model and synthetic battles, so no real
data or GPU is needed. The slowest test (learning the planted truth) takes
~20-40 s.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from optimizer import config, fitness as fitness_mod  # noqa: E402
from optimizer.battles import (  # noqa: E402
    BattleRow, load_battles, make_synthetic_battles, save_battles,
)
from optimizer.cr_api import load_card_pool  # noqa: E402
from optimizer.ga import GeneticAlgorithm, random_deck  # noqa: E402
from optimizer.heuristic import score as heuristic_score  # noqa: E402
from optimizer.learned_fitness import LearnedFitness  # noqa: E402
from optimizer.meta import (  # noqa: E402
    build_meta_decks, load_meta_decks, meta_to_deck, save_meta_decks,
)
from optimizer.model import (  # noqa: E402
    FORM_BASE, FORM_EVO, FORM_HERO, MatchupModel, encode_battles, evaluate, fit,
)

POOL = load_card_pool()


def tiny_model(**hparams) -> MatchupModel:
    torch.manual_seed(0)
    defaults = dict(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0, card_dropout=0.0)
    return MatchupModel.from_pool(POOL, **{**defaults, **hparams})


def random_pairs(n: int, seed: int = 0):
    rng = random.Random(seed)
    return [random_deck(POOL, rng) for _ in range(n)], [random_deck(POOL, rng) for _ in range(n)]


class ModelStructureTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.a, self.b = random_pairs(16)

    def test_antisymmetric(self):
        m = self.model
        a_idx, a_form = m.encode_decks(self.a)
        b_idx, b_form = m.encode_decks(self.b)
        p_ab = m.predict_proba(a_idx, a_form, b_idx, b_form)
        p_ba = m.predict_proba(b_idx, b_form, a_idx, a_form)
        self.assertLess((p_ab + p_ba - 1).abs().max().item(), 1e-5)

    def test_permutation_invariant(self):
        m = self.model
        a_idx, a_form = m.encode_decks(self.a)
        b_idx, b_form = m.encode_decks(self.b)
        base = m.predict_proba(a_idx, a_form, b_idx, b_form)
        perm = torch.randperm(8, generator=torch.Generator().manual_seed(3))
        shuffled = m.predict_proba(a_idx[:, perm], a_form[:, perm], b_idx, b_form)
        self.assertLess((base - shuffled).abs().max().item(), 1e-5)

    def test_form_changes_output(self):
        m = self.model
        rng = random.Random(5)
        deck = next(d for d in (random_deck(POOL, rng) for _ in range(500)) if d.evolved)
        cards = [c.id for c in deck.cards]
        with_evo = (cards, set(deck.evolved), set())
        without = (cards, set(), set())
        other = self.b[0]
        a1, f1 = m.encode_decks([with_evo, without])
        b_idx, b_form = m.encode_decks([other, other])
        self.assertEqual(f1[0].tolist().count(FORM_EVO), len(deck.evolved))
        self.assertTrue(all(f == FORM_BASE for f in f1[1].tolist()))
        p = m.predict_proba(a1, f1, b_idx, b_form)
        self.assertNotAlmostEqual(p[0].item(), p[1].item(), places=6)

    def test_hero_form_only_with_hero_data(self):
        knight = next(c.id for c in POOL.cards if c.name == "Knight")
        champion = next(iter(POOL.champion_set))
        others = [c.id for c in POOL.cards if c.id not in (knight, champion)][:6]
        deck = ([knight, champion] + others, set(), {knight, champion})

        _, forms = tiny_model(has_hero_data=False).encode_decks([deck])
        self.assertNotIn(FORM_HERO, forms[0].tolist())

        _, forms = tiny_model(has_hero_data=True).encode_decks([deck])
        self.assertEqual(forms[0][0].item(), FORM_HERO)   # Knight in hero form
        self.assertEqual(forms[0][1].item(), FORM_BASE)   # champions are never "hero form"

    def test_unknown_card_raises_or_masks(self):
        deck = ([1, 2, 3, 4, 5, 6, 7, 8], set(), set())
        with self.assertRaises(ValueError):
            self.model.encode_decks([deck])
        idx, _ = self.model.encode_decks([deck], unknown="mask")
        self.assertTrue((idx == self.model.mask_index).all())

    def test_cards_newer_than_model_are_masked_in_fitness(self):
        # Train on a pool missing two cards, then score decks that use them.
        from optimizer.models import CardPool

        missing = [c.id for c in POOL.cards if not c.is_champion][-2:]
        older_pool = CardPool([c for c in POOL.cards if c.id not in missing])
        torch.manual_seed(0)
        model = MatchupModel.from_pool(older_pool, d_model=16, n_heads=2, n_layers=1, d_ff=32)
        rows, _ = make_synthetic_battles(older_pool, 200, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            model_path, meta_path = Path(tmp) / "m.pt", Path(tmp) / "meta.csv"
            model.save(model_path)
            save_meta_decks(build_meta_decks(rows, 10), meta_path)
            lf = LearnedFitness(model_path, meta_path, device="cpu")
            self.assertEqual(lf.unknown_cards, sorted(missing))
            self.assertEqual(lf.info()["unknown_cards"], 2)
            others = [c.id for c in older_pool.cards if not c.is_champion][:6]
            deck = (missing + others, set(), set())
            score = lf.score_batch([deck])[0]
            self.assertTrue(0.0 <= score <= 1.0)

    def test_save_load_roundtrip(self):
        m = self.model
        a_idx, a_form = m.encode_decks(self.a)
        b_idx, b_form = m.encode_decks(self.b)
        before = m.predict_proba(a_idx, a_form, b_idx, b_form)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            m.save(path, meta={"val_logloss": 0.5})
            loaded = MatchupModel.load(path)
        self.assertEqual(loaded.meta["val_logloss"], 0.5)
        self.assertEqual(loaded.card_ids, m.card_ids)
        after = loaded.predict_proba(a_idx, a_form, b_idx, b_form)
        self.assertLess((before - after).abs().max().item(), 1e-6)


class DataTests(unittest.TestCase):
    def test_battles_csv_roundtrip_and_drop(self):
        rows, _ = make_synthetic_battles(POOL, 50, seed=1)
        bad = BattleRow("bad", "2026-01-01T00:00:00+00:00", (1, 2, 3, 4, 5, 6, 7, 8),
                        frozenset(), frozenset(), rows[0].b_cards, frozenset(),
                        frozenset(), False, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "battles.csv"
            save_battles(rows + [bad], path)
            loaded, dropped = load_battles(path, POOL)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(loaded), len(rows))
        self.assertEqual(loaded[0].a_cards, tuple(sorted(rows[0].a_cards)))
        self.assertEqual(loaded[0].a_evo, rows[0].a_evo)
        self.assertEqual(loaded[0].result, rows[0].result)
        self.assertFalse(loaded[0].hero_known)

    def test_meta_build_save_load(self):
        rows, _ = make_synthetic_battles(POOL, 300, seed=2)
        meta = build_meta_decks(rows, 20, POOL)
        self.assertEqual(len(meta), 20)
        self.assertAlmostEqual(sum(m.weight for m in meta), 1.0, places=6)
        self.assertTrue(meta[0].label)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.csv"
            save_meta_decks(meta, path)
            loaded = load_meta_decks(path, set(POOL.by_id))
        self.assertEqual([m.key for m in loaded], [m.key for m in meta])
        self.assertAlmostEqual(sum(m.weight for m in loaded), 1.0, places=6)
        deck = meta_to_deck(loaded[0], POOL)
        self.assertEqual(len(deck.cards), config.DECK_SIZE)
        for card in deck.champions:
            self.assertIn(card.id, deck.hero)

    def test_deck_from_names(self):
        from optimizer.meta import deck_from_names

        md = deck_from_names("Goblin Barrel*, Tombstone^, Golden Knight, Princess, "
                             "Dark Prince, barbarian barrel, Fireball, Baby Dragon", POOL)
        names = {POOL.get(c).name for c in md.cards}
        self.assertEqual(len(md.cards), 8)
        self.assertIn("Barbarian Barrel", names)  # case-insensitive
        self.assertEqual([POOL.get(c).name for c in md.evo], ["Goblin Barrel"])
        self.assertEqual([POOL.get(c).name for c in md.hero], ["Tombstone"])
        self.assertNotIn(next(c for c in md.cards if POOL.get(c).name == "Golden Knight"), md.hero)
        with self.assertRaises(ValueError) as ctx:
            deck_from_names("Goblin Barel, Tombstone, Golden Knight, Princess, "
                            "Dark Prince, Barbarian Barrel, Fireball, Baby Dragon", POOL)
        self.assertIn("Goblin Barrel", str(ctx.exception))  # suggestion
        with self.assertRaises(ValueError):
            deck_from_names("Knight, Archers", POOL)  # not 8 cards


class LearningTests(unittest.TestCase):
    def test_learns_planted_truth(self):
        rows, truth = make_synthetic_battles(POOL, 8000, seed=3)
        split = int(len(rows) * 0.9)
        model = tiny_model(d_model=32, n_heads=4, d_ff=64, card_dropout=0.05)
        train = encode_battles(model, rows[:split])
        val = encode_battles(model, rows[split:])
        baseline = evaluate(model, val)["logloss"]
        fit(model, train, val, epochs=15, patience=4, batch_size=512, log=None)
        result = evaluate(model, val)
        self.assertLess(result["logloss"], baseline - 0.15)

        pred = model.predict_proba(*val[:4])
        true = torch.tensor([truth.win_prob(r.a, r.b) for r in rows[split:]])
        self.assertGreater(_corr(pred, true), 0.6)

        # Planted counters: A holding x vs B holding y should beat the mirror
        # (A holding y vs B holding x). Each pair co-occurs only ~35 times in
        # 8k uniform-random battles, so judge the aggregate over all pairs.
        rng = random.Random(9)
        per_pair = []
        for x, y, _bonus in truth.counters:
            diffs = []
            for _ in range(60):
                a, b = _context(rng, exclude={x, y}), _context(rng, exclude={x, y})
                fwd = ((a[0] + [x]), a[1], a[2]), ((b[0] + [y]), b[1], b[2])
                mir = ((a[0] + [y]), a[1], a[2]), ((b[0] + [x]), b[1], b[2])
                a_idx, a_form = model.encode_decks([fwd[0], mir[0]])
                b_idx, b_form = model.encode_decks([fwd[1], mir[1]])
                with torch.inference_mode():
                    logits = model(a_idx, a_form, b_idx, b_form)
                diffs.append((logits[0] - logits[1]).item())
            per_pair.append(sum(diffs) / len(diffs))
        self.assertGreater(sum(per_pair) / len(per_pair), 0.2)
        self.assertGreaterEqual(sum(d > 0 for d in per_pair), len(per_pair) // 2)


def _context(rng, exclude):
    """A 7-card partial deck (cards, evo, hero) avoiding `exclude`."""
    deck = random_deck(POOL, rng)
    cards = [c.id for c in deck.cards if c.id not in exclude][:7]
    while len(cards) < 7:
        c = rng.choice(POOL.all_ids)
        if c not in cards and c not in exclude:
            cards.append(c)
    return cards, set(deck.evolved) & set(cards), set()


def _corr(x, y) -> float:
    x, y = x.float().cpu() - x.float().cpu().mean(), y.float() - y.float().mean()
    return ((x * y).sum() / math.sqrt((x ** 2).sum().item() * (y ** 2).sum().item())).item()


class GaIntegrationTests(unittest.TestCase):
    def test_batch_path_matches_single_path(self):
        calls: list[int] = []

        class Batched:
            def score_batch(self, decks):
                calls.append(len(decks))
                return [heuristic_score(d) for d in decks]

            def __call__(self, deck):
                raise AssertionError("single path should not be used when batching")

        kwargs = dict(population_size=30, generations=5, seed=11)
        best_batched = GeneticAlgorithm(POOL, Batched(), **kwargs).run()
        best_single = GeneticAlgorithm(POOL, heuristic_score, **kwargs).run()
        self.assertGreaterEqual(len(calls), 5)  # one batched call per generation
        self.assertTrue(all(0 < n <= 30 for n in calls))
        self.assertEqual(best_batched.key, best_single.key)
        self.assertTrue(best_batched.is_valid()[0])

    def test_learned_fitness_end_to_end(self):
        rows, _ = make_synthetic_battles(POOL, 1500, seed=4)
        model = tiny_model()
        fit(model, encode_battles(model, rows), epochs=2, log=None)
        meta = build_meta_decks(rows, 25, POOL)
        with tempfile.TemporaryDirectory() as tmp:
            model_path, meta_path = Path(tmp) / "m.pt", Path(tmp) / "meta.csv"
            model.save(model_path, meta={"val_logloss": 0.6})
            save_meta_decks(meta, meta_path)

            lf = LearnedFitness(model_path, meta_path, device="cpu")
            decks, _ = random_pairs(12, seed=7)
            scores = lf.score_batch(decks)
            self.assertEqual(len(scores), 12)
            self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
            self.assertAlmostEqual(lf(decks[0]), scores[0], places=5)
            matchups = lf.matchups(decks[0])
            self.assertEqual(len(matchups), 25)
            self.assertAlmostEqual(sum(w * p for w, p in matchups), scores[0], places=5)
            self.assertEqual(lf.info()["meta_decks"], 25)

            best = GeneticAlgorithm(POOL, lf, population_size=20, generations=3, seed=1).run()
            self.assertTrue(best.is_valid()[0])

            # A custom meta (one target deck): fitness == P(deck beats target).
            target = meta[0]
            single = LearnedFitness(model_path, meta_path, meta=[target], device="cpu")
            self.assertEqual(len(single.meta), 1)
            a_idx, a_form = single.encode([decks[0]])
            t_idx, t_form = single.encode([target])
            direct = single.model.predict_proba(a_idx, a_form, t_idx, t_form)[0].item()
            self.assertAlmostEqual(single(decks[0]), direct, places=5)
            with self.assertRaises(ValueError):
                LearnedFitness(model_path, meta_path, meta=[], device="cpu")

            # The selector picks it up from config paths and caches by mtime.
            old = config.MODEL_PATH, config.META_DECKS_CSV
            config.MODEL_PATH, config.META_DECKS_CSV = model_path, meta_path
            try:
                self.assertEqual(fitness_mod.model_available(), (True, ""))
                fn = fitness_mod.make_fitness("model")
                self.assertIs(fn, fitness_mod.make_fitness("model"))
                self.assertEqual(fitness_mod.describe("model")["kind"], "model")
                config.MODEL_PATH = Path(tmp) / "missing.pt"
                ok, reason = fitness_mod.model_available()
                self.assertFalse(ok)
                self.assertIn("missing.pt", reason)
                with self.assertRaises(RuntimeError):
                    fitness_mod.make_fitness("model")
            finally:
                config.MODEL_PATH, config.META_DECKS_CSV = old
                fitness_mod._cache.update(key=None, fitness=None)

        self.assertIs(fitness_mod.make_fitness("heuristic"), heuristic_score)


if __name__ == "__main__":
    unittest.main()
