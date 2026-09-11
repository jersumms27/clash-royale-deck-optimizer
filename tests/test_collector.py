"""Offline tests for the battle collector (no API token needed). From the project root:

    python -m unittest tests.test_collector -v

Fixtures are hand-built battle dicts in the shape the API returns, written
through the same RawWriter/slim_battle path the crawler uses.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer import config  # noqa: E402

from collector import crawl, flatten  # noqa: E402
from collector.api import NotFound, RateLimiter, bare_tag, norm_tag, tag_path  # noqa: E402
from collector.store import (  # noqa: E402
    RawWriter, battle_id, iter_raw, keep_battle, participant_tags, scan_ids, slim_battle,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
with open(config.CARDS_CSV, newline="", encoding="utf-8") as _fh:
    _ROWS = list(csv.DictReader(_fh))
KNOWN_IDS = sorted(int(r["id"]) for r in _ROWS)
DECK_A = KNOWN_IDS[:8]
DECK_B = KNOWN_IDS[8:16]
UNKNOWN_ID = 26999999


def card(cid, level=15, max_level=15, evo=None, rarity="common", **extra):
    c = {"name": f"card{cid}", "id": cid, "level": level, "maxLevel": max_level,
         "rarity": rarity, "elixirCost": 3, "iconUrls": {"medium": "https://x/y.png"}}
    if evo is not None:
        c["evolutionLevel"] = evo
        c["maxEvolutionLevel"] = 1
    c.update(extra)
    return c


def side(tag, cards, crowns, trophies=None, trophy_change=None, king=4000,
         princess=(2500, 2500), tower="Princess Tower"):
    s = {"tag": tag, "name": "player", "crowns": crowns, "kingTowerHitPoints": king,
         "princessTowersHitPoints": list(princess), "cards": cards}
    if trophies is not None:
        s["startingTrophies"] = trophies
    if trophy_change is not None:
        s["trophyChange"] = trophy_change
    if tower:
        s["supportCards"] = [{"name": tower, "id": 159000000, "level": 15, "maxLevel": 15,
                              "iconUrls": {"medium": "https://x/t.png"}}]
    return s


def battle(when, team, opp, type="pathOfLegend", mode="Ranked1v1_NewArena2",
           deck_selection="collection", **extra):
    b = {"type": type, "battleTime": when, "arena": {"id": 54000050, "name": "Legendary Arena"},
         "gameMode": {"id": 72000323, "name": mode}, "deckSelection": deck_selection,
         "team": [team] if isinstance(team, dict) else team,
         "opponent": [opp] if isinstance(opp, dict) else opp}
    b.update(extra)
    return b


def deck(ids, **kw):
    return [card(cid, **kw) for cid in ids]


def write_raw(path, battles_with_source):
    w = RawWriter(path)
    for b, src in battles_with_source:
        w.append(slim_battle(b, src))
    w.close()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def quiet_flatten(raw, out, opts=None):
    with redirect_stdout(io.StringIO()):
        return flatten.flatten(Path(raw), Path(out), opts)


# --------------------------------------------------------------------------- #
class TagTests(unittest.TestCase):
    def test_normalisation(self):
        self.assertEqual(norm_tag(" #2pp0o "), "#2PP00")
        self.assertEqual(norm_tag("2PP"), "#2PP")
        self.assertEqual(bare_tag("#2PP"), "2PP")
        self.assertEqual(tag_path("#2PP"), "%232PP")
        with self.assertRaises(ValueError):
            norm_tag("#")
        with self.assertRaises(ValueError):
            norm_tag("#2P P")


class RateLimiterTests(unittest.TestCase):
    def test_throughput_is_bounded(self):
        limiter = RateLimiter(rate=50, burst=1)
        n_threads, per_thread = 4, 25

        def work():
            for _ in range(per_thread):
                limiter.acquire()

        threads = [threading.Thread(target=work) for _ in range(n_threads)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - t0
        # 100 acquires at 50/s -> ~2 s. Generous bounds for CI/Windows timers.
        self.assertGreater(elapsed, 1.5)
        self.assertLess(elapsed, 4.0)

    def test_penalize_pauses_everyone(self):
        limiter = RateLimiter(rate=1000, burst=10)
        limiter.penalize(0.5)
        t0 = time.monotonic()
        limiter.acquire()
        self.assertGreaterEqual(time.monotonic() - t0, 0.45)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.a = side("#AAA", deck(DECK_A), 3)
        self.b = side("#BBB", deck(DECK_B), 1)
        self.battle = battle("20260910T120000.000Z", self.a, self.b)

    def test_keep_battle(self):
        self.assertTrue(keep_battle(self.battle))
        self.assertFalse(keep_battle(battle("20260910T120000.000Z", [self.a, self.a], [self.b, self.b])))
        self.assertFalse(keep_battle(battle("20260910T120000.000Z", self.a, self.b, deck_selection="draft")))
        short = dict(self.b, cards=self.b["cards"][:7])
        self.assertFalse(keep_battle(battle("20260910T120000.000Z", self.a, short)))
        nocards = {k: v for k, v in self.b.items() if k != "cards"}
        self.assertFalse(keep_battle(battle("20260910T120000.000Z", self.a, nocards)))

    def test_battle_id_is_symmetric_and_hashless(self):
        from_a = battle("20260910T120000.000Z", self.a, self.b)
        from_b = battle("20260910T120000.000Z", self.b, self.a)
        self.assertEqual(battle_id(from_a), battle_id(from_b))
        self.assertEqual(battle_id(from_a), "20260910T120000.000Z_AAA_BBB")
        self.assertNotIn("#", battle_id(from_a))

    def test_slim_strips_icons_and_puts_id_first(self):
        slim = slim_battle(self.battle, "#AAA")
        self.assertEqual(list(slim)[:2], ["battle_id", "fetched_from"])
        for s in slim["team"] + slim["opponent"]:
            for c in s["cards"] + s["supportCards"]:
                self.assertNotIn("iconUrls", c)
        # the original is untouched
        self.assertIn("iconUrls", self.battle["team"][0]["cards"][0])

    def test_participant_tags(self):
        self.assertEqual(participant_tags(self.battle, exclude="#AAA"), ["#BBB"])
        self.assertEqual(participant_tags(self.battle), ["#AAA", "#BBB"])

    def test_raw_writer_recovers_from_torn_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            w = RawWriter(path)
            w.append(slim_battle(self.battle, "#AAA"))
            w.close()
            with open(path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write('{"battle_id":"torn","fetched_from":"#X","team":[')  # crash mid-line
            w = RawWriter(path)
            other = battle("20260910T130000.000Z", self.a, self.b)
            w.append(slim_battle(other, "#AAA"))
            w.close()
            ids, bad = scan_ids(path)
            self.assertEqual(bad, 1)
            self.assertEqual(ids, {battle_id(self.battle), battle_id(other)})
            bad_lines = []
            got = list(iter_raw(path, on_bad=bad_lines.append))
            self.assertEqual(len(got), 2)
            self.assertEqual(bad_lines, [2])


class FlattenTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.t = "20260910T120000.000Z"

    def tearDown(self):
        self.tmpdir.cleanup()

    def run_flatten(self, battles_with_source, opts=None, name="raw.jsonl"):
        raw = self.tmp / name
        out = self.tmp / (name + ".csv")
        write_raw(raw, battles_with_source)
        summary = quiet_flatten(raw, out, opts)
        return summary, read_csv(out), out

    def test_columns_and_contract_values(self):
        a = side("#AAA", deck(DECK_A), 3, trophies=9000, tower="Cannoneer")
        b = side("#BBB", deck(DECK_B), 1, trophies=8990)
        summary, rows, _ = self.run_flatten([(battle(self.t, a, b, type="PvP", mode="Ladder"), "#AAA")])
        self.assertEqual(summary.written, 1)
        row = rows[0]
        self.assertEqual(list(row), flatten.COLUMNS)
        self.assertEqual(row["battle_id"], "20260910T120000.000Z_AAA_BBB")
        self.assertEqual(row["battle_time"], "2026-09-10T12:00:00Z")
        self.assertEqual(row["game_mode"], "PvP")
        self.assertEqual(row["game_mode_name"], "Ladder")
        self.assertEqual(row["hero_known"], "True")
        self.assertIn(row["result"], ("1.0", "0.0"))
        # both decks present, A/B assignment consistent with the crowns
        decks = {row["a_cards"], row["b_cards"]}
        self.assertEqual(decks, {"|".join(map(str, DECK_A)), "|".join(map(str, DECK_B))})
        a_is_aaa = row["a_cards"] == "|".join(map(str, DECK_A))
        self.assertEqual(row["result"], "1.0" if a_is_aaa else "0.0")
        self.assertEqual(row["a_trophies"], "9000" if a_is_aaa else "8990")
        self.assertEqual(row["a_tower"], "Cannoneer" if a_is_aaa else "Princess Tower")
        self.assertEqual(row["a_crowns"], "3" if a_is_aaa else "1")
        self.assertEqual(row["result_src"], "crowns")
        self.assertEqual(row["a_levels"], "|".join(["0"] * 8))

    def test_same_battle_from_both_logs_gives_one_identical_row(self):
        a = side("#AAA", deck(DECK_A), 2)
        b = side("#BBB", deck(DECK_B), 0)
        from_a = battle(self.t, a, b)
        from_b = battle(self.t, b, a)
        s1, rows1, out1 = self.run_flatten([(from_a, "#AAA"), (from_b, "#BBB")], name="one.jsonl")
        s2, rows2, out2 = self.run_flatten([(from_b, "#BBB"), (from_a, "#AAA")], name="two.jsonl")
        self.assertEqual((s1.written, s1.duplicates), (1, 1))
        self.assertEqual((s2.written, s2.duplicates), (1, 1))
        self.assertEqual(out1.read_bytes(), out2.read_bytes())
        # re-running is byte-identical too
        quiet_flatten(self.tmp / "one.jsonl", self.tmp / "again.csv")
        self.assertEqual(hashlib.sha256(out1.read_bytes()).digest(),
                         hashlib.sha256((self.tmp / "again.csv").read_bytes()).digest())

    def test_result_rules(self):
        t = "20260910T12000{}.000Z"
        cases = [
            # (team, opponent, expected src)
            (side("#AAA", deck(DECK_A), 1, trophy_change=30), side("#BBB", deck(DECK_B), 1, trophy_change=-30), "trophy"),
            (side("#AAA", deck(DECK_A), 1, king=1200, princess=(2000,)), side("#BBB", deck(DECK_B), 1, king=400, princess=(2000,)), "hp"),
            (side("#AAA", deck(DECK_A), 0), side("#BBB", deck(DECK_B), 0), "draw"),
        ]
        battles = [(battle(t.format(i), a, b, type="PvP", mode="Ladder"), "#AAA") for i, (a, b, _) in enumerate(cases)]
        summary, rows, _ = self.run_flatten(battles)
        self.assertEqual(summary.written, 3)
        by_src = {r["result_src"]: r for r in rows}
        self.assertEqual(set(by_src), {"trophy", "hp", "draw"})
        for r in rows:
            aaa_is_a = r["a_cards"] == "|".join(map(str, DECK_A))
            if r["result_src"] == "draw":
                self.assertEqual(r["result"], "0.5")
            else:  # #AAA won in both decided cases
                self.assertEqual(r["result"], "1.0" if aaa_is_a else "0.0")
        summary, rows, _ = self.run_flatten(battles, flatten.Options(drop_draws=True), name="nodraw.jsonl")
        self.assertEqual(summary.written, 2)
        self.assertEqual(summary.excluded["draw"], 1)

    def test_filters(self):
        a, b = side("#AAA", deck(DECK_A), 3), side("#BBB", deck(DECK_B), 0)
        t = "202609{:02d}T120000.000Z"
        battles = [
            (battle(t.format(1), a, b), "#AAA"),                                   # kept
            (battle(t.format(2), a, b, type="trail", mode="Rage_Ladder"), "#AAA"),   # mode deny
            (battle(t.format(3), a, b, type="friendly", mode="Friendly"), "#AAA"),   # type
            (battle(t.format(4), a, b, type="trail", mode="Ladder_CrownRush"), "#AAA"),  # kept (reward-only modifier)
            (battle(t.format(5), a, b, type="riverRacePvP", mode="CW_Battle_1v1"), "#AAA"),  # kept
            (battle(t.format(6), a, b, type="riverRacePvP", mode="7xElixir_Ladder"), "#AAA"),  # mode deny
        ]
        summary, rows, _ = self.run_flatten(battles)
        self.assertEqual(summary.written, 3)
        self.assertEqual(summary.excluded["mode"], 2)
        self.assertEqual(summary.excluded["type"], 1)
        self.assertEqual(summary.excluded_modes[("trail", "Rage_Ladder")], 1)
        self.assertEqual({r["game_mode"] for r in rows}, {"pathOfLegend", "trail", "riverRacePvP"})
        # --types narrows; --mode-allow rescues an exact name; --since/--until slice by day
        summary, rows, _ = self.run_flatten(battles, flatten.Options(types={"pathOfLegend"}), name="pol.jsonl")
        self.assertEqual(summary.written, 1)
        summary, rows, _ = self.run_flatten(battles, flatten.Options(mode_allow={"Rage_Ladder"}), name="allow.jsonl")
        self.assertEqual(summary.written, 4)
        summary, rows, _ = self.run_flatten(battles, flatten.Options(since="20260902", until="20260903"), name="dates.jsonl")
        self.assertEqual(summary.written, 0)
        self.assertEqual(summary.excluded["before --since"], 1)
        self.assertEqual(summary.excluded["after --until"], 2)  # days 4 and 5 (day 6 is a mode deny)

    def test_skips_junk_without_crashing(self):
        a, b = side("#AAA", deck(DECK_A), 3), side("#BBB", deck(DECK_B), 0)
        two_v_two = battle(self.t, [a, a], [b, b], type="PvP2v2")
        nocards = battle("20260910T130000.000Z", a, {k: v for k, v in b.items() if k != "cards"})
        raw = self.tmp / "junk.jsonl"
        # bypass slim_battle's battle_id for the junk, as a crawler bug would
        w = RawWriter(raw)
        w.append({"battle_id": "junk1", "fetched_from": "#AAA", **two_v_two})
        w.append({"battle_id": "junk2", "fetched_from": "#AAA", **nocards})
        w.append(slim_battle(battle("20260910T140000.000Z", a, b), "#AAA"))
        w.close()
        summary = quiet_flatten(raw, self.tmp / "junk.csv")
        self.assertEqual(summary.written, 1)
        self.assertEqual(summary.not_kept, 2)

    def test_deck_fields(self):
        cards_a = deck(DECK_A)
        cards_a[0] = card(DECK_A[0], evo=0)          # not evolved
        cards_a[1] = card(DECK_A[1], evo=1)          # evolved
        cards_a[2] = card(DECK_A[2], level=14)       # one below max
        cards_a[3] = card(DECK_A[3], level=5, max_level=5, rarity="champion")
        a = side("#AAA", cards_a, 3)
        b = side("#BBB", deck(DECK_B), 0, tower=None)
        summary, rows, _ = self.run_flatten([(battle(self.t, a, b), "#AAA")])
        row = rows[0]
        p = "a" if row["a_cards"] == "|".join(map(str, DECK_A)) else "b"
        q = "b" if p == "a" else "a"
        self.assertEqual(row[f"{p}_evo"], str(DECK_A[1]))
        self.assertEqual(row[f"{q}_evo"], "")
        self.assertEqual(row[f"{q}_tower"], "")
        self.assertEqual(row[f"{p}_hero"], "")
        levels = row[f"{p}_levels"].split("|")
        self.assertEqual(levels[DECK_A.index(DECK_A[2])], "1")
        self.assertEqual(levels.count("0"), 7)
        # cards are sorted ascending, and levels follow that order
        ids = [int(x) for x in row[f"{p}_cards"].split("|")]
        self.assertEqual(ids, sorted(ids))

    def test_maxed_only(self):
        cards_a = deck(DECK_A)
        cards_a[0] = card(DECK_A[0], level=13)
        a = side("#AAA", cards_a, 3)
        b = side("#BBB", deck(DECK_B), 0)
        summary, rows, _ = self.run_flatten([(battle(self.t, a, b), "#AAA")], flatten.Options(maxed_only=True))
        self.assertEqual(summary.written, 0)
        self.assertEqual(summary.excluded["not maxed"], 1)

    def test_side_flip_is_balanced_and_stable(self):
        ids = [f"2026091{i % 10}T{i:06d}.000Z_A_B{i}" for i in range(20000)]
        mean = sum(flatten.side_flip(x) for x in ids) / len(ids)
        self.assertAlmostEqual(mean, 0.5, delta=0.01)
        self.assertEqual(flatten.side_flip("20260910T120000.000Z_AAA_BBB"),
                         flatten.side_flip("20260910T120000.000Z_AAA_BBB"))

    def test_hero_form_is_evolution_level_2(self):
        cards_a = deck(DECK_A)
        cards_a[0] = card(DECK_A[0], evo=2)                     # hero form
        cards_a[1] = card(DECK_A[1], evo=2, rarity="champion")  # champions are implicit heroes: left out
        cards_a[2] = card(DECK_A[2], evo=1)                     # evolved
        cards_a[3] = card(DECK_A[3], evo=0)                     # base
        a = side("#AAA", cards_a, 3)
        b = side("#BBB", deck(DECK_B), 0)
        summary, rows, _ = self.run_flatten([(battle(self.t, a, b), "#AAA")])
        row = rows[0]
        p = "a" if row["a_cards"] == "|".join(map(str, DECK_A)) else "b"
        self.assertEqual(row[f"{p}_hero"], str(DECK_A[0]))
        self.assertEqual(row[f"{p}_evo"], str(DECK_A[2]))
        self.assertEqual(row["hero_known"], "True")
        self.assertEqual(flatten.form_of(card(1, evo=2)), "hero")
        self.assertEqual(flatten.form_of(card(1, evo=1)), "evo")
        self.assertEqual(flatten.form_of(card(1)), "base")
        self.assertEqual(flatten.form_of(card(1, evo=2, rarity="champion")), "base")


class LoaderRoundTripTests(unittest.TestCase):
    """The CSV must load through the model side's loader without drops."""

    def setUp(self):
        try:
            from optimizer.battles import load_battles  # noqa: F401
            from optimizer.cr_api import load_card_pool  # noqa: F401
        except ImportError as exc:  # the optimizer side isn't there yet
            self.skipTest(f"optimizer loader unavailable: {exc}")

    def test_loads_with_zero_dropped(self):
        from optimizer.battles import load_battles
        from optimizer.cr_api import load_card_pool

        pool = load_card_pool()
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw.jsonl", Path(tmp) / "battles.csv"
            a, b = side("#AAA", deck(DECK_A), 3), side("#BBB", deck(DECK_B), 0)
            c = side("#CCC", deck(DECK_B[:7] + [UNKNOWN_ID]), 1)
            write_raw(raw, [
                (battle("20260910T120000.000Z", a, b), "#AAA"),
                (battle("20260910T130000.000Z", a, c), "#AAA"),  # unknown id -> loader drops it
            ])
            quiet_flatten(raw, out)
            rows, dropped = load_battles(out, pool=pool)
            self.assertEqual(len(rows), 1)
            self.assertEqual(dropped, 1)
            row = rows[0]
            self.assertEqual(len(row.a_cards), 8)
            self.assertEqual(len(row.b_cards), 8)
            self.assertIn(row.result, (0.0, 1.0))
            self.assertTrue(row.hero_known)
            self.assertEqual(row.game_mode, "pathOfLegend")


class FakeClient:
    """Stands in for collector.api.Client: canned battle logs, no network."""

    def __init__(self, logs):
        self.logs = logs
        self.base = "fake://"
        self.requests = 0

    def battlelog(self, tag):
        self.requests += 1
        if tag not in self.logs:
            raise NotFound(f"404 {tag}", 404)
        return self.logs[tag]


class CrawlLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)
        a, b, c = (side("#AAA", deck(DECK_A), 3), side("#BBB", deck(DECK_B), 1), side("#CCC", deck(DECK_B), 2))
        ab = battle("20260910T120000.000Z", a, b)
        ba = battle("20260910T120000.000Z", b, a)  # same battle, B's log
        bc = battle("20260910T121000.000Z", b, c)
        two = battle("20260910T122000.000Z", [a, a], [c, c], type="PvP2v2")
        self.fake = FakeClient({"#AAA": [ab, two], "#BBB": [ba, bc]})  # #CCC -> 404
        self._orig = (crawl.make_client, crawl._install_sigint)
        crawl.make_client = lambda **kw: self.fake
        crawl._install_sigint = lambda stop: None

    def tearDown(self):
        crawl.make_client, crawl._install_sigint = self._orig
        self.tmpdir.cleanup()

    def run_crawl(self, *extra):
        args = crawl.build_parser().parse_args([
            "--data-dir", str(self.data_dir), "--no-seed", "--workers", "2", "--rps", "1000",
            "--save-every", "1", "--exit-when-idle", *extra,
        ])
        with redirect_stdout(io.StringIO()) as buf:
            rc = crawl.run(args)
        return rc, buf.getvalue()

    def test_snowball_dedup_and_resume(self):
        rc, out = self.run_crawl("--seed-tags", "#AAA")
        self.assertEqual(rc, 0, out)
        ids, bad = scan_ids(self.data_dir / "raw.jsonl")
        self.assertEqual(bad, 0)
        self.assertEqual(ids, {"20260910T120000.000Z_AAA_BBB", "20260910T121000.000Z_BBB_CCC"})
        self.assertEqual(self.fake.requests, 3)  # A, B (snowballed), C (404)
        from collector.store import load_state
        state = load_state(self.data_dir / "state.json")
        self.assertEqual(set(state["players"]), {"#AAA", "#BBB", "#CCC"})
        self.assertEqual(state["queue"], [])
        self.assertEqual(state["totals"]["kept"], 2)

        # second run: everyone is fresh, nothing is fetched, nothing duplicated
        rc, out = self.run_crawl()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.fake.requests, 3)
        ids2, _ = scan_ids(self.data_dir / "raw.jsonl")
        self.assertEqual(ids2, ids)
        self.assertEqual(sum(1 for _ in iter_raw(self.data_dir / "raw.jsonl")), 2)

        # once the recrawl window has passed everyone is re-polled; still no duplicates
        from collector.store import save_state
        state["players"] = {tag: 0.0 for tag in state["players"]}  # "fetched at the epoch"
        save_state(state, self.data_dir / "state.json")
        rc, out = self.run_crawl()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.fake.requests, 6)
        self.assertEqual(sum(1 for _ in iter_raw(self.data_dir / "raw.jsonl")), 2)

    def test_max_players_limit(self):
        rc, out = self.run_crawl("--seed-tags", "#AAA", "--max-players", "1")
        self.assertEqual(rc, 0)
        self.assertEqual(self.fake.requests, 1)
        self.assertIn("--max-players 1 reached", out)


if __name__ == "__main__":
    unittest.main()
