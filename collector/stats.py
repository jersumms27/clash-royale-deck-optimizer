"""Sanity report on the raw crawl and the flattened dataset.

    python -m collector.stats                 # data/battles/raw.jsonl + data/battles.csv
    python -m collector.stats --csv-only      # skip the (slower) raw scan

Answers: how much do we have, from when, how balanced are the labels, which
modes got in, how many rows would the model's loader drop (unknown card ids),
and how much the top players dominate (battles per player).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from optimizer import config

from collector.flatten import parse_battle_time
from collector.store import CSV_PATH, RAW_PATH, iter_raw, sides


def known_card_ids(path: Path = config.CARDS_CSV) -> set[int]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(row["id"]) for row in csv.DictReader(fh)}


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "  n/a"


def _quantile(sorted_values: list, q: float):
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


# --------------------------------------------------------------------------- #
def raw_report(path: Path, top: int) -> None:
    print(f"=== raw: {path} ===")
    if not Path(path).exists():
        print("  (missing)")
        return
    bad = 0

    def on_bad(_n: int) -> None:
        nonlocal bad
        bad += 1

    n = dups = 0
    ids: set[str] = set()
    modes: Counter = Counter()
    deck_sel: Counter = Counter()
    per_player: Counter = Counter()
    fetched_from: Counter = Counter()
    days: Counter = Counter()
    t_min = t_max = None
    for b in iter_raw(path, on_bad=on_bad):
        n += 1
        bid = b.get("battle_id", "")
        if bid in ids:
            dups += 1
            continue
        ids.add(bid)
        modes[(b.get("type", "?"), str((b.get("gameMode") or {}).get("name", "?")))] += 1
        deck_sel[b.get("deckSelection", "?")] += 1
        fetched_from[b.get("fetched_from", "?")] += 1
        team, opp = sides(b)
        for side in team + opp:
            per_player[side.get("tag", "?")] += 1
        bt = str(b.get("battleTime", ""))
        if bt:
            days[bt[:8]] += 1
            t_min = bt if t_min is None or bt < t_min else t_min
            t_max = bt if t_max is None or bt > t_max else t_max

    print(f"  battles: {n}   unreadable lines: {bad}   duplicate ids: {dups}")
    if not n:
        return
    print(f"  time range: {parse_battle_time(t_min)} .. {parse_battle_time(t_max)}  ({len(days)} days)")
    print(f"  players appearing: {len(per_player)}   players crawled: {len(fetched_from)}")
    counts = sorted(per_player.values())
    print(f"  battles per player: p50 {_quantile(counts, 0.5)}  p90 {_quantile(counts, 0.9)}  "
          f"p99 {_quantile(counts, 0.99)}  max {counts[-1]}")
    print(f"  deckSelection: {dict(deck_sel)}")
    print(f"  (type, gameMode.name) -- top {top}:")
    for (t, m), c in modes.most_common(top):
        print(f"    {c:>8} {_pct(c, n)}  {t} / {m}")
    if len(days) > 1:
        print("  battles per day (last 10):")
        for day in sorted(days)[-10:]:
            print(f"    {day[:4]}-{day[4:6]}-{day[6:]}  {days[day]}")


def csv_report(path: Path, cards_csv: Path, top: int) -> None:
    print(f"\n=== csv: {path} ===")
    if not Path(path).exists():
        print("  (missing -- run `python -m collector.flatten`)")
        return
    try:
        known = known_card_ids(cards_csv)
    except FileNotFoundError:
        known = set()
        print(f"  ({cards_csv} missing; can't check card ids)")

    n = 0
    result_sum = 0.0
    results: Counter = Counter()
    result_src: Counter = Counter()
    game_mode: Counter = Counter()
    maxed_by_mode: Counter = Counter()
    hero_known: Counter = Counter()
    evo_rows = hero_rows = 0
    evo_cards: Counter = Counter()
    hero_cards: Counter = Counter()
    slot_usage: Counter = Counter()  # (n_evo, n_hero) per deck
    towers: Counter = Counter()
    card_usage: Counter = Counter()
    unknown_ids: Counter = Counter()
    rows_with_unknown = 0
    t_min = t_max = None
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        for row in reader:
            n += 1
            r = float(row["result"])
            result_sum += r
            results[row["result"]] += 1
            result_src[row.get("result_src", "?")] += 1
            game_mode[row.get("game_mode", "?")] += 1
            hero_known[row.get("hero_known", "?")] += 1
            bt = row.get("battle_time", "")
            t_min = bt if t_min is None or bt < t_min else t_min
            t_max = bt if t_max is None or bt > t_max else t_max
            bad = False
            for prefix in ("a", "b"):
                ids = [int(x) for x in row[f"{prefix}_cards"].split("|") if x]
                for cid in ids:
                    card_usage[cid] += 1
                    if known and cid not in known:
                        unknown_ids[cid] += 1
                        bad = True
                evo = [x for x in row.get(f"{prefix}_evo", "").split("|") if x]
                hero = [x for x in row.get(f"{prefix}_hero", "").split("|") if x]
                evo_cards.update(int(x) for x in evo)
                hero_cards.update(int(x) for x in hero)
                evo_rows += bool(evo)
                hero_rows += bool(hero)
                slot_usage[(len(evo), len(hero))] += 1
                if row.get(f"{prefix}_tower"):
                    towers[row[f"{prefix}_tower"]] += 1
            if bad:
                rows_with_unknown += 1
            levels = row.get("a_levels", "") + "|" + row.get("b_levels", "")
            if levels.strip("|") and all(x == "0" for x in levels.split("|") if x):
                maxed_by_mode[row.get("game_mode", "?")] += 1

    print(f"  rows: {n}   columns: {len(columns)}")
    if not n:
        return
    print(f"  time range: {t_min} .. {t_max}")
    print(f"  result mean: {result_sum / n:.4f}   counts: {dict(sorted(results.items()))}")
    print(f"  result source: {dict(result_src.most_common())}")
    print(f"  game_mode: " + ", ".join(f"{m} {c} ({_pct(c, n).strip()})" for m, c in game_mode.most_common()))
    print("  fully maxed (all 16 cards at max level): "
          + ", ".join(f"{m} {_pct(maxed_by_mode[m], c).strip()}" for m, c in game_mode.most_common()))
    print(f"  decks with an evolution: {_pct(evo_rows, 2 * n).strip()}   decks with a (non-champion) hero: "
          f"{_pct(hero_rows, 2 * n).strip()}   hero_known: {dict(hero_known)}")
    print("  (evo, hero) slots used per deck: "
          + ", ".join(f"{k[0]}+{k[1]}: {_pct(c, 2 * n).strip()}" for k, c in sorted(slot_usage.items())))
    print(f"  tower troops: {dict(towers.most_common())}")
    if known:
        print(f"  rows the loader will drop (unknown card id): {rows_with_unknown}"
              + (f"  ids: {dict(unknown_ids.most_common(10))}" if unknown_ids else ""))
        unused = sorted(known - set(card_usage))
        if unused:
            print(f"  cards.csv ids never seen in a deck: {len(unused)} {unused[:10]}{' ...' if len(unused) > 10 else ''}")
    print(f"  most used cards (share of decks) -- top {top}:")
    names = _card_names(cards_csv)
    for cid, c in card_usage.most_common(top):
        print(f"    {_pct(c, 2 * n)}  {names.get(cid, cid)}")
    for label, counter in (("evolutions", evo_cards), ("heroes", hero_cards)):
        if counter:
            print(f"  most used {label} -- top 10:")
            for cid, c in counter.most_common(10):
                print(f"    {c:>8}  {names.get(cid, cid)}")


def _card_names(path: Path) -> dict[int, str]:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return {int(r["id"]): r["name"] for r in csv.DictReader(fh)}
    except FileNotFoundError:
        return {}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Report on the crawl and the dataset")
    parser.add_argument("--in", dest="raw", default=str(RAW_PATH))
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--cards", default=str(config.CARDS_CSV))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.csv_only:
        raw_report(Path(args.raw), args.top)
    csv_report(Path(args.csv), Path(args.cards), args.top)


if __name__ == "__main__":
    main()
