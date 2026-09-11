"""Fetch one player's battle log and print its structure.

Run this first: it shows exactly which fields the API exposes today (evolution
level, tower troop, any hero-form field, trophies vs. league), which is what
flatten.py's column mapping depends on.

    python -m collector.probe                 # a top Path of Legend player
    python -m collector.probe "#ABC123"       # a specific player
    python -m collector.probe --index 3       # expand the 4th 1v1 battle instead
    python -m collector.probe --raw           # dump the chosen battle's full JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from collector.api import ApiError, NotFound, make_client
from collector.flatten import form_of
from collector.store import is_1v1, keep_battle, sides


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Inspect one CR battle log")
    parser.add_argument("tag", nargs="?", help="player tag (default: a top PoL player)")
    parser.add_argument("--index", type=int, default=0, help="which 1v1 battle to expand")
    parser.add_argument("--raw", action="store_true", help="also dump the battle's full JSON")
    parser.add_argument("--rps", type=float, default=2.0)
    args = parser.parse_args(argv)

    # Player names can contain anything; don't let the Windows console choke.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    client = make_client(rps=args.rps)
    print(f"API base: {client.base}")

    tag = args.tag
    if not tag:
        tags = client.pol_players("global", limit=5)
        if not tags:
            raise SystemExit("global Path of Legend ranking is empty right now; pass a tag.")
        tag = tags[0]
        print(f"No tag given; using top PoL player {tag}")

    try:
        battles = client.battlelog(tag)
    except NotFound:
        raise SystemExit(f"{tag}: player not found (404).")
    except ApiError as exc:
        raise SystemExit(str(exc))

    print(f"\n{tag}: {len(battles)} battles in the log\n")
    print(f"{'#':>2}  {'type':<16} {'gameMode.name':<28} {'deckSel':<11} {'sides':<6} {'crowns':<7} battleTime")
    for i, b in enumerate(battles):
        team, opp = sides(b)
        crowns = f"{_crowns(team)}-{_crowns(opp)}"
        mode = (b.get("gameMode") or {}).get("name", "?")
        flag = "*" if keep_battle(b) else " "
        print(f"{i:>2}{flag} {b.get('type', '?'):<16} {mode:<28} {b.get('deckSelection', '?'):<11} "
              f"{len(team)}v{len(opp):<4} {crowns:<7} {b.get('battleTime', '?')}")
    print("\n(* = passes keep_battle: 1v1, collection deck, 8 cards per side)")

    kept = [b for b in battles if keep_battle(b)]
    if not kept:
        print("\nNo 1v1 collection-deck battles in this log; try another tag.")
        return
    b = kept[min(args.index, len(kept) - 1)]
    team, opp = sides(b)
    me, them = team[0], opp[0]

    print("\n=== Battle structure ===")
    print("top-level keys:      ", sorted(b.keys()))
    print("gameMode:            ", b.get("gameMode"))
    print("arena:               ", b.get("arena"))
    print("leagueNumber:        ", b.get("leagueNumber", "<absent>"))
    print("side keys (team[0]): ", sorted(k for k in me.keys() if k not in ("cards", "supportCards")))
    for label, side in (("team", me), ("opponent", them)):
        print(f"  {label:<9} startingTrophies={side.get('startingTrophies', '<absent>')} "
              f"trophyChange={side.get('trophyChange', '<absent>')} crowns={side.get('crowns')} "
              f"king={side.get('kingTowerHitPoints', '<absent>')} "
              f"princess={side.get('princessTowersHitPoints', '<absent>')}")

    cards = (me.get("cards") or []) + (them.get("cards") or [])
    key_union = Counter(k for c in cards for k in c.keys())
    print("\ncard keys (count across all 16 cards):")
    for k, n in sorted(key_union.items()):
        print(f"  {k:<20} {n:>2}")
    hero_like = [k for k in key_union if "hero" in k.lower()]
    if hero_like:
        print("hero-looking card keys:", hero_like, "<- NEW; check flatten.form_of still matches")
    print("\ncard forms (flatten.form_of decodes evolutionLevel 1 = evo, 2 = hero):")
    for c in cards:
        lvl = c.get("evolutionLevel")
        if lvl:
            print(f"  {c['name']:<20} evolutionLevel={lvl} maxEvolutionLevel={c.get('maxEvolutionLevel')} "
                  f"-> {form_of(c)}")
    if not any(c.get("evolutionLevel") for c in cards):
        print("  (no evolved or hero cards in this battle)")

    print("\none card object:")
    print(json.dumps(_strip(cards[0]), indent=2))

    support = me.get("supportCards") or []
    print("\nsupportCards[0] (tower troop):")
    print(json.dumps(_strip(support[0]), indent=2) if support else "  <absent>")

    non_1v1 = [b2 for b2 in battles if not is_1v1(b2)]
    print(f"\n{len(non_1v1)} non-1v1 battles in the log; "
          f"{len(battles) - len(kept)} battles fail keep_battle overall.")

    if args.raw:
        print("\n=== full battle JSON ===")
        print(json.dumps(b, indent=2))


def _crowns(side: list[dict]) -> str:
    return str(side[0].get("crowns", "?")) if side else "?"


def _strip(card: dict) -> dict:
    return {k: v for k, v in card.items() if k != "iconUrls"}


if __name__ == "__main__":
    main()
