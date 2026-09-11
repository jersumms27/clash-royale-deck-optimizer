"""raw.jsonl -> data/battles.csv, the dataset the matchup model trains on.

Schema is the contract in data/README.md (read by optimizer.battles.load_battles):

    battle_id, battle_time, game_mode, a_cards, a_evo, a_hero, b_cards, b_evo,
    b_hero, hero_known, result, a_trophies, b_trophies, a_tower, b_tower

followed by extra columns the loader ignores (kept for filtering/auditing):

    game_mode_name, league, a_levels, b_levels, a_crowns, b_crowns, result_src

One row per battle. Which player becomes "A" is a deterministic coin flip on
battle_id, so the file has ~50/50 labels and carries no trace of which side
the crawler happened to fetch the battle from. Re-running gives an identical
file. Mirrored rows are NOT written -- the trainer swaps sides itself.

    python -m collector.flatten                       # defaults: PoL + ladder, normal rules
    python -m collector.flatten --maxed-only          # only battles where all 16 cards are max level
    python -m collector.flatten --since 20260901      # battles from September 2026 on
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from collector.api import bare_tag
from collector.store import CSV_PATH, RAW_PATH, iter_raw, keep_battle, sides

# How the battle log encodes a card's form. There is no separate hero field:
# the API reuses `evolutionLevel`, and the value says which form was played.
#   evolutionLevel 1 -> evolved     (cards with an evolution have maxEvolutionLevel 1 or 3)
#   evolutionLevel 2 -> hero form   (hero-eligible cards have maxEvolutionLevel 2 or 3)
#   absent / 0       -> base form
# Verified on ~27k live battles: every card seen at level 2 is a hero-eligible
# card, cards with only an evolution never show 2, and champions (implicit
# heroes) carry no evolutionLevel at all. Re-check with `python -m collector.probe`
# if Supercell changes the encoding.
EVO_LEVEL = 1
HERO_LEVEL = 2
HERO_KNOWN = True  # the log does expose hero form (via the encoding above)

CONTRACT_COLUMNS = [
    "battle_id", "battle_time", "game_mode",
    "a_cards", "a_evo", "a_hero", "b_cards", "b_evo", "b_hero",
    "hero_known", "result", "a_trophies", "b_trophies", "a_tower", "b_tower",
]
EXTRA_COLUMNS = ["game_mode_name", "league", "a_levels", "b_levels", "a_crowns", "b_crowns", "result_src"]
COLUMNS = CONTRACT_COLUMNS + EXTRA_COLUMNS

# Battle `type` values that are normal-rules 1v1, as seen on the live API:
#   pathOfLegend  ranked (Ranked1v1_NewArena2)      -- ~90% fully maxed decks
#   trail         trophy-road ladder ("Ladder")     -- level gaps are common
#   riverRacePvP  clan war 1v1 (CW_Battle_1v1)      -- tournament-standard levels
#   PvP           the pre-2026 name for ladder (kept in case it comes back)
# Special-rules variants of these (7xElixir_Ladder, Touchdown_ClanWar, ...) are
# removed by DEFAULT_MODE_DENY. Restrict with e.g. `--types pathOfLegend`.
DEFAULT_TYPES = ("pathOfLegend", "trail", "riverRacePvP", "PvP")
# A gameMode.name containing any of these (case-insensitive) is a special-rules
# event, not the matchup we're modelling. Tune after looking at `stats`.
DEFAULT_MODE_DENY = (
    "rage", "triple", "double", "elixir", "sudden", "mirror", "draft", "dragon", "ramp", "heist",
    "touchdown", "capture", "lightning", "freeze", "dash", "rune", "clone", "tornado", "snowball",
    "recruit", "retro", "showdown", "duel", "infinite", "7x", "3x", "2x", "pick", "spooky", "mega",
)


@dataclass
class Options:
    types: set[str] = field(default_factory=lambda: set(DEFAULT_TYPES))
    mode_deny: tuple[str, ...] = DEFAULT_MODE_DENY
    mode_allow: set[str] = field(default_factory=set)  # exact gameMode names that bypass the deny list
    since: str = ""  # YYYYMMDD, inclusive
    until: str = ""  # YYYYMMDD, inclusive
    maxed_only: bool = False
    drop_draws: bool = False


@dataclass
class Summary:
    read: int = 0
    bad_lines: int = 0
    duplicates: int = 0
    not_kept: int = 0  # fails keep_battle (shouldn't happen for crawler output)
    written: int = 0
    excluded: Counter = field(default_factory=Counter)  # reason -> count
    excluded_modes: Counter = field(default_factory=Counter)  # (type, gameMode.name) -> count
    result_src: Counter = field(default_factory=Counter)
    result_sum: float = 0.0

    def report(self) -> str:
        lines = [f"read {self.read} battles ({self.bad_lines} unreadable lines, {self.duplicates} duplicate ids)"]
        for reason, n in self.excluded.most_common():
            lines.append(f"  excluded {n:>7}  {reason}")
        if self.excluded_modes:
            lines.append("  excluded (type, gameMode):")
            for (t, m), n in self.excluded_modes.most_common(40):
                lines.append(f"    {n:>7}  {t} / {m}")
        lines.append(f"wrote {self.written} rows")
        if self.written:
            lines.append(f"  mean result {self.result_sum / self.written:.3f}  "
                         f"(sources: {dict(self.result_src.most_common())})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Per-battle logic                                                             #
# --------------------------------------------------------------------------- #
def side_flip(bid: str) -> bool:
    """Deterministic coin flip: True -> the opponent becomes side A."""
    return bool(hashlib.blake2b(bid.encode("utf-8"), digest_size=1).digest()[0] & 1)


def parse_battle_time(s: str) -> str:
    """'20260911T140300.000Z' -> '2026-09-11T14:03:00Z' (unchanged if unparsable)."""
    try:
        dt = datetime.strptime(s, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return str(s)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def lowest_tower_hp(side: dict) -> int | None:
    """HP of the weakest *standing* tower (the in-game tiebreaker)."""
    hps = [side.get("kingTowerHitPoints")] + list(side.get("princessTowersHitPoints") or [])
    hps = [h for h in hps if isinstance(h, (int, float))]
    return int(min(hps)) if hps else None


def outcome(a: dict, b: dict) -> tuple[float, str]:
    """Result from A's perspective and which rule decided it.

    crowns  -> the normal case
    trophy  -> equal crowns but trophyChange shows a winner (ladder only)
    hp      -> equal crowns; in-game tiebreaker is the lowest tower's HP
    draw    -> genuinely tied
    """
    ca, cb = int(a.get("crowns") or 0), int(b.get("crowns") or 0)
    if ca != cb:
        return (1.0 if ca > cb else 0.0), "crowns"
    ta, tb = a.get("trophyChange"), b.get("trophyChange")
    if isinstance(ta, (int, float)) and isinstance(tb, (int, float)) and (ta > 0) != (tb > 0) and (ta or tb):
        return (1.0 if ta > tb else 0.0), "trophy"
    ha, hb = lowest_tower_hp(a), lowest_tower_hp(b)
    if ha is not None and hb is not None and ha != hb:
        return (1.0 if ha > hb else 0.0), "hp"
    return 0.5, "draw"


def form_of(card: dict) -> str:
    """'evo', 'hero' or 'base' -- see the encoding notes at the top of the file."""
    level = card.get("evolutionLevel") or 0
    if level == EVO_LEVEL:
        return "evo"
    if level == HERO_LEVEL and str(card.get("rarity", "")).lower() != "champion":
        return "hero"  # champions are heroes by definition; the contract leaves them out
    return "base"


def deck_fields(side: dict) -> dict:
    cards = sorted(side.get("cards") or [], key=lambda c: int(c["id"]))
    levels = []
    for c in cards:
        lvl, mx = c.get("level"), c.get("maxLevel")
        levels.append(str(int(mx) - int(lvl)) if isinstance(lvl, int) and isinstance(mx, int) else "")
    support = side.get("supportCards") or []
    forms = {int(c["id"]): form_of(c) for c in cards}
    return {
        "cards": "|".join(str(c["id"]) for c in cards),
        "evo": "|".join(str(c["id"]) for c in cards if forms[int(c["id"])] == "evo"),
        "hero": "|".join(str(c["id"]) for c in cards if forms[int(c["id"])] == "hero"),
        "tower": str(support[0].get("name", "")) if support else "",
        "trophies": str(side["startingTrophies"]) if isinstance(side.get("startingTrophies"), int) else "",
        "crowns": str(side.get("crowns", "")),
        "levels": "|".join(levels),
        "maxed": bool(levels) and all(lv == "0" for lv in levels),
    }


def exclusion_reason(b: dict, opts: Options) -> str | None:
    btype = b.get("type", "")
    mode = str((b.get("gameMode") or {}).get("name", ""))
    if opts.types and btype not in opts.types:
        return "type"
    if mode not in opts.mode_allow and any(k in mode.lower() for k in opts.mode_deny):
        return "mode"
    day = str(b.get("battleTime", ""))[:8]
    if opts.since and day < opts.since:
        return "before --since"
    if opts.until and day > opts.until:
        return "after --until"
    return None


def battle_row(b: dict, opts: Options) -> tuple[list[str] | None, str | None]:
    """(row, None) or (None, reason it was excluded)."""
    reason = exclusion_reason(b, opts)
    if reason:
        return None, reason
    team, opp = sides(b)
    # Canonical order by tag first, so the same battle stored from either
    # player's log lands on the same A/B assignment; then the coin flip.
    p, q = sorted((team[0], opp[0]), key=lambda s: bare_tag(s.get("tag", "")))
    a, bb = (q, p) if side_flip(b["battle_id"]) else (p, q)
    fa, fb = deck_fields(a), deck_fields(bb)
    if opts.maxed_only and not (fa["maxed"] and fb["maxed"]):
        return None, "not maxed"
    result, src = outcome(a, bb)
    if src == "draw" and opts.drop_draws:
        return None, "draw"
    row = [
        b["battle_id"], parse_battle_time(b.get("battleTime", "")), b.get("type", ""),
        fa["cards"], fa["evo"], fa["hero"], fb["cards"], fb["evo"], fb["hero"],
        str(HERO_KNOWN), f"{result:.1f}", fa["trophies"], fb["trophies"], fa["tower"], fb["tower"],
        # extras
        str((b.get("gameMode") or {}).get("name", "")),
        str(b.get("leagueNumber", "")),
        fa["levels"], fb["levels"], fa["crowns"], fb["crowns"], src,
    ]
    return row, None


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def flatten(raw_path: Path = RAW_PATH, out_path: Path = CSV_PATH, opts: Options | None = None) -> Summary:
    opts = opts or Options()
    summary = Summary()
    seen: set[str] = set()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out_path).with_suffix(".csv.tmp")

    def on_bad(_line_no: int) -> None:
        summary.bad_lines += 1

    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(COLUMNS)
        for b in iter_raw(raw_path, on_bad=on_bad):
            summary.read += 1
            bid = b.get("battle_id")
            if not bid:
                summary.bad_lines += 1
                continue
            if bid in seen:
                summary.duplicates += 1
                continue
            seen.add(bid)
            if not keep_battle(b):
                summary.not_kept += 1
                continue
            row, reason = battle_row(b, opts)
            if row is None:
                summary.excluded[reason] += 1
                if reason in ("type", "mode"):
                    summary.excluded_modes[(b.get("type", ""), str((b.get("gameMode") or {}).get("name", "")))] += 1
                continue
            writer.writerow(row)
            summary.written += 1
            summary.result_src[row[COLUMNS.index("result_src")]] += 1
            summary.result_sum += float(row[COLUMNS.index("result")])
    tmp.replace(out_path)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Flatten raw.jsonl into battles.csv")
    parser.add_argument("--in", dest="raw", default=str(RAW_PATH))
    parser.add_argument("--out", default=str(CSV_PATH))
    parser.add_argument("--types", default=",".join(DEFAULT_TYPES),
                        help="battle types to keep, comma-separated ('' = all)")
    parser.add_argument("--mode-deny", default=",".join(DEFAULT_MODE_DENY),
                        help="drop gameMode names containing any of these keywords ('' = none)")
    parser.add_argument("--mode-allow", default="", help="exact gameMode names that bypass --mode-deny")
    parser.add_argument("--since", default="", help="YYYYMMDD inclusive")
    parser.add_argument("--until", default="", help="YYYYMMDD inclusive")
    parser.add_argument("--maxed-only", action="store_true", help="only battles where all 16 cards are max level")
    parser.add_argument("--drop-draws", action="store_true", help="drop draws instead of writing result=0.5")
    args = parser.parse_args(argv)

    opts = Options(
        types={t.strip() for t in args.types.split(",") if t.strip()},
        mode_deny=tuple(k.strip().lower() for k in args.mode_deny.split(",") if k.strip()),
        mode_allow={m.strip() for m in args.mode_allow.split(",") if m.strip()},
        since=args.since, until=args.until, maxed_only=args.maxed_only, drop_draws=args.drop_draws,
    )
    summary = flatten(Path(args.raw), Path(args.out), opts)
    print(summary.report())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
