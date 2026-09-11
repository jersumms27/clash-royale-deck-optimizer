"""Battle outcomes -- the training data for the learned matchup model (stdlib only).

File format: data/battles.csv, documented in data/README.md. One row per 1v1
battle with both decks (cards + evo/hero forms) and the result from A's side.
`make_synthetic_battles` fabricates data with the same schema from a planted
"truth" so the whole pipeline can be exercised before real data exists.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from optimizer import config
from optimizer.cr_api import _to_bool
from optimizer.ga import random_deck
from optimizer.models import CardPool


def parse_id_list(text: str | None) -> list[int]:
    """'26000000|26000001' -> [26000000, 26000001]; blank -> []."""
    if text is None:
        return []
    return [int(part) for part in str(text).split("|") if part.strip()]


def format_id_list(ids) -> str:
    return "|".join(str(i) for i in sorted(ids))


@dataclass(frozen=True)
class BattleRow:
    battle_id: str
    battle_time: str  # ISO-8601; only used for ordering (time-based split)
    a_cards: tuple[int, ...]
    a_evo: frozenset[int]
    a_hero: frozenset[int]  # non-champions played in hero form
    b_cards: tuple[int, ...]
    b_evo: frozenset[int]
    b_hero: frozenset[int]
    hero_known: bool  # False -> the source didn't expose hero form at all
    result: float  # 1.0 A won, 0.0 B won, 0.5 draw
    game_mode: str = ""

    @property
    def a(self) -> tuple[tuple[int, ...], frozenset[int], frozenset[int]]:
        return self.a_cards, self.a_evo, self.a_hero

    @property
    def b(self) -> tuple[tuple[int, ...], frozenset[int], frozenset[int]]:
        return self.b_cards, self.b_evo, self.b_hero


CSV_FIELDS = ["battle_id", "battle_time", "game_mode", "a_cards", "a_evo", "a_hero",
              "b_cards", "b_evo", "b_hero", "hero_known", "result"]


def load_battles(
    path=config.BATTLES_CSV, pool: CardPool | None = None
) -> tuple[list[BattleRow], int]:
    """Read battles.csv. Returns (rows, dropped) where `dropped` counts rows
    skipped for referencing a card id not in `pool` or for a malformed deck."""
    known = set(pool.by_id) if pool is not None else None
    rows: list[BattleRow] = []
    dropped = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            try:
                row = _row_from_csv(raw)
            except (KeyError, ValueError):
                dropped += 1
                continue
            if (
                len(row.a_cards) != config.DECK_SIZE
                or len(row.b_cards) != config.DECK_SIZE
                or (known is not None and not (set(row.a_cards) | set(row.b_cards)) <= known)
            ):
                dropped += 1
                continue
            rows.append(row)
    return rows, dropped


def _row_from_csv(raw: dict) -> BattleRow:
    a_cards = tuple(parse_id_list(raw["a_cards"]))
    b_cards = tuple(parse_id_list(raw["b_cards"]))
    return BattleRow(
        battle_id=str(raw.get("battle_id") or ""),
        battle_time=str(raw.get("battle_time") or ""),
        a_cards=a_cards,
        a_evo=frozenset(parse_id_list(raw.get("a_evo"))) & set(a_cards),
        a_hero=frozenset(parse_id_list(raw.get("a_hero"))) & set(a_cards),
        b_cards=b_cards,
        b_evo=frozenset(parse_id_list(raw.get("b_evo"))) & set(b_cards),
        b_hero=frozenset(parse_id_list(raw.get("b_hero"))) & set(b_cards),
        hero_known=_to_bool(raw.get("hero_known", "")),
        result=float(raw["result"]),
        game_mode=str(raw.get("game_mode") or ""),
    )


def save_battles(rows: list[BattleRow], path=config.BATTLES_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "battle_id": r.battle_id,
                    "battle_time": r.battle_time,
                    "game_mode": r.game_mode,
                    "a_cards": format_id_list(r.a_cards),
                    "a_evo": format_id_list(r.a_evo),
                    "a_hero": format_id_list(r.a_hero),
                    "b_cards": format_id_list(r.b_cards),
                    "b_evo": format_id_list(r.b_evo),
                    "b_hero": format_id_list(r.b_hero),
                    "hero_known": r.hero_known,
                    "result": r.result,
                }
            )


# --------------------------------------------------------------------------- #
# Synthetic data                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticTruth:
    """The hidden model that generated a synthetic dataset (for tests/eval).

    Deck strength = sum of card strengths (+ evo/hero bonuses, + synergy bonus
    when both cards of a synergy pair are in the deck). A beats B with
    probability sigmoid((S_A - S_B + counters) / temperature), where each
    counter pair (x, y, bonus) adds `bonus` when A holds x and B holds y (and
    subtracts it in the mirrored case). Counter pairs share a strength so the
    counter effect is isolated from raw card strength.
    """

    strength: dict[int, float]
    synergies: list[tuple[int, int, float]] = field(default_factory=list)
    counters: list[tuple[int, int, float]] = field(default_factory=list)
    evo_bonus: float = 0.4
    hero_bonus: float = 0.0
    temperature: float = 1.0

    def deck_strength(self, cards, evo, hero) -> float:
        in_deck = set(cards)
        total = sum(self.strength[c] for c in cards)
        total += self.evo_bonus * len(set(evo) & in_deck)
        total += self.hero_bonus * len(set(hero) & in_deck)
        total += sum(b for x, y, b in self.synergies if x in in_deck and y in in_deck)
        return total

    def logit(self, a, b) -> float:
        """True log-odds that deck a (cards, evo, hero) beats deck b."""
        diff = self.deck_strength(*a) - self.deck_strength(*b)
        a_set, b_set = set(a[0]), set(b[0])
        for x, y, bonus in self.counters:
            if x in a_set and y in b_set:
                diff += bonus
            if x in b_set and y in a_set:
                diff -= bonus
        return diff / self.temperature

    def win_prob(self, a, b) -> float:
        return 1.0 / (1.0 + math.exp(-self.logit(a, b)))


def make_synthetic_battles(
    pool: CardPool,
    n: int,
    seed: int = 0,
    *,
    hero_known: bool = False,
    n_synergies: int = 6,
    n_counters: int = 6,
    pair_bonus: float = 2.0,
    draw_rate: float = 0.03,
) -> tuple[list[BattleRow], SyntheticTruth]:
    """Fabricate `n` battles between random legal decks from a planted truth.

    Decks come from ga.random_deck so forms obey the real slot rules. With
    hero_known=False (the default, matching data whose source can't see hero
    form) the hero bonus is zero and hero columns are left blank.
    """
    rng = random.Random(seed)
    non_champ = [c.id for c in pool.cards if not c.is_champion]
    truth = SyntheticTruth(
        strength={c.id: rng.gauss(0.0, 1.0) for c in pool.cards},
        hero_bonus=0.5 if hero_known else 0.0,
    )
    for _ in range(n_synergies):
        x, y = rng.sample(non_champ, 2)
        truth.synergies.append((x, y, pair_bonus))
    for _ in range(n_counters):
        x, y = rng.sample(non_champ, 2)
        truth.strength[y] = truth.strength[x]  # isolate the counter effect
        truth.counters.append((x, y, pair_bonus))

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[BattleRow] = []
    for i in range(n):
        da, db = random_deck(pool, rng), random_deck(pool, rng)
        a = (tuple(c.id for c in da.cards), da.evolved, _non_champion_heroes(da))
        b = (tuple(c.id for c in db.cards), db.evolved, _non_champion_heroes(db))
        if rng.random() < draw_rate:
            result = 0.5
        else:
            result = 1.0 if rng.random() < truth.win_prob(a, b) else 0.0
        rows.append(
            BattleRow(
                battle_id=f"synthetic-{seed}-{i}",
                battle_time=(start + timedelta(minutes=i)).isoformat(),
                a_cards=a[0],
                a_evo=frozenset(a[1]),
                a_hero=frozenset(a[2]) if hero_known else frozenset(),
                b_cards=b[0],
                b_evo=frozenset(b[1]),
                b_hero=frozenset(b[2]) if hero_known else frozenset(),
                hero_known=hero_known,
                result=result,
                game_mode="synthetic",
            )
        )
    return rows, truth


def _non_champion_heroes(deck) -> frozenset[int]:
    return frozenset(c.id for c in deck.cards if c.id in deck.hero and not c.is_champion)
