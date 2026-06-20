"""Your deck-scoring heuristic. The GA maximizes score(deck); higher = better.

Deck aggregates: deck.cards, deck.avg_elixir, deck.champions, deck.evolved_cards,
deck.troops, deck.buildings, deck.spells, deck.air_troops, deck.win_conditions,
deck.primary_win_conditions, deck.card_ids. The deck passed in is always legal.

Card fields:
  core:  .name .elixir .rarity .type ("troop"/"building"/"spell") .has_evolution
         .is_champion .is_champion_hero .air
         .win_condition ("primary"/"secondary"/"conditional"/"")
         .spell_size ("small"/"medium"/"large"/"")
  stats (float or None -- spells have no hitpoints, etc., so guard for None):
         .hitpoints .damage .damage_per_second .attack_period .range .radius
         .lifetime .crown_tower_damage .special_damage
  spawn (str or None):  .troop_spawned .spawn_count_period .max_troops_spawned
  evo (float/str or None):  .evo_cycles .evo_overall_cost .evo_stat_boosts
"""

from __future__ import annotations

from models import Deck

# --- Weights: how much each raw metric matters. They all sit in [0, 1] and sum
# to 1.0, so a deck that scores a perfect 1.0 on every metric gets a final 1.0.
# A metric whose data isn't available yet is dropped from the average and its
# weight is redistributed across the rest, so the score always stays in [0, 1].
WEIGHTS = {
    "avg_elixir": 0.15,
    "air_troops": 0.10,
    "buildings": 0.10,
    "total_hp": 0.15,
    "total_dps": 0.15,
    "spells": 0.15,
    "win_conditions": 0.20,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"

# --- Calibration targets. These are rough starting points; tune them against
# the real stat ranges once cards.csv is populated by scrape.ipynb. ---
IDEAL_AVG_ELIXIR = 3.8     # the "sweet spot" average elixir cost
ELIXIR_SPREAD = 2.0        # how far from ideal before the metric hits 0

IDEAL_BUILDINGS = 1.0      # ~1 defensive building is typical
BUILDINGS_SPREAD = 2.5

IDEAL_SPELLS = 2.0         # ~2 spells is typical
SPELLS_SPREAD = 3.0

IDEAL_AIR_TROOPS = 2.0     # ~1-2 air units helps cover anti-air threats
AIR_SPREAD = 3.0

TARGET_TOTAL_HP = 20_000.0   # deck total HP that earns a full HP score
TARGET_TOTAL_DPS = 3_000.0   # deck total DPS that earns a full DPS score

IDEAL_WIN_SCORE = 1.5      # weighted win-condition presence we aim for
WIN_SPREAD = 1.5

# How much each win-condition tier counts toward the weighted win score.
WIN_TIER_WEIGHTS = {"primary": 1.0, "secondary": 0.6, "conditional": 0.3}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _triangular(value: float, ideal: float, spread: float) -> float:
    """Peaked score: 1.0 at `ideal`, falling linearly to 0 at `spread` away."""
    return _clamp01(1.0 - abs(value - ideal) / spread)


def _saturating(value: float, target: float) -> float:
    """More-is-better: 0 at 0, rising linearly to 1.0 at `target`, capped there."""
    if target <= 0:
        return 0.0
    return _clamp01(value / target)


# --- Per-metric normalizers. Each maps a raw deck value into [0, 1], or returns
# None when the underlying data isn't available yet (so the metric is skipped). ---

def _norm_avg_elixir(deck: Deck) -> float | None:
    return _triangular(deck.avg_elixir, IDEAL_AVG_ELIXIR, ELIXIR_SPREAD)


def _norm_air_troops(deck: Deck) -> float | None:
    return _triangular(len(deck.air_troops), IDEAL_AIR_TROOPS, AIR_SPREAD)


def _norm_buildings(deck: Deck) -> float | None:
    return _triangular(len(deck.buildings), IDEAL_BUILDINGS, BUILDINGS_SPREAD)


def _norm_total_hp(deck: Deck) -> float | None:
    stats = [c.hitpoints for c in deck.cards if c.hitpoints is not None]
    if not stats:
        return None  # scraped combat stats not loaded into cards.csv yet
    return _saturating(sum(stats), TARGET_TOTAL_HP)


def _norm_total_dps(deck: Deck) -> float | None:
    stats = [c.damage_per_second for c in deck.cards if c.damage_per_second is not None]
    if not stats:
        return None  # scraped combat stats not loaded into cards.csv yet
    return _saturating(sum(stats), TARGET_TOTAL_DPS)


def _norm_spells(deck: Deck) -> float | None:
    return _triangular(len(deck.spells), IDEAL_SPELLS, SPELLS_SPREAD)


def _norm_win_conditions(deck: Deck) -> float | None:
    # Weight each win condition by its tier (primary > secondary > conditional),
    # then peak the score around IDEAL_WIN_SCORE: too few means no way to take a
    # tower, too many means no elixir left for support cards.
    win_score = sum(
        WIN_TIER_WEIGHTS.get(c.win_condition, 0.0) for c in deck.cards
    )
    return _triangular(win_score, IDEAL_WIN_SCORE, WIN_SPREAD)


_NORMALIZERS = {
    "avg_elixir": _norm_avg_elixir,
    "air_troops": _norm_air_troops,
    "buildings": _norm_buildings,
    "total_hp": _norm_total_hp,
    "total_dps": _norm_total_dps,
    "spells": _norm_spells,
    "win_conditions": _norm_win_conditions,
}


def score(deck: Deck) -> float:
    # Weighted average of the normalized metrics. Metrics that return None are
    # skipped and their weight is redistributed, so the result stays in [0, 1].
    weighted_sum = 0.0
    total_weight = 0.0
    for name, normalize in _NORMALIZERS.items():
        normalized = normalize(deck)
        if normalized is None:
            continue
        weight = WEIGHTS[name]
        weighted_sum += weight * normalized
        total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight
