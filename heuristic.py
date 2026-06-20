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

# How much each metric counts. They sum to 1.0, so a deck that scores a perfect
# 1.0 on every metric gets a final 1.0.
WEIGHTS = {
    "avg_elixir": 0.10,
    "air_troops": 0.10,
    "buildings": 0.10,
    "total_hp": 0.15,
    "total_dps": 0.20,
    "spells": 0.15,
    "win_conditions": 0.20,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"

# Average elixir is minimized: full marks at CHEAP_ELIXIR or below, zero at
# EXPENSIVE_ELIXIR or above.
CHEAP_ELIXIR = 1.5
EXPENSIVE_ELIXIR = 5.0

# Count metrics score best near an ideal and fall off linearly to zero SPREAD away.
IDEAL_AIR_TROOPS = 2.0
AIR_SPREAD = 3.0
IDEAL_BUILDINGS = 1.0
BUILDINGS_SPREAD = 2.5
IDEAL_SPELLS = 2.0
SPELLS_SPREAD = 3.0

# Total HP / DPS are maximized: full marks once the deck reaches this much.
TARGET_TOTAL_HP = 20_000.0
TARGET_TOTAL_DPS = 3_000.0

# Win conditions: weight each card by its tier, then aim for an ideal total.
IDEAL_WIN_SCORE = 1.5
WIN_SPREAD = 1.0
WIN_TIER_WEIGHTS = {"primary": 1.0, "secondary": 0.67, "conditional": 0.33}


def score(deck: Deck) -> float:
    """Score a legal deck in [0, 1] (higher is better) as a weighted average.

    Each metric is normalized to [0, 1] and combined using WEIGHTS. Total HP and
    DPS are skipped -- and their weight shared across the other metrics -- until
    combat stats are loaded into cards.csv.
    """
    metrics: dict[str, float] = {}

    # Average elixir -- minimize it; cheaper decks cycle faster.
    cheapness = (EXPENSIVE_ELIXIR - deck.avg_elixir) / (EXPENSIVE_ELIXIR - CHEAP_ELIXIR)
    metrics["avg_elixir"] = max(0.0, min(1.0, cheapness))

    # Air troops / buildings / spells -- best near an ideal count.
    metrics["air_troops"] = max(0.0, 1.0 - abs(len(deck.air_troops) - IDEAL_AIR_TROOPS) / AIR_SPREAD)
    metrics["buildings"] = max(0.0, 1.0 - abs(len(deck.buildings) - IDEAL_BUILDINGS) / BUILDINGS_SPREAD)
    metrics["spells"] = max(0.0, 1.0 - abs(len(deck.spells) - IDEAL_SPELLS) / SPELLS_SPREAD)

    # Total HP / DPS -- maximize them. Only scored once the stats exist.
    hp = [c.hitpoints for c in deck.cards if c.hitpoints is not None]
    if hp:
        metrics["total_hp"] = min(1.0, sum(hp) / TARGET_TOTAL_HP)

    dps = [c.damage_per_second for c in deck.cards if c.damage_per_second is not None]
    if dps:
        metrics["total_dps"] = min(1.0, sum(dps) / TARGET_TOTAL_DPS)

    # Win conditions -- weight each card by tier, then aim for an ideal total.
    win_score = sum(WIN_TIER_WEIGHTS.get(c.win_condition, 0.0) for c in deck.cards)
    metrics["win_conditions"] = max(0.0, 1.0 - abs(win_score - IDEAL_WIN_SCORE) / WIN_SPREAD)

    # Weighted average over whatever metrics we computed.
    weighted_sum = sum(WEIGHTS[name] * value for name, value in metrics.items())
    total_weight = sum(WEIGHTS[name] for name in metrics)
    return weighted_sum / total_weight if total_weight else 0.0
