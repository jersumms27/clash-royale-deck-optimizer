"""Fetch the card list from the official CR API and read/write data/cards.csv (stdlib only).

This module owns the cards.csv *format*. Rebuilding the file from scratch (API
fetch + wiki scrape + merge) lives in build_dataset.py.
"""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request

from optimizer import config
from optimizer.models import Card, CardPool, build_card

# Base columns: straight from the API plus the classifications build_card adds.
CSV_FIELDS = ["id", "name", "elixir", "rarity", "type", "has_evolution",
              "is_champion", "is_champion_hero", "win_condition", "spell_size", "air"]

# Scraped attribute columns, in file order. build_dataset.py fills them in from
# the Fandom wiki; dev_sample.py leaves them blank.
COMBAT_ATTRS = ["hitpoints", "damage", "damage_per_second", "attack_period",
                "range", "radius", "lifetime", "crown_tower_damage", "special_damage"]
SPAWN_ATTRS = ["troop_spawned", "spawn_count_period", "max_troops_spawned"]
EVO_ATTRS = ["evo_cycles", "evo_overall_cost", "evo_stat_boosts"]
ATTRIBUTE_FIELDS = COMBAT_ATTRS + SPAWN_ATTRS + EVO_ATTRS
# Parsed as float; every other attribute is free text.
NUMERIC_ATTRS = set(COMBAT_ATTRS) | {"evo_cycles", "evo_overall_cost"}


def _to_bool(value: str) -> bool:
    """Parse a CSV cell as bool. Accepts True/False (and legacy 1/0)."""
    return str(value).strip().lower() in ("true", "1", "yes")


def _to_float(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_text(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() != "nan" else None


def fetch_raw_cards() -> list[dict]:
    """GET /cards from the official API and return the raw `items` list."""
    token = config.get_api_token()
    if not token:
        raise RuntimeError(
            "No API token. Get one at https://developer.clashroyale.com, then set "
            "CR_API_TOKEN or put it in token.txt in the project root."
        )

    url = f"{config.CR_API_BASE}/cards"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        # 403 usually means the token's whitelisted IP no longer matches yours.
        raise RuntimeError(f"CR API returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the CR API: {exc.reason}") from exc

    items = payload.get("items", [])
    if not items:
        raise RuntimeError("CR API returned no cards (unexpected response shape).")
    return items


def card_from_api_item(item: dict) -> Card:
    """One raw API card -> Card (classified via build_card; no stat columns)."""
    rarity = item.get("rarity", "common")
    icons = item.get("iconUrls", {})
    return build_card(
        id=item["id"],
        name=item["name"],
        elixir=item.get("elixirCost") or 0,
        rarity=rarity,
        # The icon set is the reliable signal for both forms. maxEvolutionLevel
        # also counts hero form (hero-only cards report 2, evo+hero 3), so it
        # can't be used on its own. build_dataset reports any card where
        # has_evolution disagrees with the wiki's evolution table.
        has_evolution="evolutionMedium" in icons,
        # Champions are heroes by definition and carry no heroMedium icon.
        is_champion_hero=rarity.lower() == config.CHAMPION_RARITY or "heroMedium" in icons,
    )


def fetch_cards_from_api() -> list[Card]:
    return [card_from_api_item(item) for item in fetch_raw_cards()]


def card_to_row(card: Card) -> dict:
    """A Card as one cards.csv row (None -> blank cell)."""
    row = {f: getattr(card, f) for f in CSV_FIELDS + ATTRIBUTE_FIELDS}
    return {k: ("" if v is None else v) for k, v in row.items()}


def save_cards_csv(cards: list[Card], path=config.CARDS_CSV) -> None:
    """Write the full cards.csv schema (base + attribute columns)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS + ATTRIBUTE_FIELDS)
        writer.writeheader()
        for c in cards:
            writer.writerow(card_to_row(c))


def load_cards_csv(path=config.CARDS_CSV) -> list[Card]:
    """Read every field straight from cards.csv -- it's the source of truth.
    (Classifications are baked in when the file is generated; see build_card.)"""
    cards: list[Card] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            attrs = {
                a: (_to_float if a in NUMERIC_ATTRS else _to_text)(row.get(a))
                for a in ATTRIBUTE_FIELDS
            }
            cards.append(
                Card(
                    id=int(row["id"]),
                    name=row["name"],
                    elixir=int(row["elixir"]),
                    rarity=row["rarity"],
                    has_evolution=_to_bool(row["has_evolution"]),
                    is_champion=_to_bool(row["is_champion"]),
                    is_champion_hero=_to_bool(row["is_champion_hero"]),
                    type=_to_text(row["type"]),
                    win_condition=_to_text(row["win_condition"]),
                    spell_size=_to_text(row["spell_size"]),
                    air=_to_bool(row["air"]),
                    **attrs,
                )
            )
    return cards


def load_card_pool(refresh: bool = False) -> CardPool:
    """Load cards.csv. With refresh=True (or no file yet) rebuild it from scratch
    first -- API fetch *and* wiki scrape -- so the stat columns are never lost."""
    if refresh or not config.CARDS_CSV.exists():
        # Needs pandas/requests/lxml; imported lazily so the optimizer itself stays stdlib-only.
        from optimizer.build_dataset import build_cards_csv

        build_cards_csv()
    return CardPool(load_cards_csv())
