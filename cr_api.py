"""Fetch the card list from the official CR API and cache it to CSV (stdlib only)."""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request

import config
from models import Card, CardPool

# Base columns written by the API fetch. The scraped attribute columns
# (hitpoints, damage, ... evo_stat_boosts) are layered onto cards.csv by
# scrape.ipynb and read back by load_cards_csv, but are not written here.
CSV_FIELDS = ["id", "name", "elixir", "rarity", "type", "has_evolution",
              "is_champion_hero", "win_condition", "spell_size", "air"]

# Scraped attribute columns: numeric ones parsed as float, text ones as str.
_NUMERIC_ATTRS = ["hitpoints", "damage", "damage_per_second", "attack_period",
                  "range", "radius", "lifetime", "crown_tower_damage",
                  "special_damage", "evo_cycles", "evo_overall_cost"]
_TEXT_ATTRS = ["troop_spawned", "spawn_count_period", "max_troops_spawned",
               "evo_stat_boosts"]


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


def fetch_cards_from_api() -> list[Card]:
    token = config.get_api_token()
    if not token:
        raise RuntimeError(
            "No API token. Get one at https://developer.clashroyale.com, then set "
            "CR_API_TOKEN or put it in token.txt next to this script."
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

    cards: list[Card] = []
    for item in payload.get("items", []):
        cards.append(
            Card(
                id=item["id"],
                name=item["name"],
                elixir=item.get("elixirCost") or 0,
                rarity=item.get("rarity", "common"),
                # Only trust an explicit evolution level. The "evolutionMedium"
                # icon URL is present for many cards without a real evolution, so
                # using it here over-counted evolutions badly (see load_cards_csv,
                # which reconciles this against the scraped evo_* data).
                has_evolution="maxEvolutionLevel" in item,
            )
        )
    if not cards:
        raise RuntimeError("CR API returned no cards (unexpected response shape).")
    return cards


def save_cards_csv(cards: list[Card], path=config.CARDS_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for c in cards:
            writer.writerow(
                {
                    "id": c.id,
                    "name": c.name,
                    "elixir": c.elixir,
                    "rarity": c.rarity,
                    "type": c.type,
                    "has_evolution": c.has_evolution,
                    "is_champion_hero": c.is_champion_hero,
                    "win_condition": c.win_condition,
                    "spell_size": c.spell_size,
                    "air": c.air,
                }
            )


def load_cards_csv(path=config.CARDS_CSV) -> list[Card]:
    cards: list[Card] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            attrs = {a: _to_float(row.get(a)) for a in _NUMERIC_ATTRS}
            attrs.update({a: _to_text(row.get(a)) for a in _TEXT_ATTRS})
            # The scraped evo data is the ground truth for "has an evolution":
            # a card with real evo cycles/cost always has one, regardless of what
            # the (historically over-broad) has_evolution column says.
            has_real_evo_data = (
                attrs["evo_cycles"] is not None
                or attrs["evo_overall_cost"] is not None
            )
            cards.append(
                Card(
                    id=int(row["id"]),
                    name=row["name"],
                    elixir=int(row["elixir"]),
                    rarity=row["rarity"],
                    has_evolution=_to_bool(row["has_evolution"]) or has_real_evo_data,
                    **attrs,
                )
            )
    return cards


def load_card_pool(refresh: bool = False) -> CardPool:
    if refresh or not config.CARDS_CSV.exists():
        cards = fetch_cards_from_api()
        save_cards_csv(cards)
    else:
        cards = load_cards_csv()
    return CardPool(cards)
