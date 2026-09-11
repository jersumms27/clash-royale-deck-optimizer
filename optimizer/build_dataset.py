"""Rebuild data/cards.csv from scratch: official API -> Fandom wiki scrape -> merge.

    python -m optimizer.build_dataset            # full rebuild (re-downloads the wiki page)
    python -m optimizer.build_dataset --cached   # reuse data/scrape_cache.html

or from Python / notebooks/build_cards.ipynb:  build_cards_csv()

Steps (each its own function so the notebook can show what happens in between):
  1. fetch_cards_from_api  -> base columns (id, name, elixir, rarity, ...)   [cr_api]
  2. fetch_wiki_html       -> the Fandom "Cards" page, cached at data/scrape_cache.html
  3. parse_wiki_tables     -> one row per wiki card: combat / spawn / evolution stats
  4. match_wiki_to_api     -> wiki names normalised to API spelling
  5. apply_attributes + save -> data/card_attributes.csv and data/cards.csv

Needs an API token (config.get_api_token) plus pandas, requests and lxml -- the
`hackathon` conda env. The rest of the optimizer package stays stdlib-only; only
cr_api.load_card_pool(refresh=True) imports this, lazily.
"""

from __future__ import annotations

import argparse
import io
import re
import time
from dataclasses import dataclass, replace

try:
    import pandas as pd
    import requests
except ImportError as exc:
    raise ImportError(
        "build_dataset needs pandas, requests and lxml -- use the 'hackathon' conda "
        "env (environment.yml)."
    ) from exc

from optimizer import config
from optimizer.cr_api import (
    ATTRIBUTE_FIELDS,
    NUMERIC_ATTRS,
    fetch_cards_from_api,
    load_cards_csv,
    save_cards_csv,
)
from optimizer.models import Card

WIKI_URL = "https://clashroyale.fandom.com/wiki/Cards"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Wiki card name -> API card name, for names the automatic normalisation (exact
# match, then trailing-dot strip) can't resolve. The build report lists every
# unmatched wiki row and every API card without stats; add fixes here.
ALIASES: dict[str, str] = {}

# (substring of the lowercased wiki header) -> attribute column. Order matters:
# more specific keys first ("crown tower damage" before "damage").
_HEADER_KEYS = [
    ("damage per second", "damage_per_second"),
    ("crown tower damage", "crown_tower_damage"),
    ("special damage", "special_damage"),
    ("hitpoint", "hitpoints"),
    ("health", "hitpoints"),
    ("attack period", "attack_period"),
    ("damage", "damage"),
    ("range", "range"),
    ("radius", "radius"),
    ("lifetime", "lifetime"),
    ("maximum troops", "max_troops_spawned"),
    ("spawn count", "spawn_count_period"),
    ("troop spawned", "troop_spawned"),
    ("cycles", "evo_cycles"),
    ("overall cost", "evo_overall_cost"),
    ("stat boost", "evo_stat_boosts"),
]


# --------------------------------------------------------------------------- #
# 2. wiki page                                                                #
# --------------------------------------------------------------------------- #
def fetch_wiki_html(refresh: bool = True, attempts: int = 5) -> str:
    """The Fandom Cards page. Fandom 403s intermittently, so retry; the page is
    cached at config.SCRAPE_CACHE and refresh=False reuses that cache. If every
    attempt fails but a cache exists, fall back to it (loudly)."""
    cache = config.SCRAPE_CACHE
    if not refresh and cache.exists():
        return cache.read_text(encoding="utf-8")

    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(WIKI_URL, headers=_HEADERS, timeout=30)
        except requests.RequestException as exc:
            status = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                cache.parent.mkdir(exist_ok=True)
                cache.write_text(resp.text, encoding="utf-8")
                return resp.text
            status = f"HTTP {resp.status_code}"
        print(f"    wiki fetch attempt {attempt}/{attempts} failed ({status})")
        time.sleep(1.5)

    if cache.exists():
        print(f"WARNING: could not download {WIKI_URL}; using the cached copy at {cache}. "
              "Stats for cards added since then will be missing.")
        return cache.read_text(encoding="utf-8")
    raise RuntimeError(f"Could not fetch {WIKI_URL} (Fandom kept blocking). Re-run shortly.")


# --------------------------------------------------------------------------- #
# 3. parse the card tables                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class WikiStats:
    frame: pd.DataFrame  # one row per wiki card: `card` + whichever ATTRIBUTE_FIELDS were found
    tables_found: int
    tables_used: int


def _clean_number(val) -> float:
    """Leading number of a wiki cell: "366 (122x3)" -> 366.0, "1,356" -> 1356.0, prose -> NaN."""
    m = re.search(r"\d[\d,]*\.?\d*", str(val))
    return float(m.group(0).replace(",", "")) if m else float("nan")


def _clean_text(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})


def _attribute_column(header) -> str | None:
    h = str(header).lower()
    if h.strip() == "dps":
        return "damage_per_second"
    for key, col in _HEADER_KEYS:
        if key in h:
            return col
    return None


def parse_wiki_tables(html: str) -> WikiStats:
    """Every wiki table with a `Card` column, cleaned and coalesced by card name."""
    tables = pd.read_html(io.StringIO(html))  # pandas 3 wants a file-like object
    frames = []
    for t in tables:
        t.columns = [str(c) for c in t.columns]
        if "Card" not in t.columns:
            continue
        sub = pd.DataFrame({"card": t["Card"].astype(str).str.strip()})
        for col in t.columns:
            out = _attribute_column(col)
            if out is None or out in sub.columns:
                continue
            sub[out] = t[col].map(_clean_number) if out in NUMERIC_ATTRS else _clean_text(t[col])
        sub = sub[sub["card"].notna() & ~sub["card"].isin(["nan", "Card"])]
        if sub.shape[1] > 1:  # at least one data column besides `card`
            frames.append(sub)
    if not frames:
        raise RuntimeError("No card tables found on the wiki page -- has its layout changed?")

    big = pd.concat(frames, ignore_index=True)
    # A card can appear in several tables; first() coalesces the first non-null per column.
    merged = big.groupby("card", as_index=False).first()
    return WikiStats(merged, tables_found=len(tables), tables_used=len(frames))


# --------------------------------------------------------------------------- #
# 4. wiki names -> API names                                                  #
# --------------------------------------------------------------------------- #
def to_api_name(wiki_name: str, api_names: set[str]) -> str | None:
    n = str(wiki_name).strip()
    if n in api_names:
        return n
    if n.rstrip(".") in api_names:  # "P.E.K.K.A." / "Mini P.E.K.K.A."
        return n.rstrip(".")
    return ALIASES.get(n)


def match_wiki_to_api(stats: pd.DataFrame, api_names) -> tuple[pd.DataFrame, list[str]]:
    """Returns (attribute table keyed by API `name`, unmatched wiki names).
    Unmatched rows are normal: tower troops, spawned sub-units, removed cards."""
    api_names = set(api_names)
    stats = stats.assign(name=stats["card"].map(lambda n: to_api_name(n, api_names)))
    unmatched = sorted(stats.loc[stats["name"].isna(), "card"].unique())

    cols = ["name"] + [c for c in ATTRIBUTE_FIELDS if c in stats.columns]
    attrs = (
        stats[stats["name"].notna()][cols]
        .groupby("name", as_index=False).first()  # two wiki spellings -> one API card
        .sort_values("name").reset_index(drop=True)
    )
    return attrs, unmatched


# --------------------------------------------------------------------------- #
# 5. merge + save                                                             #
# --------------------------------------------------------------------------- #
def _cell(value, numeric: bool):
    if value is None or pd.isna(value):
        return None
    return float(value) if numeric else str(value).strip()


def apply_attributes(cards: list[Card], attrs: pd.DataFrame) -> list[Card]:
    """Copy the scraped columns onto each Card by name; cards with no row keep None."""
    by_name = {
        row["name"]: {c: _cell(row[c], c in NUMERIC_ATTRS) for c in ATTRIBUTE_FIELDS if c in row}
        for row in attrs.to_dict("records")
    }
    return [replace(c, **by_name.get(c.name, {})) for c in cards]


@dataclass
class BuildReport:
    cards: list[Card]
    tables_found: int
    tables_used: int
    wiki_cards: int
    attribute_columns: list[str]
    unmatched_wiki: list[str]     # wiki rows with no API card
    api_without_stats: list[str]  # API cards no wiki row matched
    evo_mismatch: list[str]       # has_evolution (API) disagrees with the wiki evolution table
    added: list[str]              # not in the previous cards.csv
    removed: list[str]            # in the previous cards.csv, gone from the API

    def summary(self) -> str:
        def lst(items):
            return f"({len(items)})" + (": " + ", ".join(items) if items else "")

        n_cols = len(self.cards[0].__dataclass_fields__) if self.cards else 0
        return "\n".join([
            f"cards.csv rebuilt: {len(self.cards)} cards x {n_cols} columns -> {config.CARDS_CSV}",
            f"  wiki: {self.tables_used}/{self.tables_found} tables used, {self.wiki_cards} wiki "
            f"rows, {len(self.cards) - len(self.api_without_stats)} API cards matched "
            f"-> {config.CARD_ATTRIBUTES_CSV.name}",
            f"  added since last build {lst(self.added)}",
            f"  removed since last build {lst(self.removed)}",
            f"  API cards with no wiki stats {lst(self.api_without_stats)}",
            f"  unmatched wiki rows (tower troops / sub-units / removed cards are expected) "
            f"{lst(self.unmatched_wiki)}",
            f"  has_evolution disagrees with wiki evolution table {lst(self.evo_mismatch)}",
        ])


def build_cards_csv(refresh_scrape: bool = True) -> BuildReport:
    """The whole pipeline: API -> wiki -> data/card_attributes.csv + data/cards.csv.
    refresh_scrape=False reuses data/scrape_cache.html instead of re-downloading."""
    previous = {c.name for c in load_cards_csv(config.CARDS_CSV)} if config.CARDS_CSV.exists() else set()

    print("1/3 fetching the card list from the official API ...")
    base = fetch_cards_from_api()
    print(f"    {len(base)} cards")

    print("2/3 scraping card stats from the wiki ...")
    wiki = parse_wiki_tables(fetch_wiki_html(refresh=refresh_scrape))
    attrs, unmatched = match_wiki_to_api(wiki.frame, {c.name for c in base})
    print(f"    {wiki.tables_used}/{wiki.tables_found} tables, {len(wiki.frame)} wiki rows, "
          f"{len(attrs)} matched to API cards")

    print("3/3 writing data files ...")
    cards = apply_attributes(base, attrs)
    config.DATA_DIR.mkdir(exist_ok=True)
    attrs.to_csv(config.CARD_ATTRIBUTES_CSV, index=False)
    save_cards_csv(cards, config.CARDS_CSV)

    names = {c.name for c in cards}
    report = BuildReport(
        cards=cards,
        tables_found=wiki.tables_found,
        tables_used=wiki.tables_used,
        wiki_cards=len(wiki.frame),
        attribute_columns=[c for c in attrs.columns if c != "name"],
        unmatched_wiki=unmatched,
        api_without_stats=sorted(names - set(attrs["name"])),
        evo_mismatch=sorted(c.name for c in cards if c.has_evolution != (c.evo_cycles is not None)),
        added=sorted(names - previous) if previous else [],
        removed=sorted(previous - names),
    )
    print(report.summary())
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild data/cards.csv from the API + wiki.")
    parser.add_argument("--cached", action="store_true",
                        help="reuse data/scrape_cache.html instead of re-downloading the wiki page")
    args = parser.parse_args(argv)
    build_cards_csv(refresh_scrape=not args.cached)


if __name__ == "__main__":
    main()
