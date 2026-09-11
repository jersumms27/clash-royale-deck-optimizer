# `data/` — what lives here and who writes it

| file | written by | read by |
|---|---|---|
| `cards.csv` | `optimizer/build_dataset.py` (`python -m optimizer.build_dataset`, `main.py --refresh`, or `notebooks/build_cards.ipynb`) | everything (`optimizer/cr_api.py`) |
| `card_attributes.csv` | `optimizer/build_dataset.py` | `UI/server.py` (card modal) |
| `scrape_cache.html` | `optimizer/build_dataset.py` (cached wiki page) | `optimizer/build_dataset.py` |
| `battles.csv` | **the data-collection agent** (contract below) | `notebooks/train_model.ipynb` |
| `meta_decks.csv` | `notebooks/train_model.ipynb` (and hand edits) | `optimizer/learned_fitness.py` |
| `matchup_model.pt` | `notebooks/train_model.ipynb` | `optimizer/learned_fitness.py` |

`cards.csv` is rebuilt from scratch in one step -- official API (id, name, elixir, rarity,
evolution) plus a Fandom wiki scrape (combat / spawn / evolution stats) merged by card name.
Rebuild it whenever new cards are released.

Card ids everywhere are the official CR API ids (`26xxxxxx` troops, `27xxxxxx` buildings,
`28xxxxxx` spells) exactly as they appear in `cards.csv`.

## `battles.csv` — data contract

One row per **1v1** battle. Cards are pipe-separated lists of API ids; order inside a list is
irrelevant. Required columns:

| column | type | notes |
|---|---|---|
| `battle_id` | str | unique per battle; used to dedupe (e.g. `battleTime + sorted player tags`) |
| `battle_time` | ISO-8601 (`2026-09-11T14:03:00Z`) | used for a time-based train/validation split |
| `game_mode` | str | informational, e.g. `pathOfLegend`, `PvP` |
| `a_cards` | `id\|id\|id\|id\|id\|id\|id\|id` | player A's 8 cards |
| `a_evo` | `id\|id` or empty | subset of `a_cards` that were **evolved** in this battle (battle-log `evolutionLevel == 1`) |
| `a_hero` | `id\|id` or empty | subset of `a_cards` played in **hero form** (battle-log `evolutionLevel == 2`; non-champions only — champions carry no `evolutionLevel` and are implicitly heroes) |
| `b_cards`, `b_evo`, `b_hero` | same | player B |
| `hero_known` | `True` / `False` | `True` once the flattener maps `evolutionLevel == 2` to the hero columns; `False` means hero form was not extracted (then `a_hero`/`b_hero` are empty and mean "unknown", not "none") |
| `result` | `1.0` / `0.0` / `0.5` | from A's perspective: A won / B won / draw (equal crowns) |

Optional columns (kept for filtering and future work; the model ignores them):

| column | type | notes |
|---|---|---|
| `a_trophies`, `b_trophies` | int | starting trophies / league |
| `a_tower`, `b_tower` | str | tower troop name (`Princess Tower`, `Cannoneer`, …) |

Rules:

- 1v1 modes only (no 2v2, draft, or special event rules). Card levels are ignored on purpose.
- Which side is "A" is arbitrary — the trainer randomly swaps sides, and the model is
  antisymmetric by construction, so no need to duplicate rows.
- Rows containing a card id that is not in `cards.csv` are dropped at load time (the count is
  reported), so a newly released card simply needs a refreshed `cards.csv`.
- Loader: `optimizer.battles.load_battles`. A synthetic generator with the same schema
  (`optimizer.battles.make_synthetic_battles`) exists for development before real data lands.

## `meta_decks.csv`

The opponents the GA is optimised against. Columns: `cards`, `evo`, `hero` (same pipe encoding
as above), `weight` (usage share, sums to 1 across the file), `label` (free text, optional).
`train_model.ipynb` regenerates it as the top-N most-played decks in `battles.csv`; edit it by
hand to pin specific archetypes or adjust weights. Fitness of a candidate deck is
`sum(weight_i * P(candidate beats meta_i))`.

## `matchup_model.pt`

PyTorch checkpoint written by `train_model.ipynb` and loaded by `optimizer.model.MatchupModel.load`.
Contains the weights, hyperparameters, the card-id vocabulary, the per-card feature matrix, and a
`has_hero_data` flag. When that flag is `False`, hero form is treated as base form at inference,
so the GA's hero slot has no effect on fitness until data with hero information is available.
