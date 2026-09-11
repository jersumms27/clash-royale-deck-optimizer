"""On-disk layout of the crawl: raw battle log, crawl state, and the CSV path.

    data/battles/raw.jsonl   one kept battle per line, raw API JSON minus icon
                             URLs, plus `battle_id` and `fetched_from`
    data/battles/state.json  which players were fetched when + the frontier
    data/battles.csv         the flattened dataset (see data/README.md)

raw.jsonl is append-only and the crawler rebuilds its "seen" set from it on
start-up, so a crash can never lose more than the last partial line.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

from optimizer import config

from collector.api import bare_tag

BATTLES_DIR = config.DATA_DIR / "battles"
RAW_PATH = BATTLES_DIR / "raw.jsonl"
STATE_PATH = BATTLES_DIR / "state.json"
CSV_PATH = getattr(config, "BATTLES_CSV", config.DATA_DIR / "battles.csv")

DECK_SIZE = 8
STATE_VERSION = 1


# --------------------------------------------------------------------------- #
# Battle helpers                                                               #
# --------------------------------------------------------------------------- #
def sides(b: dict) -> tuple[list[dict], list[dict]]:
    return list(b.get("team") or []), list(b.get("opponent") or [])


def is_1v1(b: dict) -> bool:
    team, opp = sides(b)
    return len(team) == 1 and len(opp) == 1


def keep_battle(b: dict) -> bool:
    """1v1, normal deck selection, and a full 8-card deck visible on each side.

    Mode/type filtering (ladder vs. ranked vs. event) is deliberately left to
    flatten.py so those choices can change without recrawling.
    """
    if not is_1v1(b):
        return False
    if b.get("deckSelection", "collection") != "collection":
        return False
    team, opp = sides(b)
    return all(len(side.get("cards") or []) == DECK_SIZE for side in (team[0], opp[0]))


def battle_id(b: dict) -> str:
    """battleTime + the two bare tags, sorted -- identical from either player's log."""
    team, opp = sides(b)
    tags = sorted(bare_tag(s["tag"]) for s in team + opp)
    return b["battleTime"] + "_" + "_".join(tags)


def participant_tags(b: dict, exclude: str | None = None) -> list[str]:
    """Normalised tags of everyone in the battle except `exclude`."""
    team, opp = sides(b)
    out = []
    for side in team + opp:
        tag = side.get("tag")
        if not tag:
            continue
        try:
            tag = "#" + bare_tag(tag)
        except ValueError:
            continue
        if tag != exclude:
            out.append(tag)
    return out


def slim_battle(b: dict, fetched_from: str) -> dict:
    """The battle as stored: id + provenance first, icon URLs stripped."""
    out = {"battle_id": battle_id(b), "fetched_from": fetched_from}
    for key, value in b.items():
        if key in ("team", "opponent"):
            value = [_slim_side(side) for side in value]
        out[key] = value
    return out


def _slim_side(side: dict) -> dict:
    out = dict(side)
    for key in ("cards", "supportCards"):
        if key in out and out[key]:
            out[key] = [{k: v for k, v in card.items() if k != "iconUrls"} for card in out[key]]
    return out


# --------------------------------------------------------------------------- #
# raw.jsonl                                                                    #
# --------------------------------------------------------------------------- #
_ID_RE = re.compile(rb'^\{"battle_id":\s*"([^"]+)"')


def _open_text(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _open_binary(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def scan_ids(path: Path = RAW_PATH) -> tuple[set[str], int]:
    """Every battle_id in the file (regex on the line prefix; no JSON parsing).

    Returns (ids, bad_lines). Lines that don't start with a battle_id -- e.g.
    a torn last line after a crash -- are counted, not raised.
    """
    ids: set[str] = set()
    bad = 0
    if not Path(path).exists():
        return ids, bad
    with _open_binary(Path(path)) as fh:
        for line in fh:
            if not line.strip():
                continue
            m = _ID_RE.match(line)
            if m and line.rstrip(b"\r\n").endswith(b"}"):
                ids.add(m.group(1).decode("utf-8"))
            else:
                bad += 1
    return ids, bad


def iter_raw(path: Path = RAW_PATH, *, on_bad=None) -> Iterator[dict]:
    """Yield one battle dict per line; unparsable lines are skipped (and passed
    to `on_bad(line_no)` if given)."""
    if not Path(path).exists():
        return
    with _open_text(Path(path)) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if on_bad is not None:
                    on_bad(n)


class RawWriter:
    """Append-only JSON-lines writer. Safe to reopen after a crash."""

    def __init__(self, path: Path = RAW_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                needs_newline = fh.read(1) != b"\n"
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
        if needs_newline:  # isolate a torn last line so the next record parses
            self._fh.write("\n")
        self.written = 0

    def append(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.written += 1

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()


# --------------------------------------------------------------------------- #
# state.json                                                                   #
# --------------------------------------------------------------------------- #
def empty_state() -> dict:
    return {
        "version": STATE_VERSION,
        "players": {},  # "#TAG" -> unix time of the last battlelog fetch
        "queue": [],  # frontier, in order
        "totals": {"fetches": 0, "kept": 0, "runs": 0},
        "updated": None,
    }


def load_state(path: Path = STATE_PATH) -> dict:
    state = empty_state()
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
    except FileNotFoundError:
        return state
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is corrupt ({exc}); delete it to start over "
                           "(raw.jsonl is untouched)") from exc
    state.update(loaded)
    state["totals"] = {**empty_state()["totals"], **state.get("totals", {})}
    return state


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    """Atomic write (tmp + rename). Retries the rename, which Windows refuses
    while another process has the file open."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = time.time()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    for attempt in range(3):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))
