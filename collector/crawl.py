"""Snowball-crawl battle logs from the official CR API into data/battles/raw.jsonl.

Seeds come from the Path of Legend / trophy rankings; every opponent seen in a
battle log joins the frontier (BFS), so the crawl stays in the skill band it
started from and never runs dry. The log only holds a player's last ~25
battles, so players are re-polled after --recrawl-after hours; leave it
running (or re-run it daily) and the dataset keeps growing.

    python -m collector.crawl                              # run until Ctrl+C
    python -m collector.crawl --max-players 20             # quick smoke test
    python -m collector.crawl --max-minutes 120 --rps 4    # bounded run
    python -m collector.crawl --seed-tags "#ABC,#DEF"      # add your own seeds

Resumable: state.json remembers who was fetched when and the queue; the set of
battles already stored is rebuilt from raw.jsonl on start-up. Ctrl+C once =
stop cleanly after in-flight requests; twice = force.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from collector.api import ApiError, Client, FatalApiError, NotFound, make_client, norm_tag
from collector.store import (
    BATTLES_DIR, RawWriter, battle_id, is_1v1, keep_battle, load_state,
    participant_tags, save_state, scan_ids, slim_battle,
)


@dataclass
class FetchResult:
    tag: str
    battles: list[dict]
    status: str  # "ok" | "notfound" | "error"
    error: str = ""


def fetch_one(client: Client, tag: str) -> FetchResult:
    """Worker: one battlelog GET. Only a bad token escapes as an exception."""
    try:
        return FetchResult(tag, client.battlelog(tag), "ok")
    except NotFound:
        return FetchResult(tag, [], "notfound")
    except FatalApiError:
        raise
    except ApiError as exc:
        return FetchResult(tag, [], "error", str(exc))
    except Exception as exc:  # never let a surprise kill the pool
        return FetchResult(tag, [], "error", repr(exc))


class Frontier:
    """FIFO of player tags to fetch, deduplicated and freshness-aware.

    `players` is the shared tag -> last-fetched map from state.json; a tag is
    "fresh" (not worth refetching yet) for `recrawl_after` seconds after that.
    """

    def __init__(self, queue: list[str], players: dict[str, float],
                 recrawl_after: float, max_queue: int):
        self.queue: deque[str] = deque()
        self.queued: set[str] = set()
        self.in_flight: set[str] = set()
        self.players = players
        self.recrawl_after = recrawl_after
        self.max_queue = max_queue
        self.dropped = 0  # pushes refused because the queue was full
        for tag in queue:
            self.push(tag)

    def __len__(self) -> int:
        return len(self.queue)

    def is_fresh(self, tag: str, now: float | None = None) -> bool:
        last = self.players.get(tag)
        return last is not None and (now if now is not None else time.time()) - last < self.recrawl_after

    def push(self, tag: str, front: bool = False) -> bool:
        if tag in self.queued or tag in self.in_flight or self.is_fresh(tag):
            return False
        if len(self.queue) >= self.max_queue:
            self.dropped += 1
            return False
        (self.queue.appendleft if front else self.queue.append)(tag)
        self.queued.add(tag)
        return True

    def pop(self) -> str | None:
        while self.queue:
            tag = self.queue.popleft()
            self.queued.discard(tag)
            if self.is_fresh(tag):  # fetched via another path since it was queued
                continue
            self.in_flight.add(tag)
            return tag
        return None

    def done(self, tag: str) -> None:
        self.in_flight.discard(tag)

    def refill_from_stale(self, now: float) -> int:
        """Queue every known player whose last fetch is older than recrawl_after, oldest first."""
        stale = [t for t, ts in self.players.items()
                 if now - ts >= self.recrawl_after and t not in self.queued and t not in self.in_flight]
        stale.sort(key=self.players.__getitem__)
        n = 0
        for tag in stale:
            if len(self.queue) >= self.max_queue:
                break
            if self.push(tag):
                n += 1
        return n

    def next_stale_at(self) -> float | None:
        if not self.players:
            return None
        return min(self.players.values()) + self.recrawl_after


# --------------------------------------------------------------------------- #
# Seeds                                                                        #
# --------------------------------------------------------------------------- #
def explicit_seed_tags(args) -> list[str]:
    tags: list[str] = []
    if args.seed_tags:
        tags += [t for t in args.seed_tags.replace(";", ",").split(",") if t.strip()]
    if args.seed_file:  # one tag per line ("//" lines are comments; "#" can't be, tags start with it)
        with open(args.seed_file, encoding="utf-8") as fh:
            tags += [line.strip() for line in fh if line.strip() and not line.lstrip().startswith("//")]
    out = []
    for t in tags:
        try:
            out.append(norm_tag(t))
        except ValueError:
            print(f"  ignoring invalid seed tag {t!r}")
    return out


def ranking_seed_tags(client: Client, args, stop: threading.Event, log) -> list[str]:
    """Top Path of Legend players, global + per country.

    (The old trophy-road ranking, /locations/global/rankings/players, returns
    zero items on the live API now that top ladder is Path of Legend.)
    """
    seen: set[str] = set()
    tags: list[str] = []

    def add(label: str, fetch) -> None:
        try:
            got = fetch()
        except NotFound:
            got = []
        except ApiError as exc:
            log(f"  seed {label}: {exc}")
            got = []
        fresh = [t for t in got if t not in seen]
        seen.update(fresh)
        tags.extend(fresh)
        if fresh:
            log(f"  seed {label}: +{len(fresh)}")

    add("global Path of Legend", lambda: client.pol_players("global", args.seed_limit))
    if args.seed_locations != "global" and not stop.is_set():
        try:
            locations = [loc for loc in client.locations() if loc.get("isCountry")]
        except ApiError as exc:
            log(f"  could not list locations: {exc}")
            locations = []
        if args.seed_locations != "all":
            locations = locations[: int(args.seed_locations)]
        log(f"  seeding from {len(locations)} country rankings ...")
        for loc in locations:
            if stop.is_set():
                break
            name = loc.get("name", loc.get("id"))
            add(str(name), lambda loc_id=loc["id"]: client.pol_players(loc_id, args.seed_limit))
    return tags


# --------------------------------------------------------------------------- #
# Main loop                                                                    #
# --------------------------------------------------------------------------- #
def _install_sigint(stop: threading.Event) -> None:
    def handler(signum, frame):
        print("\nStopping after in-flight requests ... (Ctrl+C again to force)", flush=True)
        stop.set()
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, handler)


def _pause(stop: threading.Event, seconds: float) -> None:
    """Sleep in 1 s slices so Ctrl+C is noticed promptly on Windows."""
    end = time.monotonic() + seconds
    while not stop.is_set() and time.monotonic() < end:
        stop.wait(min(1.0, end - time.monotonic()))


def run(args) -> int:
    stop = threading.Event()
    _install_sigint(stop)

    def log(msg: str) -> None:
        print(msg, flush=True)

    data_dir = Path(args.data_dir)
    raw_path, state_path = data_dir / "raw.jsonl", data_dir / "state.json"
    client = make_client(rps=args.rps, burst=args.burst, timeout=args.timeout, stop=stop)
    log(f"API base: {client.base}   data: {data_dir}")

    seen, bad_lines = scan_ids(raw_path)
    log(f"raw.jsonl: {len(seen)} battles already stored" + (f" ({bad_lines} unreadable lines)" if bad_lines else ""))
    state = load_state(state_path)
    state["totals"]["runs"] += 1
    frontier = Frontier(state["queue"], state["players"], args.recrawl_after * 3600.0, args.max_queue)
    log(f"state.json: {len(state['players'])} players known, {len(frontier)} queued")

    for tag in explicit_seed_tags(args):
        frontier.push(tag)
    first_run = not state["players"] and len(frontier) == 0
    if not args.no_seed and (args.reseed or first_run):
        log("Seeding from rankings ...")
        n = sum(frontier.push(t) for t in ranking_seed_tags(client, args, stop, log))
        log(f"  {n} seed players queued")
    if len(frontier) == 0:
        frontier.refill_from_stale(time.time())

    writer = RawWriter(raw_path)
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="fetch")
    pending: dict = {}
    fetched = kept = errors = notfound = 0
    kept_at_last_report = 0
    base_fetches, base_kept = state["totals"]["fetches"], state["totals"]["kept"]
    started = time.time()
    fatal: Exception | None = None
    limit_reason = ""

    def checkpoint() -> None:
        nonlocal kept_at_last_report
        writer.flush()
        state["queue"] = list(frontier.queue)
        state["totals"]["fetches"] = base_fetches + fetched
        state["totals"]["kept"] = base_kept + kept
        save_state(state, state_path)
        elapsed = max(1e-9, time.time() - started)
        log(f"[{time.strftime('%H:%M:%S')}] fetched {fetched:>6} | kept {kept:>7} (+{kept - kept_at_last_report}) "
            f"| seen {len(seen):>8} | queue {len(frontier):>6} | players {len(state['players']):>6} "
            f"| {client.requests / elapsed:4.1f} req/s | err {errors} nf {notfound}")
        kept_at_last_report = kept

    def limits_hit() -> str:
        if args.max_players and fetched >= args.max_players:
            return f"--max-players {args.max_players} reached"
        if args.max_battles and kept >= args.max_battles:
            return f"--max-battles {args.max_battles} reached"
        if args.max_minutes and time.time() - started >= args.max_minutes * 60:
            return f"--max-minutes {args.max_minutes} reached"
        return ""

    try:
        while not stop.is_set():
            limit_reason = limits_hit()
            if limit_reason:
                break
            while len(pending) < 2 * args.workers and not stop.is_set():
                tag = frontier.pop()
                if tag is None:
                    break
                pending[executor.submit(fetch_one, client, tag)] = tag
            if not pending:
                now = time.time()
                if frontier.refill_from_stale(now) > 0:
                    continue
                if args.exit_when_idle:
                    limit_reason = "frontier empty (all known players fetched recently)"
                    break
                next_at = frontier.next_stale_at()
                delay = 60.0 if next_at is None else max(1.0, min(60.0, next_at - now))
                log(f"idle: all {len(state['players'])} known players fetched within "
                    f"{args.recrawl_after}h; next re-poll in {delay:.0f}s")
                _pause(stop, delay)
                continue

            done, _ = wait(list(pending), timeout=0.5, return_when=FIRST_COMPLETED)
            for fut in done:
                tag = pending.pop(fut)
                frontier.done(tag)
                try:
                    res = fut.result()
                except FatalApiError as exc:
                    fatal = exc
                    stop.set()
                    break
                except Exception as exc:
                    res = FetchResult(tag, [], "error", repr(exc))

                state["players"][tag] = time.time()  # ok, 404 and error alike: no retry loops
                fetched += 1
                if res.status == "notfound":
                    notfound += 1
                elif res.status == "error":
                    errors += 1
                    if not args.quiet and errors <= 20:
                        log(f"  {tag}: {res.error}")

                for b in res.battles:
                    if not is_1v1(b):
                        continue
                    for other in participant_tags(b, exclude=tag):
                        frontier.push(other)
                    if not keep_battle(b):
                        continue
                    try:
                        bid = battle_id(b)
                    except (KeyError, ValueError):
                        continue
                    if bid in seen:
                        continue
                    seen.add(bid)
                    writer.append(slim_battle(b, tag))
                    kept += 1

                if fetched % args.save_every == 0:
                    checkpoint()
    except KeyboardInterrupt:  # second Ctrl+C
        log("Forced stop.")
        stop.set()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        for fut, tag in pending.items():  # nothing is lost: unfinished tags go back first
            frontier.done(tag)
            frontier.push(tag, front=True)
        checkpoint()
        writer.close()

    elapsed = time.time() - started
    log(f"\nDone{': ' + limit_reason if limit_reason else ''}. {fetched} players fetched, "
        f"{kept} new battles kept ({kept / max(elapsed, 1) * 3600:.0f}/h), {len(seen)} total in raw.jsonl, "
        f"{elapsed / 60:.1f} min.")
    if frontier.dropped:
        log(f"({frontier.dropped} frontier pushes dropped because the queue was full -- harmless)")
    if fatal is not None:
        log(f"\nABORTED: {fatal}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl CR battle logs into raw.jsonl")
    g = parser.add_argument_group("throughput")
    g.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    g.add_argument("--rps", type=float, default=5.0, help="requests per second across all workers (default 5)")
    g.add_argument("--burst", type=int, default=5)
    g.add_argument("--timeout", type=float, default=20.0, help="per-request timeout in seconds")
    g = parser.add_argument_group("limits (0 = unlimited)")
    g.add_argument("--max-players", type=int, default=0, help="stop after fetching this many battle logs")
    g.add_argument("--max-battles", type=int, default=0, help="stop after storing this many new battles")
    g.add_argument("--max-minutes", type=float, default=0)
    g.add_argument("--exit-when-idle", action="store_true",
                   help="exit instead of waiting when every known player was fetched recently")
    g = parser.add_argument_group("frontier")
    g.add_argument("--recrawl-after", type=float, default=6.0, help="hours before a player is re-polled (default 6)")
    g.add_argument("--max-queue", type=int, default=200_000)
    g.add_argument("--save-every", type=int, default=100, help="checkpoint state every N fetches")
    g = parser.add_argument_group("seeds (rankings are used on the first run, or with --reseed)")
    g.add_argument("--seed-locations", default="all", help="'all', 'global', or a number of countries (default all)")
    g.add_argument("--seed-limit", type=int, default=100, help="players per ranking (default 100)")
    g.add_argument("--seed-tags", default="", help="comma-separated player tags to add")
    g.add_argument("--seed-file", default="", help="file with one player tag per line")
    g.add_argument("--reseed", action="store_true", help="re-run the ranking seeds even if state exists")
    g.add_argument("--no-seed", action="store_true", help="never hit the ranking endpoints")
    parser.add_argument("--data-dir", default=str(BATTLES_DIR), help=f"where raw.jsonl/state.json live (default {BATTLES_DIR})")
    parser.add_argument("--quiet", action="store_true", help="don't print per-player errors")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1 or args.save_every < 1:
        parser.error("--workers and --save-every must be >= 1")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
