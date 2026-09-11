"""Battle-log collector: crawls the official CR API into data/battles.csv.

    python -m collector.probe      # inspect one player's battle log (verify API fields)
    python -m collector.crawl      # snowball-crawl battle logs into data/battles/raw.jsonl
    python -m collector.flatten    # raw.jsonl -> data/battles.csv (schema: data/README.md)
    python -m collector.stats      # sanity report on the raw log and the CSV

Standard library only. The optimizer package is only *read* (config paths + token).
"""
