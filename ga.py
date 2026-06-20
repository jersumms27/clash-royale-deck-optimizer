"""A small, dependency-free genetic algorithm for evolving valid CR decks.

The engine only enforces validity; how *good* a deck is comes entirely from the
fitness function you pass in (see heuristic.py).
"""

from __future__ import annotations

import random
from typing import Callable

import config
from models import CardPool, Deck

FitnessFn = Callable[[Deck], float]


def _repair_card_ids(ids: list[int], pool: CardPool, rng: random.Random) -> list[int]:
    """DECK_SIZE distinct cards, at most MAX_CHAMPIONS of them champions."""
    ids = list(dict.fromkeys(ids))  # drop duplicates, keep order

    champs = [c for c in ids if c in pool.champion_set]
    drop = set(champs[config.MAX_CHAMPIONS:])
    ids = [c for c in ids if c not in drop][: config.DECK_SIZE]

    in_deck = set(ids)
    num_champs = len(in_deck & pool.champion_set)
    while len(ids) < config.DECK_SIZE:
        cand = rng.choice(pool.all_ids)
        is_champ = cand in pool.champion_set
        if cand in in_deck or (is_champ and num_champs >= config.MAX_CHAMPIONS):
            continue
        ids.append(cand)
        in_deck.add(cand)
        num_champs += is_champ
    return ids


def _repair_forms(
    deck_ids: list[int],
    evolved: set[int],
    hero: set[int],
    pool: CardPool,
    rng: random.Random,
) -> tuple[frozenset[int], frozenset[int]]:
    """Coerce the requested evo/hero forms into a legal assignment.

    Champions are always hero form; every other special card holds at most one
    form; evolutions and heroes share the wild slot (config.slots_ok). Optional
    forms beyond the budget are dropped at random so the GA keeps exploring.
    """
    in_deck = set(deck_ids)
    final_evo: set[int] = set()
    final_hero = {cid for cid in deck_ids if cid in pool.champion_set}  # mandatory

    requests = [("evo", c) for c in evolved if c in pool.evolvable_ids]
    requests += [("hero", c) for c in hero if c in pool.hero_set]
    rng.shuffle(requests)  # unbiased trimming when the budget is tight
    for form, cid in requests:
        if cid not in in_deck or cid in final_evo or cid in final_hero:
            continue  # not in this deck, or already holds a form
        if form == "evo" and config.slots_ok(len(final_evo) + 1, len(final_hero)):
            final_evo.add(cid)
        elif form == "hero" and config.slots_ok(len(final_evo), len(final_hero) + 1):
            final_hero.add(cid)
    return frozenset(final_evo), frozenset(final_hero)


def _build_deck(
    ids: list[int],
    evolved: set[int],
    hero: set[int],
    pool: CardPool,
    rng: random.Random,
) -> Deck:
    ids = _repair_card_ids(ids, pool, rng)
    evo, her = _repair_forms(ids, evolved, hero, pool, rng)
    return Deck(cards=tuple(pool.get(cid) for cid in ids), evolved=evo, hero=her)


def random_deck(pool: CardPool, rng: random.Random) -> Deck:
    ids = _repair_card_ids(rng.sample(pool.all_ids, config.DECK_SIZE), pool, rng)
    evolvable = [cid for cid in ids if cid in pool.evolvable_ids]
    hero_eligible = [
        cid for cid in ids if cid in pool.hero_set and cid not in pool.champion_set
    ]
    rng.shuffle(evolvable)
    rng.shuffle(hero_eligible)
    evolved = set(evolvable[: rng.randint(0, len(evolvable))])
    hero = set(hero_eligible[: rng.randint(0, len(hero_eligible))])
    return _build_deck(ids, evolved, hero, pool, rng)


class GeneticAlgorithm:
    def __init__(
        self,
        pool: CardPool,
        fitness_fn: FitnessFn,
        *,
        population_size: int = config.POPULATION_SIZE,
        generations: int = config.GENERATIONS,
        elitism: int = config.ELITISM,
        tournament_size: int = config.TOURNAMENT_SIZE,
        crossover_rate: float = config.CROSSOVER_RATE,
        mutation_rate: float = config.MUTATION_RATE,
        seed: int | None = config.RANDOM_SEED,
    ):
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        self.pool = pool
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.generations = generations
        self.elitism = min(elitism, population_size)
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)
        self._fitness_cache: dict[tuple, float] = {}

    def fitness(self, deck: Deck) -> float:
        k = deck.key
        cached = self._fitness_cache.get(k)
        if cached is None:
            cached = self.fitness_fn(deck)
            self._fitness_cache[k] = cached
        return cached

    def _tournament(self, population: list[Deck]) -> Deck:
        k = min(self.tournament_size, len(population))
        return max(self.rng.sample(population, k), key=self.fitness)

    def _crossover(self, a: Deck, b: Deck) -> Deck:
        gene_pool = list(a.card_ids | b.card_ids)
        self.rng.shuffle(gene_pool)
        return _build_deck(
            gene_pool,
            set(a.evolved | b.evolved),
            set(a.hero | b.hero),
            self.pool,
            self.rng,
        )

    def _flip_form(self, ids, claim, other, eligible) -> None:
        """Toggle a random eligible card's membership in `claim`, clearing `other`
        (so an evo+hero card like Knight flips between its two forms)."""
        targets = [cid for cid in ids if cid in eligible]
        if not targets:
            return
        cid = self.rng.choice(targets)
        if cid in claim:
            claim.discard(cid)
        else:
            claim.add(cid)
            other.discard(cid)

    def _mutate(self, deck: Deck) -> Deck:
        ids = [c.id for c in deck.cards]
        evolved = set(deck.evolved)
        hero = set(deck.hero)
        roll = self.rng.random()
        if roll < 0.6:
            # Replace a card; drop whatever form the outgoing card held.
            idx = self.rng.randrange(len(ids))
            evolved.discard(ids[idx])
            hero.discard(ids[idx])
            ids[idx] = self.rng.choice([c for c in self.pool.all_ids if c not in ids])
        elif roll < 0.8:
            self._flip_form(ids, evolved, hero, self.pool.evolvable_ids)
        else:
            # Champions are forced hero by the repair, so only flip other heroes.
            self._flip_form(
                ids, hero, evolved, self.pool.hero_set - self.pool.champion_set
            )
        return _build_deck(ids, evolved, hero, self.pool, self.rng)

    def run(
        self, on_generation: Callable[[int, list[Deck]], None] | None = None
    ) -> Deck:
        population = [
            random_deck(self.pool, self.rng) for _ in range(self.population_size)
        ]
        best: Deck | None = None

        for gen in range(self.generations):
            population.sort(key=self.fitness, reverse=True)
            if best is None or self.fitness(population[0]) > self.fitness(best):
                best = population[0]
            if on_generation is not None:
                on_generation(gen, population)

            next_gen = population[: self.elitism]
            while len(next_gen) < self.population_size:
                parent1 = self._tournament(population)
                if self.rng.random() < self.crossover_rate:
                    child = self._crossover(parent1, self._tournament(population))
                else:
                    child = parent1
                if self.rng.random() < self.mutation_rate:
                    child = self._mutate(child)
                next_gen.append(child)
            population = next_gen

        assert best is not None
        return best
