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
    """Coerce ids into DECK_SIZE distinct cards with at most MAX_CHAMPIONS champions."""
    deduped: list[int] = []
    seen: set[int] = set()
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            deduped.append(cid)
    ids = deduped

    champs = [c for c in ids if c in pool.champion_set]
    if len(champs) > config.MAX_CHAMPIONS:
        keep = set(champs[: config.MAX_CHAMPIONS])
        ids = [c for c in ids if c not in pool.champion_set or c in keep]

    ids = ids[: config.DECK_SIZE]

    current = set(ids)
    while len(ids) < config.DECK_SIZE:
        cand = rng.choice(pool.all_ids)
        if cand in current:
            continue
        if (
            cand in pool.champion_set
            and sum(c in pool.champion_set for c in ids) >= config.MAX_CHAMPIONS
        ):
            continue
        ids.append(cand)
        current.add(cand)
    return ids


def _repair_evolutions(
    deck_ids: list[int], evolved: set[int], pool: CardPool, rng: random.Random
) -> frozenset[int]:
    """Keep only evolvable cards in the deck, capped by the shared wild slot."""
    legal = [cid for cid in evolved if cid in deck_ids and cid in pool.evolvable_ids]
    num_champions = sum(1 for cid in deck_ids if cid in pool.champion_set)
    cap = config.max_evolutions_allowed(num_champions)
    if len(legal) > cap:
        legal = rng.sample(legal, cap)
    return frozenset(legal)


def _build_deck(
    ids: list[int], evolved: set[int], pool: CardPool, rng: random.Random
) -> Deck:
    ids = _repair_card_ids(ids, pool, rng)
    evo = _repair_evolutions(ids, evolved, pool, rng)
    return Deck(cards=tuple(pool.get(cid) for cid in ids), evolved=evo)


def random_deck(pool: CardPool, rng: random.Random) -> Deck:
    ids = _repair_card_ids(rng.sample(pool.all_ids, config.DECK_SIZE), pool, rng)
    evolvable_in_deck = [cid for cid in ids if cid in pool.evolvable_ids]
    rng.shuffle(evolvable_in_deck)
    cap = config.max_evolutions_allowed(
        sum(1 for cid in ids if cid in pool.champion_set)
    )
    evolved = set(evolvable_in_deck[: rng.randint(0, cap)])
    return _build_deck(ids, evolved, pool, rng)


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
        return _build_deck(gene_pool, set(a.evolved | b.evolved), self.pool, self.rng)

    def _mutate(self, deck: Deck) -> Deck:
        ids = [c.id for c in deck.cards]
        evolved = set(deck.evolved)
        if self.rng.random() < 0.7:
            idx = self.rng.randrange(len(ids))
            evolved.discard(ids[idx])
            for _ in range(50):
                cand = self.rng.choice(self.pool.all_ids)
                if cand not in ids:
                    ids[idx] = cand
                    break
        else:
            evolvable = [cid for cid in ids if cid in self.pool.evolvable_ids]
            if evolvable:
                target = self.rng.choice(evolvable)
                evolved.discard(target) if target in evolved else evolved.add(target)
        return _build_deck(ids, evolved, self.pool, self.rng)

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
