"""
This module contains all the search algorithms we will include in our
comparison. They're exported at this level too, making imports easier.
"""
# Baselines
from .baselines.random_search import RandomSearch

# B.O.
from .bayesian_optimization.bayesian_optimization import BayesianOptimization

# Evolutionary strategies
from .evolutionary_strategies.simple_evolution_strategy import SimpleEvolutionStrategy
from .evolutionary_strategies.snes import SNES
from .evolutionary_strategies.pgpe import PGPE
from .evolutionary_strategies.cma_es import CMA_ES
