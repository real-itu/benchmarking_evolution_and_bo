"""
Implements a basic evolution strategy using torch,
and tests it on the objective functions.
"""
import torch
import matplotlib.pyplot as plt

from search_algorithms.evolutionary_strategies.simple_evolution_strategy import (
    SimpleEvolutionStrategy,
)
from objective_functions.objective_function import ObjectiveFunction
from utils.visualization.evolutionary_strategies import plot_algorithm
from utils.seeding import seed_python_numpy_torch_cuda


if __name__ == "__main__":
    # Defining the function to optimize
    # name = "alpine_01"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"
    # model_specification = None 
    # n_dims = 5  # Whatever int your mind desires (in most cases)
    # seed_objective = None # Seed for sthocastic objective functions like RL tasks
    # seed_search = None # Seed for the search algorithm, if set to None a different seed is used for each run (needed for reproducibility)
    # visualize = False
    
    name = "CartPole-v1"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"
    model_specification = {'library' : 'torch', 'model' : 'feedforward', 'hidden_layers' : [8,8], 'activation': torch.nn.Tanh(), 'bias' : True, 'dtype' : torch.float64}  # None or a dictionary defining the NN
    n_dims = None  # Whatever int your mind desires (in most cases)
    seed_objective = None # Seed for sthocastic objective functions like RL tasks, is set to None, a different seed is used for each run
    seed_search = None # Seed for the search algorithm, if set to None a different seed is used for each run (needed for reproducibility)
    visualize = False

    # Seeding evertyhing (except for the RL objective function)
    seed_python_numpy_torch_cuda(seed_search)

    # Hyperparameters for the search
    # Num. of generations and population size
    n_generations = 30
    population_size = 20

    # Breaking as soon as the best fitness is this close to the actual optima
    # in absolute terms.
    tolerance_for_optima = 1e-3

    # Do we actually want to break?
    break_when_close_to_optima = True

    # Initial covariance, needs to be modified depending on {name}
    exploration = 0.2

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective_function = ObjectiveFunction(name=name, n_dims=n_dims, model=model_specification, seed=seed_objective)
    limits = objective_function.limits
    solution_length = objective_function.solution_length
    
    simple_evo = SimpleEvolutionStrategy(
        objective_function=objective_function,
        population_size=population_size,
        exploration=exploration,
        solution_length=solution_length,
        limits=limits,
        initial_best = torch.zeros(solution_length)
    )

    if model_specification is None and visualize:
        fig, ax = plt.subplots(1, 1)
    for _ in range(n_generations):
        # Save the current mean for plotting
        current_mean = simple_evo.get_current_best()

        # Run a step
        samples = simple_evo.step()

        # Visualize
        if model_specification is None and visualize:
            plot_algorithm(
                ax=ax,
                obj_function=objective_function,
                limits=limits,
                current_best=current_mean,
                population=samples,
                next_best=simple_evo.best,
            )

            plt.pause(0.01)
            ax.clear()

        # (uncounted) best fitness evaluation
        best_fitness = objective_function(simple_evo.get_current_best())
        print(f"Best fitness: {best_fitness}")

        if (model_specification is None and (torch.isclose(best_fitness, objective_function.optima, atol=tolerance_for_optima) and break_when_close_to_optima)):
            print(f"Found a good-enough optima, breaking.")
            break

    print(
        f"The obj. function was evaluated in {simple_evo.objective_function.n_points} points ({simple_evo.objective_function.calls} calls)"
    )
