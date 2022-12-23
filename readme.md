# Benchmarking evolution against Bayesian Optimization

This repo contains a couple of examples of evolutionary strategies, plus a very vanilla implementation of Bayesian Optimization. It was inspired by David Ha's blogpost on visualizing evolutionary strategies.

We aim at benchmarking how efficients these algorithms are in terms of the number of evaluations of the objective function. In real world scenarios, calls to the objective function are expensive (e.g. having an agent play a given level, or running a protein folder).

## Instructions

### Create a new environment and install dependencies

Start by installing the requirements in a new enviroment, maybe something like

```
conda create -n evo-benchmark python=3.9
conda activate evo-benchmark
pip install -r requirements.txt
```

### Add the working folder to your PYTHONPATH

...

## Play with the evolution scripts

The objective functions are implemented in `objective_functions.py`, and are wrapped by a single class that abstracts properties like `limits` and `optima_location` for each one of them. Check the implmenentation of `ObjectiveFunction` therein.

With this, it is quite easy to change the objective function to optimize: using CMA-ES as an example, you can go to the `main` and change the `name` variable to any of the ones implemented (`"shifted_sphere"`, `"easom"`, `"cross_in_tray"` and `"egg_holder"` at time of writing).

```python
if __name__ == "__main__":
    # Defining the function to optimize
    name = "shifted_sphere"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"
```

We also expose several hyperparameters for the search, which you can also find in the main.

## The calls of the objective function are being counted

You will see that we wrap the objective function with a counter:

```python
@counted
def obj_function_counted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return obj_function(x, y)
```

Each time we call `obj_function_counted`, we maintain a count of the number of calls and the number of points the objective function was called. The `counter` decorator is implemented as follows:

```python
def counted(obj_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    """
    Counts on how many points the obj function was evaluated
    """

    def wrapped(x: torch.Tensor, y: torch.Tensor):
        wrapped.calls += 1

        if len(x.shape) == 0:
            # We are evaluating at a single point [x, y]
            wrapped.n_points += 1
        else:
            # We are evaluating at a several points.
            wrapped.n_points += len(x)

        return obj_function(x, y)

    wrapped.calls = 0
    wrapped.n_points = 0
    return wrapped
```


## Playing with the Bayesian Optimization script

If you run `bayesian_optimization.py`, you'll see not only the objective function, but also the GP model's predictions and the acquisition function.

[TODO: add a short description of the algorithms and on B.O.]
