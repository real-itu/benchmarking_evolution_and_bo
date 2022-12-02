# Benchmarking evolution against Bayesian Optimization

This repo contains a couple of examples of evolutionary strategies, plus a very vanilla implementation of Bayesian Optimization. It was inspired by David Ha's blogpost on visualizing evolutionary strategies.

## Instructions

### Create a new environment and install dependencies

Start by installing the requirements in a new enviroment, maybe something like

```
conda create -n evo-benchmark python=3.9
conda activate evo-benchmark
pip install -r requirements.txt
```

### Play with the evolution scripts

Using CMA-ES as an example, you can go to the `main` and comment in/out any of the objective functions you'd like to test.

```python
if __name__ == "__main__":
    # Defining the bounds for the specific obj. functions
    obj_function = shifted_sphere
    limits = [-4.0, 4.0]
    exploration = 0.1

    # obj_function = easom
    # limits = [np.pi - 4, np.pi + 4]
    # exploration = 0.1

    # obj_function = cross_in_tray
    # limits = [-10, 10]
    # exploration = 0.1

    # obj_function = egg_holder
    # limits = [-512, 512]
    # exploration = 10.0
```

We provide four objective functions for testing. Running each of these scripts should give you an animation with the obj. function in the background. `exploration` is usually the (initial) standard deviation.

### Playing with the Bayesian Optimization script

If you run `bayesian_optimization.py`, you'll see not only the objective function, but also the GP model's predictions and the acquisition function.

[TODO: add a short description of the algorithms and on B.O.]
