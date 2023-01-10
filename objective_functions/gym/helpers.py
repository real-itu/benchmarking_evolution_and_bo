import gym
from typing import Callable, Tuple, Union, TypedDict
import numpy as np
from utils.types import Model

from utils.NN_policies.torch_policies import MLP, CNN


def dimensions_env(environment : str):
    """
    Look up observation and action space dimension
    """
    from gym.spaces import Discrete, Box

    env = gym.make(environment)
    if len(env.observation_space.shape) == 3:  # Pixel-based environment
        pixel_env = True
        input_dim = 3  # number of channels of the observation space
    elif len(env.observation_space.shape) == 1:  # State-based environment
        pixel_env = False
        input_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, Discrete):
        pixel_env = False
        input_dim = env.observation_space.n
    else:
        raise ValueError("Observation space not supported")

    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    else:
        raise ValueError("Action space not supported")
    return input_dim, action_dim, pixel_env


def nb_parameters(environment : str, model_dict : Model):
    """
    Calculate number of parameters of a torch model for a given Gym environment
    """
    input_dim, action_dim, pixel_env = dimensions_env(environment)

    # Initilise the model
    if not pixel_env:
        model = MLP(input_dim, action_dim, model_dict['hidden_layers'], activation=model_dict['activation'], bias=model_dict['bias'])
    else:
        raise NotImplementedError
        # TODO implement CNN variable number of hidden layers in the mlp
        # model = CNN(input_dim, action_dim, 32, bias=model_dict['bias'])

    nb_parameters = sum(p.numel() for p in model.parameters())
    return nb_parameters



class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float64
        )

    def observation(self, observation):
        # This undoes the memory optimization, use with smaller replay buffers only.
        return np.array(observation).astype(np.float64) / 255.0