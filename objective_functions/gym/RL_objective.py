import torch
import gym
from typing import Callable, Tuple, Union, TypedDict
import numpy as np

from utils.NN_policies.torch_policies import MLP, CNN
from .helpers import nbParameters, dimensions_env, ScaledFloatFrame

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)

class ObjectiveRLGym:
    def __init__(self, environment: str, model: TypedDict, seed: int, maximize: bool, limits: Tuple[float, float]) -> None:
        self.seed = seed
        self.solution_length = nbParameters(environment, model)
        self.known_optima: bool = False
        self.maximize: bool = True
        assert (maximize), "You want negative reward?"
        self.limits = (-1, 1)
        self.input_dim, self.action_dim, self.pixel_env = dimensions_env(environment)
        self.device = "cpu"

        # Instantiate model
        if model["model"] == "feedforward":
            if self.pixel_env:
                raise NotImplementedError
                self.model = CNN(self.input_dim, self.action_dim, model["hidden_layers"], bias=model["bias"])
            else:
                self.model = MLP(self.input_dim, self.action_dim, model["hidden_layers"], activation=model["activation"], bias=model["bias"])
        else:
            raise ValueError("Unknown model type: {}".format(model["type"]))
        # Instatiate and seeding the environment
        env = gym.make(environment)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        # Resize and normilise input for pixel environments
        if self.pixel_env == True:
            env = gym.wrappers.ResizeObservation(env, 84)
            env = ScaledFloatFrame(env)
        
        self.environment = env
        
    def env_rollout(self) -> torch.Tensor:
        
        with torch.no_grad():
            
            observation = self.environment.reset()
            seed = np.random.randint(0, 2**32 - 1) if self.seed is None else self.seed
            self.environment.seed(seed)
            done = False
            episodeReward = 0
            while not done:

                # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
                if isinstance(self.environment.observation_space, gym.spaces.Discrete):
                    observation = (observation == torch.arange(self.environment.observation_space.n)).float()
                # Swap axes to the correct order for pytorch
                if self.pixel_env:
                    observation = np.swapaxes(observation, 0, 2)  # resizing hardcoded to (3, 84, 84)
                observation = torch.from_numpy(observation).double().to(self.device)

                # Model predicts the action
                action = self.model(observation).to("cpu").detach().numpy()
                # action = env.action_space.sample()

                # Bound the action or convert it to a discrete action
                if isinstance(self.environment.action_space, gym.spaces.Box):
                    action = np.clip(action, self.environment.action_space.low, self.environment.action_space.high)
                elif isinstance(self.environment.action_space, gym.spaces.Discrete):
                    action = np.argmax(action)

                # Forward step in the envionrment
                observation, reward, done, info = self.environment.step(action)

                # Save reward
                episodeReward += reward

                # Render the environment (for debugging)
                # env.render()
            
        return torch.tensor(episodeReward, dtype=torch.float64)
        
    
    def evaluate_objective(self, x: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            
            population_fitnesses = [] # TODO: parallelize this loop
            for individual_solution in x:
            
                # Load the final optimised parameters into the model
                torch.nn.utils.vector_to_parameters(individual_solution, self.model.parameters())
                
                # Environment evaluation
                solution_fitness = self.env_rollout()
                
                # Append to population fitnesses
                population_fitnesses.append(solution_fitness)
                
        return torch.tensor(population_fitnesses, dtype=torch.float64)


            
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        return self.evaluate_objective(x)
    
