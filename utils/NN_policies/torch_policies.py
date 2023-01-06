import torch 

def MLP(inputDim, outputDim, hiddenLayersDims, activation, bias):
    """This function creates a multi-layer perceptron.

    Args:
        inputDim (int): The input dimension of the MLP.
        outputDim (int): The output dimension of the MLP.
        hiddenLayersDims (list): A list of integers that represent the number of hidden layers and their dimensions.
        activation (torch.nn.functional, optional): The activation function to use. Defaults to tanh.

    Returns:
        torch.nn.Module: The MLP.
    """
    layers = []
    layers.append(torch.nn.Linear(inputDim, hiddenLayersDims[0], bias=bias))
    layers.append(activation)
    for i in range(1, len(hiddenLayersDims)):
        layers.append(torch.nn.Linear(hiddenLayersDims[i-1], hiddenLayersDims[i], bias=bias))
        layers.append(activation)
    layers.append(torch.nn.Linear(hiddenLayersDims[-1], outputDim, bias=bias))
    return torch.nn.Sequential(*layers)


class CNN(torch.nn.Module):
    "CNN+MLP with n=input_channels frames as input. Non-activated last layer's output"
    def __init__(self, input_channels, action_space_dim, hidden_dim, bias):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=3, stride=1, bias=bias)   
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=2, bias=bias)
        self.fc1 = torch.nn.Linear(648, hidden_dim, bias=bias)  # resized to 84x84
        self.fc2 = torch.nn.Linear(hidden_dim, action_space_dim, bias=bias)
    
    def forward(self, observation):
        state = torch.unsqueeze(observation,0)
        x = self.pool(torch.tanh(self.conv1(state)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1)
        x = torch.tanh(self.fc1(x))   
        x = self.fc2(x)
        return x
        
    def get_weights(self):
        return  torch.nn.utils.parameters_to_vector(self.parameters()).detach().numpy()