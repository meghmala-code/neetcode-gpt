import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std= (2/(fan_in + fan_out))**(1/2)
        weights= torch.randn(fan_out, fan_in)*std
        return torch.round(weights,decimals=4).tolist()
        pass

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std= (2/fan_in)**(1/2)
        weights= torch.randn(fan_out,fan_in)*std
        return torch.round(weights,decimals=4).tolist()
        pass

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
       
        weight_matrices=[]
        current_in= input_dim
        for i in range(num_layers):
            if init_type =="kaiming":
                std = (2 / current_in)**0.5
                weights = torch.randn(hidden_dim, current_in) * std
                weight_matrices.append(weights)
            elif init_type == "xavier":
                std = (2 / (current_in + hidden_dim))**0.5
                weights = torch.randn(hidden_dim, current_in) * std
                weight_matrices.append(weights)
            elif init_type == "random":
                std = 1.0
                weights = torch.randn(hidden_dim, current_in) * std
                weight_matrices.append(weights)
            
            #weights = torch.randn(hidden_dim, current_in) * std
            #weight_matrices.append(weights)
            current_in= hidden_dim
        stds=[]
        x = torch.randn(1, input_dim)
        for weights in weight_matrices:
            x=torch.matmul(x,weights.t())
            x= torch.relu(x)
            current_std = torch.std(x).item()
            stds.append(round(current_std,2))
        return stds
            
        pass