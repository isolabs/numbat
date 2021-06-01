
import math

import torch
import torch.nn as nn

class FineTuner(nn.Module):
    """
    The fine tuner fundamentally works by wrapping a vision transformer and using its CLS
    token outputs to perform classification with a single FC layer
    """
    
    def __init__(
        self, 
        vit, 
        vit_embed_dim,
        hidden_sizes,
        n_classes 
    ):
        """
        Instantiate the 'extra' classification layers
        """

        super().__init__()

        # Save the VIT
        self.vit = vit

        # Now the hidden layers (usually very few - the point being we can do 
        # this cheaply)
        self.layers = []
        self.activations   = []
        sizes = [vit_embed_dim] + hidden_sizes + [n_classes]

        self.num_layers = len(sizes) - 1
        for i in range(self.num_layers):
            # Create the hidden layer
            layer = nn.Linear(sizes[i], sizes[i+1])
            self.layers.append(layer)

            # What is the activation that happens after this layer? No final
            # activation
            if i != self.num_layers - 1:
                self.activations.append(nn.GELU())

        # Ensure properly registered, and will be visible by all module methods
        self.layers      = nn.ModuleList(self.layers)
        self.activations = nn.ModuleList(self.activations)

        # Softmax for final classification. At this stage shape is (B, Classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Through the VIT and then through the classification layers
        """
        
        B, C, W, H = x.shape

        # Apply the VIT (includes patch embedder at the start)
        cls_tokens = self.vit(x)
        y = cls_tokens

        # Shape is now (B, E). Run it through the linear
        for i in range(self.num_layers):
            # Create the hidden layer
            y = self.layers[i](y)

            # What is the activation that happens after this layer? No final
            # activation
            if i != self.num_layers - 1:
                y = self.activations[i](y)

        # End with the softmax and return shape (B, Classes)
        y = self.softmax(y)

        return y