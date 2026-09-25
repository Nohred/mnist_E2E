import torch.nn as nn
from models.base import BaseNN

class MLP(BaseNN): # herencia
    def __init__(self, name):
        super(MLP, self).__init__(name='mlp')
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), # Image(28x28), 128 Neurons
            nn.ReLU(),
            nn. Linear(128, 10), # 10 outputs neurons = 10 clases
            # logits -> raw outputs
        )

    def forward(self, x):
        return self.network(x)
    
