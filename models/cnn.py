import torch.nn as nn
from models.base import BaseNN
class CNN(BaseNN): # herencia

    def __init__(self, num_classes=10): # Constructor

        super(CNN, self).__init__(name='cnn') # Llamada al constructor de la clase base nn.Module
        
        self.features = nn.Sequential( # Extractor de características (CNN)
            nn.Conv2d( # I -> 28x28
                1, # 
                16, # Numero de filtros
                kernel_size=3,
                stride=1, # Desplazamiento del kernel en la imagen
                padding=1 # Agrega ceros alrededor de la imagen para mantener el tamaño de salida igual al de entrada
            ), # O -> 28x28x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # O -> 14x14x16
            nn.Conv2d( # I -> 14x14x16
                16, # Numero de filtros
                32, # Numero de filtros
                kernel_size=3,
                stride=1,
                padding=1 # Agrega ceros alrededor de la imagen para mantener el tamaño de salida igual al de entrada
            ), # O -> 14x14x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # O -> 7x7x32
        )

        self.classifier = nn.Sequential( # Cabeza de Clasificación (MLP)
            nn.Flatten(),
            nn.Linear(7*7*32, 128), # 32 filters, 7x7 feature maps, 128 Neurons
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x