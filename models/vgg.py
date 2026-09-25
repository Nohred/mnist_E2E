import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg11, VGG11_Weights

from models.base import BaseNN

class VGG11(BaseNN):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__(name='vgg11')
        # Crear modelo y cargar pesos preentrenados
        self.network = vgg11(weights=VGG11_Weights.DEFAULT)

        for parameter in self.network.features.parameters():
            parameter.requires_grad = False 
            # Congela los pesos de las capas convolucionales para que 
            # no se actualicen durante el entrenamiento

        # Obtiene el número de características de la última capa
        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(in_features, num_classes) # Reemplaza la última capa de clasificación para que tenga el número de clases deseado

        self.register_buffer(
            "mnist_std",
            torch.tensor([0.3081]).view(1, 1, 1, 1) # Define la desviación estándar de MNIST para normalizar las imágenes
        )

        self.register_buffer(
            "mnist_mean",
            torch.tensor([0.1307]).view(1, 1, 1, 1) # Define la media de MNIST para normalizar las imágenes
        )


        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) # Define la desviación estándar de ImageNet para normalizar las imágenes
        )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) # Define la media de ImageNet para normalizar las imágenes
        )

    def forward(self, x):
        x = x* self.mnist_std + self.mnist_mean # desnormaliza las imágenes de MNIST
        x = x.repeat(1, 3, 1, 1) # Convierte las imágenes de MNIST de 1 canal a 3 canales para que sean compatibles con V
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False) # Redimensiona las imágenes a 224x224 píxeles
        x = (x - self.imagenet_mean) / self.imagenet_std # Normaliza las imágenes de MNIST con la media y desviación estándar de ImageNet

        return self.network(x)