from utils.plotting import plot_history
import torch
from utils.device import get_device, clear_device_cache
from datasets.main import get_mnist_loaders
# from models.mlp import MLP
# from models.cnn import CNN
from models.factory import create_model
from models.base import BaseNN

from engine.trainer import evaluate, fit
from callbacks.early_stopping import EarlyStopping

def main(): 
    ## HIPERPARAMETROS ##
    batch_size = 128 # No elementos a procesar al mismo tiempo
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 40 # callback - early stopping - regularizar
    # ReduceRLROnPlateau - Cambia el ratio de aprendizaje cuando la métrica de validación deja de mejorar
    patience = 6
    min_delta = 1e-3 # criterio de mejora mínima para considerar que la métrica ha mejorado

    # clear_device_cache()
    device = get_device() 
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_mnist_loaders(data_dir="./data", batch_size=batch_size)

    ## MODEL ##

    model = create_model(model_name='cnn', num_classes=10) # 'mlp' o 'cnn' 'vgg11'
    model = model.to(device)

    ## Train
    criterion = torch.nn.CrossEntropyLoss() # funcion de perdida

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=0.5, 
                                                           patience=2) # Reduce el learning rate cuando la métrica de validación deja de mejorar

    history = fit(
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=num_epochs,
        scheduler=scheduler,
        early_stopping=early_stopping
    )

    torch.save(model.state_dict(), "artifacts/best_model.pth")  # Guardar el modelo entrenado

    test_loss = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test Loss: {test_loss:.4f}")
    plot_history(history)




if __name__ == "__main__":
    main()