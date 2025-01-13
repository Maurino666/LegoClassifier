import torch
from src.data_preprocessing import get_data_loader
from src.model import SimpleCNN
from src.train import train_model
from src.utils import save_model

def main():
    dataset_path = "Dataset/D1"
    batch_size = 128
    num_epochs = 5

    # Usa la GPU con CUDA se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # Carica il dataset
    train_loader, classes = get_data_loader(dataset_path, batch_size)
    print("Caricate classi...")
    # Inizializza il modello
    model = SimpleCNN(num_classes=len(classes)).to(device)
    print("Modello inizializzato...")
    # Addestra il modello
    train_model(model, train_loader, device, num_epochs)
    print("Addestramento completato!")
    # Salva il modello
    save_model(model, path="simple_cnn.pth")

if __name__ == "__main__":
    main()
