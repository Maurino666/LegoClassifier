import torch

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Modello salvato in {path}")
