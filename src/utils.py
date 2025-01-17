import torch

DEFAULT_MODEL_PATH = "model.pth"

def save_model(model, path=DEFAULT_MODEL_PATH):
    """
    Saves only the model weights (state_dict).

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The path to save the model file.

    """
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

def load_model(model_class, path=DEFAULT_MODEL_PATH, device=torch.device("cuda")):
        """
        Loads the model weights (state_dict) into a given model class.

        Args:
            model_class (callable): A callable that returns an instance of the model class.
            path (str): The path to the saved model file.
            device (torch.device): The device to load the model onto (default is CPU).

        Returns:
            torch.nn.Module: The model with loaded weights.
        """
        # Instantiate the model
        model = model_class()

        # Load the state_dict from the file
        model.load_state_dict(torch.load(path, map_location=device))

        # Move the model to the specified device
        model.to(device)

        print(f"Model loaded from {path}")
        return model
