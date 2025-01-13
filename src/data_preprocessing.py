from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loader(dataset_path, batch_size=64, num_workers=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    return loader, dataset.classes
