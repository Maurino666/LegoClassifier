from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, device, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{num_epochs}", total=len(train_loader))

        for inputs, labels in progress_bar:
            # Trasferisci l'intero batch in VRAM
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({
                "Loss": f"{running_loss / (total or 1):.4f}",
                "Acc": f"{100. * correct / (total or 1):.2f}%"
            })

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%")
