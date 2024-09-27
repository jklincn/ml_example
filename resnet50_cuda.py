import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_epochs = 1

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_set = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=4
)

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

total_data_transfer_time = 0.0
iteration = 0

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        torch.cuda.synchronize()
        data_transfer_start_time = time.time()
        images, labels = images.to(device), labels.to(device)
        torch.cuda.synchronize()
        data_transfer_end_time = time.time()

        data_transfer_time = data_transfer_end_time - data_transfer_start_time
        total_data_transfer_time += data_transfer_time

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iteration += 1

        if iteration % 100 == 0:
            print(f"Iteration {iteration} - Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

end_time = time.time()

print(f"Total time spent: {(end_time - start_time):.2f} seconds")
print(f"Total data transfer time: {total_data_transfer_time:.2f} seconds")
