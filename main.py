import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from models.customnet import CustomNet
from torch import nn
from train import train
from eval import validate

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

# root/{classX}/x001.jpg

tiny_imagenet_dataset_train = ImageFolder(root='data/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='data/tiny-imagenet-200/val', transform=transform)

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=128, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=128, shuffle=False, num_workers=2)

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)


print(f'Best validation accuracy: {best_acc:.2f}%')
