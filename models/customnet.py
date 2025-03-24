import torch
from torch import nn
import numpy as np

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self, num_classes=200):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.convStack = nn.Sequential(                      
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 256, kernel_size=3, stride=2),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Flatten()
        )

        self.linear_stack = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 200)
        )

    def forward(self, x):
        x = self.convStack(x)
        logits = self.linear_stack(x)
        return logits

# # Test with a sample input
# model = CustomNet()
# sample_input = torch.randn(1, 3, 224, 224)  # Example batch size 1
# output = model(sample_input)

# print(output.shape)  #
# print(np.prod((output.shape)))


