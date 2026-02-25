import torch
import torch.nn as nn
import torch.nn.functional as F

class GalaxyCNN(nn.Module):
    """
    Convolutional Neural Network for Galaxy Morphology Classification.
    Input: (Batch, 1, 64, 64) grayscale images.
    Output: (Batch, 2) logits for Elliptical vs Spiral.
    """
    def __init__(self):
        super(GalaxyCNN, self).__init__()
        
        # Convolutional Block 1
        # Input: 1 channel (grayscale), Output: 32 channels, Kernel: 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 2
        # Input: 32 channels, Output: 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Convolutional Block 3
        # Input: 64 channels, Output: 64 channels
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Fully Connected Layers
        # Image is 64x64. After 2 pooling layers (if we add another pool), size reduces.
        # Let's track dimensions:
        # Input: 64x64
        # Conv1 -> 64x64 -> Pool -> 32x32
        # Conv2 -> 32x32 -> Pool -> 16x16
        # Conv3 -> 16x16
        # Flatten: 64 channels * 16 * 16
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))
        # Block 2
        x = self.pool(F.relu(self.conv2(x)))
        # Block 3 (No pooling here to keep more features, or we can add one)
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output logits (no softmax here, CrossEntropyLoss handles it)
        return x

if __name__ == "__main__":
    model = GalaxyCNN()
    print(model)
    # Test with a random input
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    print("Output shape:", output.shape)
