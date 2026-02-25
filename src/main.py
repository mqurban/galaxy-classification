import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys
# update 
# Add the parent directory to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.data_generator import generate_dataset
from src.model import GalaxyCNN

# --- Configuration ---
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
NUM_SAMPLES = 5000  # Number of synthetic images to generate
MODEL_SAVE_PATH = "models/galaxy_classifier.pth"

def train_and_evaluate():
    """
    Main function to run the galaxy classification pipeline using PyTorch.
    """
    
    # 1. Generate Synthetic Data
    print("\n--- Generating Synthetic Galaxy Data ---")
    X, y = generate_dataset(NUM_SAMPLES, "data")
    
    # Convert to PyTorch Tensors
    # PyTorch expects (Batch, Channels, Height, Width)
    # Our X is (Batch, H, W, 1), so we need to permute
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Split into Train and Validation sets (80/20)
    split_idx = int(NUM_SAMPLES * 0.8)
    
    train_dataset = TensorDataset(X_tensor[:split_idx], y_tensor[:split_idx])
    val_dataset = TensorDataset(X_tensor[split_idx:], y_tensor[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    
    # 2. Build the CNN Model
    print("\n--- Building CNN Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GalaxyCNN().to(device)
    
    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train the Model
    print("\n--- Starting Training ---")
    
    # Create directory for saving models if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    
    # 5. Visualize Results
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    visualize_predictions(model, val_loader, device)

def plot_training_history(train_loss, val_loss, train_acc, val_acc):
    """Plots training/validation accuracy and loss."""
    epochs_range = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig("training_results.png")
    print("Training plots saved to 'training_results.png'")

def visualize_predictions(model, val_loader, device, num_samples=5):
    """Visualizes model predictions on a few validation samples."""
    model.eval()
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)
    
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        # Convert back to (H, W) for plotting: (1, 64, 64) -> (64, 64)
        img_np = images[i].cpu().squeeze().numpy()
        plt.imshow(img_np, cmap='gray')
        
        predicted_label = predictions[i].item()
        true_label = labels[i].item()
        
        label_map = {0: "Elliptical", 1: "Spiral"}
        color = 'green' if predicted_label == true_label else 'red'
        
        plt.title(f"Pred: {label_map[predicted_label]}\nTrue: {label_map[true_label]}", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("prediction_samples.png")
    print("Prediction samples saved to 'prediction_samples.png'")

if __name__ == "__main__":
    train_and_evaluate()
