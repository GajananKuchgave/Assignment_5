import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os
from model import MNISTModel
from tqdm import tqdm

def train_model():
    # Force CPU usage
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets with progress bar for download
    print("Loading datasets...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training
    model.train()
    print("\nStarting training...")
    
    # Create progress bar for the entire epoch
    pbar = tqdm(train_loader, desc='Training')
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update running loss and progress bar
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        pbar.set_postfix({'loss': f'{running_loss:.4f}'})
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_mnist_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    return save_path

if __name__ == '__main__':
    train_model() 