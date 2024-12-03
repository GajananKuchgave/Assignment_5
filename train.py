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
    
    # Changed optimizer to SGD with specified parameters
    optimizer = optim.SGD(model.parameters(), 
                         lr=0.05,          # Learning rate
                         momentum=0.9)      # Momentum
    
    print("\nOptimizer settings:")
    print(f"- Type: SGD")
    print(f"- Learning rate: 0.05")
    print(f"- Momentum: 0.9")
    
    # Training
    model.train()
    print("\nStarting training...")
    
    # Initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar for the entire epoch
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Calculate current accuracy
        accuracy = 100 * correct / total
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'accuracy': f'{accuracy:.2f}%'
        })
        
        # Print detailed stats every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'\nBatch [{batch_idx + 1}/{len(train_loader)}]')
            print(f'Loss: {running_loss:.4f}')
            print(f'Training Accuracy: {accuracy:.2f}%')
    
    # Print final training stats
    final_accuracy = 100 * correct / total
    print(f'\nFinal Training Stats:')
    print(f'Loss: {running_loss:.4f}')
    print(f'Accuracy: {final_accuracy:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_mnist_{timestamp}_acc{final_accuracy:.1f}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    return save_path

if __name__ == '__main__':
    train_model() 