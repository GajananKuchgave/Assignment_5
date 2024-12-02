import torch
import pytest
from torchvision import datasets, transforms
import os
from model import MNISTModel
from tqdm import tqdm

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, should be < 100000"

def test_model_accuracy():
    device = torch.device('cpu')
    model = MNISTModel().to(device)
    
    # Load the latest model
    model_files = [f for f in os.listdir('.') if f.startswith('model_mnist_') and f.endswith('.pth')]
    if not model_files:
        pytest.skip("No model file found")
    
    latest_model = max(model_files)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    
    # Test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("\nLoading test dataset...")
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    print("Running accuracy test...")
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%" 