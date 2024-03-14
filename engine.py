import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def create_data_loaders(data_path, batch_size=32, num_workers=0, train_transform=None, test_transform=None):
    """
    Create data loaders for training and testing.

    Args:
        data_path (str): Path to the root directory containing train and test data folders.
        batch_size (int): Number of samples in each batch.
        num_workers (int): Number of subprocesses to use for data loading.
        train_transform (callable): A function/transform to preprocess the training images.
        test_transform (callable): A function/transform to preprocess the testing images.

    Returns:
        tuple: A tuple containing train_loader, test_loader, class_names, and class_dict.
    """
    train_dataset = datasets.ImageFolder(root=data_path / 'train', transform=train_transform)
    test_dataset = datasets.ImageFolder(root=data_path / 'test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    class_dict = train_dataset.class_to_idx

    return train_loader, test_loader, class_names, class_dict

def train_step(device, model, data_loader, loss_fn, optimizer):
    """
    Perform one training step.

    Args:
        device (torch.device): Device to perform computations on (CPU or GPU).
        model (torch.nn.Module): The neural network model.
        data_loader (DataLoader): DataLoader for the training dataset.
        loss_fn: Loss function.
        optimizer: Optimizer for updating the model parameters.

    Returns:
        tuple: A tuple containing average loss and accuracy over the training set.
    """
    model.train()
    total_loss = 0
    correct = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets).sum().item()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)

def test_step(device, model, data_loader, loss_fn):
    """
    Perform one testing step.

    Args:
        device (torch.device): Device to perform computations on (CPU or GPU).
        model (torch.nn.Module): The neural network model.
        data_loader (DataLoader): DataLoader for the testing dataset.
        loss_fn: Loss function.

    Returns:
        tuple: A tuple containing average loss and accuracy over the testing set.
    """
    model.eval()
    total_loss = 0
    correct = 0
    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)

def train(device, model, train_loader, test_loader, loss_fn, optimizer, epochs=10):
    """
    Train the model.

    Args:
        device (torch.device): Device to perform computations on (CPU or GPU).
        model (torch.nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        loss_fn: Loss function.
        optimizer: Optimizer for updating the model parameters.
        epochs (int): Number of epochs for training.
    """
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(device, model, train_loader, loss_fn, optimizer)
        test_loss, test_accuracy = test_step(device, model, test_loader, loss_fn)
        print('\n' + '-' * 30)
        print(f'\nEpoch {epoch+1}\nTrain Loss: {train_loss:.4f}, Train Accuracy: {100*train_accuracy:.2f}'
              f'\nTest Loss:  {test_loss:.4f}, Test Accuracy:  {100*test_accuracy:.2f}')