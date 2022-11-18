import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


def get_data_loader(training=True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.FashionMNIST(
        './data', train=True, download=True, transform=custom_transform)
    test_set = datasets.FashionMNIST(
        './data', train=False, transform=custom_transform)

    if training == True:
        train_loader = DataLoader(train_set, batch_size=64, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,
                                  pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, prefetch_factor=2, persistent_workers=False)
        return train_loader

    if training == False:
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, prefetch_factor=2, persistent_workers=False)
        return test_loader


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(nn.Flatten(), nn.Linear(
        784, 196), nn.ReLU(), nn.Linear(196, 49), nn.ReLU(), nn.Linear(49, 10))
    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        total = len(train_loader.dataset)
        running_loss = 0.0
        correct = 0
        model.train()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()*50
            prediction = outputs.argmax(dim=1, keepdim=True)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()

        print(
            f'Train Epoch: {epoch}   Accuracy: {correct}/{total}({100. * correct / total:.2f}%)   Loss: {running_loss / total:.3f}')


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()

    running_loss = 0
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data)
            loss = criterion(output, labels)
            running_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()

    average_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    if show_loss == True:
        print(f'Average Loss: {average_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
    else:
        print(f'Accuracy: {accuracy:.2f}%')


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """
    output = model(test_images[index])
    prob = F.softmax(output, dim=1)

    class_names = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    prob = prob.detach_().numpy()
    temp_index = prob.argsort()[-3:][::-1]
    flatten_index = temp_index.flatten()
    flatten_index = flatten_index[::-1]

    prob = prob.flatten() * 100
    for i in range(0, 3):
        print(f'{class_names[flatten_index[i]]}: {(prob[flatten_index[i]]):.2f}%')


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
