from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model import ConvNet

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    model.train()
    losses = []
    correct = 0

    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx + 1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
          train_loss, correct, (batch_idx + 1) * batch_size,
          100. * train_acc))
    return train_loss, train_acc

def test(model, device, test_loader):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = F.cross_entropy(output, target, reduction='mean')
            losses.append(loss.item())

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
          test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy

def plot_metrics(train_losses, test_losses, train_accs, test_accs):
    '''
    Plots training and testing loss and accuracy curves.
    '''
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,5))

    # Loss curves
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, test_losses, marker='o', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Accuracy curves
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, marker='o', label='Train Accuracy')
    plt.plot(epochs, test_accs, marker='o', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_test_samples(model, device, test_loader, num_images=6):
    '''
    Visualizes a few test images.
    '''
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    plt.figure(figsize=(15,3))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"True: {labels[i].item()}\nPred: {preds[i].item()}")
        plt.axis('off')
    plt.show()

def run_main(FLAGS):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected:", device)

    model = ConvNet(FLAGS.mode).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-7)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('./data/', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset1, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)

    best_accuracy = 0.0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    start_time = time.time()

    for epoch in range(1, FLAGS.num_epochs + 1):
        print("\nEpoch:", epoch)
        tr_loss, tr_acc = train(model, device, train_loader, optimizer, criterion, epoch, FLAGS.batch_size)
        te_loss, te_acc = test(model, device, test_loader)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(100. * tr_acc)
        test_accs.append(te_acc)

        if te_acc > best_accuracy:
            best_accuracy = te_acc

    end_time = time.time()
    total_time = end_time - start_time

    print("\nBest Test Accuracy: {:2.2f}%".format(best_accuracy))
    print("Total training time: {:.2f} seconds".format(total_time))

    plot_metrics(train_losses, test_losses, train_accs, test_accs)

    visualize_test_samples(model, device, test_loader)

    print("Training and evaluation finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode', type=int, default=3, help='Select mode between 1-3.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to put logging.')

    FLAGS, unparsed = parser.parse_known_args()

    print("Mode:", FLAGS.mode)
    print("LR:", FLAGS.learning_rate)
    print("Batch size:", FLAGS.batch_size)

    run_main(FLAGS)