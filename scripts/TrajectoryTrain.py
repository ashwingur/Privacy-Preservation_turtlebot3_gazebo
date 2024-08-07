'''
This script takes in a directory of captured images and a csv that contains the corresponding state of the
robot at that time (we are interested in image name and angular velocity). It trains a neural network on this
data by mapping the input images to a steering command. The steering command is the angular velocity translated to
a direction (left, right or straight). The model is saved so that it can be loaded by another script.
'''

import os
import sys
from matplotlib import ticker
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from TurtlebotDataLoader import TurtlebotDataLoader
import time
import matplotlib.pyplot as plt

# Define CNN model


class TurtlebotCNN(nn.Module):
    def __init__(self):
        super(TurtlebotCNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 93 * 51, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 3)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # 3 input channels for R,G,B
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)  # Output layer for actions: left, forward, right


    def forward(self, x):
        """Forward pass of the neural network"""
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the feature map
        x = x.view(-1, 256 * 16 * 16)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
def test_model(model, test_loader, device, num_classes=3):
    start_time = time.time()
    model.eval()
    correct = [0] * num_classes
    total_per_class = [0] * num_classes  # Renamed to avoid conflict
    print("Running test")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(num_classes):
                # print(f'{predicted}, {labels}')
                correct[i] += ((predicted == i) & (labels == i)).sum().item()
                total_per_class[i] += (labels == i).sum().item()  # Fixed the variable name

    overall_accuracy = 100 * sum(correct) / sum(total_per_class)
    total_time = time.time() - start_time
    print(f'Overall Accuracy of the model on the test images: {overall_accuracy:.2f}%')
    for i in range(num_classes):
        if total_per_class[i] == 0:
            print(f"Accuracy for class '{TurtlebotDataLoader.label_to_direction_string(i)}': ({correct[i]}/{total_per_class[i]})")
        else:
            print(f"Accuracy for class '{TurtlebotDataLoader.label_to_direction_string(i)}': {(correct[i]/total_per_class[i]*100):.2f}% ({correct[i]}/{total_per_class[i]})")
    print(f"Finished testing, total time taken: {total_time/60.0:.2f} minutes")

    return overall_accuracy

def plot_performance(train_metric, val_metric, metric_name):
    plt.close()
    epochs = range(1, len(train_metric) + 1)
    plt.plot(epochs, train_metric, 'bo-', label=f'Training Accuracy')
    plt.plot(epochs, val_metric, 'ro-', label=f'Validation Accuracy')
    # plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.xticks(range(1, len(train_metric) + 1))
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f'results/Training_and_Validation_{metric_name}.png')
    plt.savefig(f'results/Training_and_Validation_{metric_name}.eps',format='eps', bbox_inches='tight')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage python3 ImageRead.py <scenario_name> <model.pth>')
        sys.exit(1)
    
    # CSV and image paths
    SCENARIO_NAME = sys.argv[1]
    IMAGE_DIR = os.path.join('images', SCENARIO_NAME)
    CSV_FILE = os.path.join('csv', SCENARIO_NAME) + '.csv'
    MODEL_FILENAME = os.path.join('models', sys.argv[2])

    if not os.path.exists('results'):
        os.makedirs('results')
        print("'results' directory does not exist, creating it.")


    # Proportion of dataset for training (0-1)
    ENABLE_EPOCH_TESTING = True
    TRAINING_PORTION = 0.95

    EPOCHS = 10

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = TurtlebotDataLoader(csv_file=CSV_FILE,
                                    image_dir=IMAGE_DIR,
                                    transform=data_transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = TurtlebotCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training loop...")

    # Training loop
    total_time = 0.0
    training_losses = []
    training_accuracies = []
    validation_accuracies = []
    for epoch in range(EPOCHS):
        start_time = time.time()
        # Set to 1 to not skip anything, 2 skip every second image and so on
        SKIP_AMOUNT = 1
        count = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        training_losses.append(loss.item())
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}. Time taken: {epoch_time/60.0:.2f} minutes')

        # Evaluate the model accuracy on testing and training data at the current epoch
        if ENABLE_EPOCH_TESTING:
            print("Evaluating training set accuracy:")
            training_accuracies.append(test_model(model, train_loader, device))
            print("Evaluating validation set accuracy:")
            validation_accuracies.append(test_model(model, val_loader, device))
        total_time += time.time() - start_time
        print(f'Starting next epoch\n')

    print(f"Finished training, total time taken: {total_time/60.0:.2f} minutes")

    # Plot the loss
    plt.plot(list(range(1, EPOCHS + 1)), training_losses, marker='o')
    # plt.title('Loss Over Time')
    plt.xticks(range(1, EPOCHS + 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    # Set integer ticks on the x-axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Save the plot as an image
    plt.savefig(f'results/{os.path.basename(CSV_FILE)[:-4]}_loss_plot.png')
    plt.savefig(f'results/{os.path.basename(CSV_FILE)[:-4]}_loss_plot.eps',format='eps', bbox_inches='tight')

    if ENABLE_EPOCH_TESTING:
        plot_performance(training_accuracies, validation_accuracies, f"{os.path.basename(CSV_FILE)[:-4]}_Accuracy")

    # Save the model weights
    if not os.path.exists('models'):
        os.makedirs('models')
        print("'models' directory does not exist, creating it.")
    print('Saving model...')
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f'Saved model to models/{MODEL_FILENAME}!')
