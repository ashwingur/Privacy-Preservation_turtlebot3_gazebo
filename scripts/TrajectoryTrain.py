'''
This script takes in a directory of captured images and a csv that contains the corresponding state of the
robot at that time (we are interested in image name and angular velocity). It trains a neural network on this
data by mapping the input images to a steering command. The steering command is the angular velocity translated to
a direction (left, right or straight). The model is saved so that it can be loaded by another script.
'''

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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 93 * 51, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        """Forward pass of the neural network"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
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
    plt.plot(epochs, train_metric, 'bo-', label=f'Training {metric_name}')
    plt.plot(epochs, val_metric, 'ro-', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f'Training_and_Validation_{metric_name}.png')


if __name__ == "__main__":
    # Proportion of dataset for training (0-1)
    ENABLE_EPOCH_TESTING = True
    TRAINING_PORTION = 0.9

    EPOCHS = 10

    # CSV and image paths
    CSV_FILE = "training.csv"
    IMAGE_DIR = "training"

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    # Resize 1920x1080 down by 5x
    data_transform = transforms.Compose([
        transforms.Resize((384, 216)),
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
        total_time += epoch_time
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}. Time taken: {epoch_time/60.0:.2f} minutes')

        # Evaluate the model accuracy on testing and training data at the current epoch
        if ENABLE_EPOCH_TESTING:
            training_accuracies.append(test_model(model, train_loader, device))
            validation_accuracies.append(test_model(model, val_loader, device))

    print(f"Finished training, total time taken: {total_time/60.0:.2f} minutes")

    # Plot the loss
    plt.plot(list(range(1, EPOCHS + 1)), training_losses, marker='o')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    # Set integer ticks on the x-axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Save the plot as an image
    plt.savefig('loss_plot.png')

    if ENABLE_EPOCH_TESTING:
        plot_performance(training_accuracies, validation_accuracies, "Accuracy")

    # Save the model weights
    print('Saving model...')
    torch.save(model.state_dict(), 'model_weights.pth')
    print('Saved!')
