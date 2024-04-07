

'''
This script takes in a directory of captured images and a csv that contains the corresponding state of the
robot at that time (we are interested in image name and angular velocity). It trains a neural network on this
data by mapping the input images to a steering command. The steering command is the angular velocity translated to
a direction (left, right or straight). The model is saved so that it can be loaded by another script.
'''

from matplotlib import ticker
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from TurtlebotDataLoader import TurtlebotImages
import time
import matplotlib.pyplot as plt

# Define CNN model


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 93 * 51, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Forward pass of the neural network"""
        x = self.pool(F.relu(self.conv1(x)))
        # print(f'SHAPE: {x.shape}')
        x = self.pool(F.relu(self.conv2(x)))
        # print(f'SHAPE: {x.shape}')
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(f'SHAPE: {x.shape}')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    # Resize 1920x1080 down by 5x
    data_transform = transforms.Compose([
        transforms.Resize((384, 216)),
        transforms.ToTensor(),
    ])

    # Load dataset
    train_dataset = TurtlebotImages(csv_file='training.csv',
                                    image_dir='training',
                                    transform=data_transform)

    # Define DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize the model
    model = CNN(num_classes=3).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Training loop
    total_time = 0.0
    training_losses = []
    num_epochs = 10  # Change this as needed
    for epoch in range(num_epochs):
        start_time = time.time()
        SKIP_AMOUNT = 2
        count = 0
        model.train()
        for images, labels in train_loader:
            # Only do a portion of the dataset
            count += 1
            if count % SKIP_AMOUNT != 0:
                continue

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}. Time taken: {epoch_time/60.0:.2f} minutes')
    print(f"Finished training, total time taken: {total_time/60.0:.2f} minutes")

    # Plot the loss
    plt.plot(list(range(1, num_epochs + 1)), training_losses, marker='o')
    plt.title('Loss Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    # Set integer ticks on the x-axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Save the plot as an image
    plt.savefig('loss_plot.png')

    # Save the model weights
    print('Saving model...')
    torch.save(model.state_dict(), 'model_weights.pth')
    print('Saved!')
