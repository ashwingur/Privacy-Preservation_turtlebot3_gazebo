import torch
from TrajectoryTrain import CNN
from TurtlebotDataLoader import TurtlebotImages
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
data_transform = transforms.Compose([
    transforms.Resize((384, 216)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = TurtlebotImages(csv_file='training.csv',
                               image_dir='training',
                               transform=data_transform)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

def test_model(model, test_loader, device):
    start_time = time.time()
    model.eval()
    correct = 0
    total = 0
    print("Running test")
    with torch.no_grad():
        for images, labels in test_loader:
            print(images[0].size())
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    total_time = time.time() - start_time
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    print(f"Finished testing, total time taken: {total_time/60.0:.2f} minutes")

model = CNN().to(device)

model.load_state_dict(torch.load('model_weights.pth'))

test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_model(model, test_loader, device)
