import torch
from TrajectoryTrain import CNN
from TurtlebotDataLoader import TurtlebotImages
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = TurtlebotImages(csv_file='training.csv',
                               image_dir='training',
                               transform=data_transform)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

model = CNN(3).to(device)

model.load_state_dict(torch.load('model_weights.pth'))

test_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_model(model, test_loader, device)




