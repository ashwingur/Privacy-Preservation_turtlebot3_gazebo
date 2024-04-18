import torch
from TrajectoryTrain import TurtlebotCNN
from TurtlebotDataLoader import TurtlebotDataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time


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

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    data_transform = transforms.Compose([
        transforms.Resize((384, 216)),
        transforms.ToTensor(),
    ])

    # Load dataset
    train_dataset = TurtlebotDataLoader(csv_file='training.csv',
                                image_dir='training',
                                transform=data_transform)
    model = TurtlebotCNN().to(device)
    model.load_state_dict(torch.load('model_weights.pth'))
    test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_model(model, test_loader, device)
