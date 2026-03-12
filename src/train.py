import torch
from model import PlantDiseaseClassifier
from dataset import PlantVillageDataset
from transform import train_transform
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def train(config):
    dataset = PlantVillageDataset(config["root_dir"], train_transform)
    train_loader = DataLoader(
        dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseClassifier(
        config["dataset"]["num_classes"], config["train"]["freeze_backbone"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config["train"]["epochs"]):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Loss: {avg_loss:.4f}")
