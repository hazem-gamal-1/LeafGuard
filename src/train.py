import torch
from model import PlantDiseaseClassifier
from dataset import prepare_datasets
import torch.optim as optim
import torch.nn as nn
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(config):
    train_loader, val_loader, _ = prepare_datasets(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseClassifier(
        config["dataset"]["num_classes"], config["train"]["freeze_backbone"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    mlflow.log_params(
        {
            "learning_rate": config["train"]["learning_rate"],
            "batch_size": config["train"]["batch_size"],
            "optimizer": "Adam",
            "epochs": config["train"]["epochs"],
            "freeze_backbone": config["train"]["freeze_backbone"],
            "test_size": config["dataset"]["test_size"],
            "val_size": config["dataset"]["val_size"],
        }
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(config["train"]["epochs"]):
        model.train()
        running_loss, correct, total,best_val_acc= 0.0, 0, 0,0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        if epoch == config["train"]["unfreeze_backbone_epoch"]:
          for param in model.model.parameters():
              param.requires_grad = True

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            mlflow.pytorch.log_model(model, name="best_model")


        mlflow.log_metrics(
            {
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "best_val_acc":best_val_acc
            },
            step=epoch,
        )

        print(
            f"Epoch [{epoch+1}/{config['train']['epochs']}]: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}"
            f"Best_Val_acc {best_val_acc}"
        )


    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("loss_curves.png")
    mlflow.log_artifact("loss_curves.png", "plots")
    plt.close()

    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.savefig("accuracy_curves.png")
    mlflow.log_artifact("accuracy_curves.png", "plots")
    plt.close()

    return model
