import os
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from transform import train_transform, val_transform


class PlantVillageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.images = []
        self.labels = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = sorted([folder for folder in os.listdir(self.root_dir)])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {
            idx: cls_name for cls_name, idx in self.class_to_idx.items()
        }
        for class_name in classes:
            folder_path = os.path.join(self.root_dir, class_name)
            for image in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, image))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label


def prepare_datasets(config):
    full_dataset = PlantVillageDataset(config["root_dir"], transform=None)

    val_size = int(len(full_dataset) * config["dataset"]["test_size"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["train"]["batch_size"], shuffle=False
    )

    return train_loader, val_loader
