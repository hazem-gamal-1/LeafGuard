import os
from torch.utils.data import Dataset
import cv2


class PlantVillage(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.images = []
        self.labels = []

    def _prepare_dataset(self):
        classes = sorted([folder for folder in os.listdir(self.root_dir)])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {
            idx: cls_name for cls_name, idx in self.class_to_idx.items()
        }
        for class_name in classes:
            folder_path = os.path.join(self.root_dir, class_name)
            for image in os.listdir(folder_path):
                self.images.append(image)
                self.labels.append(self.class_to_idx[image])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image)
            image = augmented["image"]
        return image, label
