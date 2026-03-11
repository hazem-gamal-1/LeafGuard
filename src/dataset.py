import os
from torch.utils.data import Dataset


class PlantVillage(Dataset):
    def __int__(self, data_dir, transform=None):
        super.__init__()
        self.data_dir = data_dir
        self.transform = transform

        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            self.dataset = [
                (os.path.join(folder_path, image), image)
                for image in os.listdir(folder_path)
            ]


    def _prepare_dataset(self):
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset
