from utils import load_config
from dataset import PlantVillage

config = load_config("configs/config.yaml")
print(config)
dataset = PlantVillage(config["root_dir"])
dataset._prepare_dataset()
