import mlflow
from train import train
from evaluate import evaluate
from dataset import prepare_datasets
from utils import load_config


def main(eval=False):
    config = load_config("configs/config.yaml")

    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        print("Starting training...")
        model = train(config)

        print("Evaluating on test set...")
        if eval:
            _, _, test_loader = prepare_datasets(config)
            class_names = [
                test_loader.dataset.dataset.idx_to_class[i]
                for i in range(len(test_loader.dataset.dataset.idx_to_class))
            ]
            evaluate(model, test_loader, class_names)

        print(f"Run finished. Run ID: {run_id}")


if __name__ == "__main__":
    main(True)
