from pathlib import Path
from analysis.plot import draw_subplots
from model.build import create_classification_model, create_mobile_net_v3_large
from process.data_gen import get_classification_train_val_datasets
from train.train import fit_model

Path("artifacts/plots").mkdir(parents=True, exist_ok=True)
Path("artifacts/logs").mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "Basic":        (None,         False,   None),
    "L1":           ("L1",         False,   1e-5),
    "L2":           ("L2",         False,   1e-5),
    "ElasticNetL1": ("ElasticNet", False,   (1e-5, 1e-6)),
    "ElasticNetL2": ("ElasticNet", False,   (1e-6, 1e-5)),
    "ElasticNetEq": ("ElasticNet", False,   (1e-5, 1e-5)),
    "Dropout":      (None,         True,    None),
    "DropoutL2":    ("L2",         True,    1e-5),
}

if __name__ == "__main__":
    image_size = (256, 256)
    image_dir = "dataset"

    base_model, preprocess_input_function = create_mobile_net_v3_large(image_size)

    train_dataset, val_dataset = get_classification_train_val_datasets(
        image_dir,
        preprocess_input_function,
        image_size,
        batch_size=64,
        validation_split=0.15,
    )

    num_classes = len([d for d in Path(image_dir).iterdir() if d.is_dir()])

    for name, (reg_type, is_dropout, lambdas) in CONFIGS.items():
        model = create_classification_model(base_model,
                                            num_classes,
                                            reg_type,
                                            is_dropout=is_dropout,
                                            lambdas=lambdas,
                                            learning_rate=1e-3)

        history = fit_model(model,
                            train_dataset,
                            val_dataset,
                            epochs=15,
                            model_name=name,
                            logging=True)

        draw_subplots(history, filename=f'{name}.png', title=name)