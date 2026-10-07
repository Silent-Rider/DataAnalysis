import random
from pathlib import Path
from typing import Callable
import tensorflow as tf
from tensorflow.data import Dataset


def get_classification_train_val_datasets(image_dir: str,
                                          preprocess_input_function: Callable,
                                          image_size: tuple,
                                          batch_size: int,
                                          validation_split: float):

    image_paths, labels = get_image_paths_and_labels(image_dir)
    num_classes = len(set(labels))

    return get_train_val_datasets(
        image_paths=image_paths,
        annotations=labels,
        annotation_function=lambda label: tf.one_hot(label, depth=num_classes),
        preprocess_input_function=preprocess_input_function,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split
    )


def load_image(image_path: str, image_size: tuple, preprocess_input_function):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = preprocess_input_function(image)
    return image


def get_train_val_datasets(image_paths: list[str],
                           annotations: list,
                           annotation_function: Callable,
                           preprocess_input_function: Callable,
                           image_size: tuple,
                           batch_size: int,
                           validation_split: float):

    combined = list(zip(image_paths, annotations))
    random.seed(42)
    random.shuffle(combined)
    image_paths, annotations = zip(*combined)

    image_paths = list(image_paths)
    annotations = list(annotations)

    split_idx = int(len(image_paths) * (1 - validation_split))
    train_img, val_img = image_paths[:split_idx], image_paths[split_idx:]
    train_ann, val_ann = annotations[:split_idx], annotations[split_idx:]

    train_dataset = Dataset.from_tensor_slices((train_img, train_ann))
    val_dataset = Dataset.from_tensor_slices((val_img, val_ann))

    train_dataset = (train_dataset.map(
        lambda image_path, annotation: (
            load_image(image_path, image_size, preprocess_input_function), annotation_function(annotation)
        ), num_parallel_calls=tf.data.AUTOTUNE)
                     .cache()
                     .shuffle(buffer_size=1000, seed=42)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = (val_dataset.map(
        lambda image_path, annotation: (
            load_image(image_path, image_size, preprocess_input_function), annotation_function(annotation)
        ), num_parallel_calls=tf.data.AUTOTUNE)
                   .cache()
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

    return train_dataset, val_dataset


def get_image_paths_and_labels(image_dir: str):
    image_dir = Path(image_dir)

    class_names = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    image_paths, labels = [], []
    for cls in class_names:
        for p in (image_dir / cls).iterdir():
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                image_paths.append(str(p))
                labels.append(class_to_idx[cls])

    return image_paths, labels