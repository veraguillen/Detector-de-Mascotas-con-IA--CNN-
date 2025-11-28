import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EPOCHS = 15
MODEL_FILENAME = "modelo_perros_pro.keras"


def create_data_generators(
    base_dir: str = "data",
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
):
    """Crea generadores de datos con aumento agresivo y normalización 1./255."""

    train_dir = os.path.join(base_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"No se encontró el directorio de entrenamiento: {train_dir}")

    seed = 42
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=seed,
    )

    validation_generator = validation_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=seed,
    )

    return train_generator, validation_generator


def build_transfer_model(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = 3,
) -> Model:
    """Construye un modelo transfer learning basado en MobileNetV2 congelado."""

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train(
    base_dir: str = "data",
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
):
    """Entrena el modelo con MobileNetV2 y guarda el mejor checkpoint."""

    train_generator, validation_generator = create_data_generators(
        base_dir=base_dir, image_size=image_size, batch_size=batch_size
    )
    num_classes = train_generator.num_classes
    print(f"Clases detectadas: {train_generator.class_indices}")

    model = build_transfer_model(
        input_shape=(*image_size, 3),
        num_classes=num_classes,
    )

    class_indices = train_generator.class_indices
    class_counts = train_generator.classes
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(class_indices.values())),
        y=class_counts,
    )
    class_weight_dict = {
        class_name: weight
        for class_name, weight in zip(class_indices.keys(), class_weight_values)
    }
    print("Pesos por clase:")
    for class_name, weight in class_weight_dict.items():
        print(f"- {class_name}: {weight:.4f}")

    checkpoint_path = os.path.join(base_dir, MODEL_FILENAME)
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight={
            class_indices[class_name]: weight
            for class_name, weight in class_weight_dict.items()
        },
    )

    print(f"Mejor modelo guardado en {checkpoint_path}")


if __name__ == "__main__":
    train()
