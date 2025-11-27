import os

from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_model(input_shape=(150, 150, 3), num_classes: int = 3) -> Model:
    """Define una CNN sencilla con tres bloques convolucionales para clasificación multiclase."""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_generators(base_dir: str, image_size=(150, 150), batch_size=32):
    """Crea generadores de entrenamiento y validación con aumento y normalización."""
    train_dir = os.path.join(base_dir, "train")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, validation_generator


def train(base_dir: str = "data", epochs: int = 50, batch_size: int = 32):
    """Entrena la CNN y guarda el mejor modelo en formato .keras."""
    train_generator, validation_generator = get_generators(
        base_dir, batch_size=batch_size
    )
    print(train_generator.class_indices)
    model = build_model(
        input_shape=(*train_generator.image_shape[:2], 3),
        num_classes=len(train_generator.class_indices),
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(base_dir, "modelo_multiclase.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
    )

    save_path = os.path.join(base_dir, "modelo_multiclase.keras")
    print(f"Modelo guardado en {save_path}")


if __name__ == "__main__":
    train()
