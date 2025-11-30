import os
from typing import Tuple, Dict, Any
import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
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
    learning_rate: float = 1e-3,
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
    
    # Métricas adicionales
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # Precision-Recall curve
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=metrics,
    )

    return model


def plot_training_history(history, output_dir: str = 'results'):
    """Genera y guarda gráficos del historial de entrenamiento."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Métricas para graficar
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'prc']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Graficar métrica de entrenamiento
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Entrenamiento {metric}')
        
        # Graficar métrica de validación si existe
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validación {metric}')
        
        plt.title(f'Evolución de {metric}')
        plt.xlabel('Época')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        
        # Guardar figura
        plt.savefig(os.path.join(output_dir, f'{metric}_history.png'))
        plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: str = 'results'):
    """Genera y guarda la matriz de confusión."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción')
    
    # Guardar figura
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_classification_report(y_true, y_pred, class_names, output_dir: str = 'results'):
    """Guarda un reporte de clasificación en un archivo de texto."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar reporte
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Guardar como JSON
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Guardar como texto formateado
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))

def train(
    base_dir: str = "data",
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = 1e-3,
):
    """Entrena el modelo con MobileNetV2, guarda el mejor checkpoint y genera métricas."""
    
    # Crear directorio de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'training_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Configurar generadores de datos
    train_generator, validation_generator = create_data_generators(
        base_dir=base_dir, 
        image_size=image_size, 
        batch_size=batch_size
    )
    
    num_classes = train_generator.num_classes
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    
    print(f"Clases detectadas: {class_indices}")
    
    # Calcular pesos de clase
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
    
    print("\nPesos por clase:")
    for class_name, weight in class_weight_dict.items():
        print(f"- {class_name}: {weight:.4f}")
    
    # Construir modelo
    model = build_transfer_model(
        input_shape=(*image_size, 3),
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    
    # Resumen del modelo
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    with open(os.path.join(results_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(model_summary))
    
    # Configurar callbacks
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(results_dir, MODEL_FILENAME),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=os.path.join(results_dir, 'logs'),
        histogram_freq=1
    )
    
    callbacks = [
        model_checkpoint,
        early_stopping,
        reduce_lr,
        tensorboard_callback
    ]
    
    # Entrenamiento
    print("\nIniciando entrenamiento...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight={
            class_indices[class_name]: weight
            for class_name, weight in class_weight_dict.items()
        },
    )
    
    # Guardar historial de entrenamiento
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    # Evaluación del modelo
    print("\nEvaluando el modelo en el conjunto de validación...")
    validation_generator.reset()
    
    # Obtener predicciones
    y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes
    
    # Guardar métricas
    plot_training_history(history, results_dir)
    plot_confusion_matrix(y_true, y_pred_classes, class_names, results_dir)
    save_classification_report(y_true, y_pred_classes, class_names, results_dir)
    
    # Guardar configuración
    config = {
        'image_size': image_size,
        'batch_size': batch_size,
        'epochs': len(history.epoch),
        'learning_rate': learning_rate,
        'class_indices': class_indices,
        'class_weights': class_weight_dict,
        'timestamp': timestamp,
    }
    
    with open(os.path.join(results_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nEntrenamiento completado. Resultados guardados en: {results_dir}")
    print(f"Mejor modelo guardado en: {os.path.join(results_dir, MODEL_FILENAME)}")
    
    return model, history


if __name__ == "__main__":
    train()
