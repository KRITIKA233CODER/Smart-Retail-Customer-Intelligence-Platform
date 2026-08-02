import json
import os

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_DIR = "data/products/train"
VALIDATION_DIR = "data/products/validation"

MODEL_DIR = "app/models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 1

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n=== PRODUCT CLASSIFIER TRAINING ===\n")

print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


class_names = train_dataset.class_names

NUM_CLASSES = len(class_names)

print("\nClasses:")
for index, name in enumerate(class_names):
    print(index, "->", name)


# --------------------------------------------------
# Save class mapping
# --------------------------------------------------

class_mapping = {
    index: name
    for index, name in enumerate(class_names)
}

with open(
    os.path.join(MODEL_DIR, "product_classes.json"),
    "w"
) as file:

    json.dump(
        class_mapping,
        file,
        indent=4
    )


print("\nClass mapping saved.")


# --------------------------------------------------
# Dataset performance
# --------------------------------------------------

# train_dataset = train_dataset.prefetch(1)
# validation_dataset = validation_dataset.prefetch(1)


# --------------------------------------------------
# Data augmentation
# --------------------------------------------------

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# --------------------------------------------------
# MobileNetV2 base model
# --------------------------------------------------

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained network
base_model.trainable = False


# --------------------------------------------------
# Build classifier
# --------------------------------------------------

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="smart_retail_product_classifier"
)


# --------------------------------------------------
# Compile
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel architecture:\n")

model.summary()


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            MODEL_DIR,
            "product_classifier.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

print("\nEvaluating model...\n")

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset
)

print(
    f"\nValidation Loss: "
    f"{validation_loss:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


# --------------------------------------------------
# Save final model
# --------------------------------------------------

model.save(
    os.path.join(
        MODEL_DIR,
        "product_classifier_final.keras"
    )
)


# --------------------------------------------------
# Save training history
# --------------------------------------------------

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    os.path.join(
        MODEL_DIR,
        "product_training_history.json"
    ),
    "w"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )


print("\nTraining complete.")

print(
    "Best model: "
    "app/models/product_classifier.keras"
)

print(
    "Final model: "
    "app/models/product_classifier_final.keras"
)