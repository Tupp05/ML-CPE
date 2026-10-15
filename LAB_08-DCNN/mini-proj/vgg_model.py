
import json
import os

from tensorflow import keras
from tensorflow.keras import layers


# VGG model configurations
VGG16 = [
    (64, 2),
    (128, 2),
    (256, 3),
    (512, 3),
    (512, 3)
]

VGG_SMALL = [
    (32, 2),
    (64, 2),
    (128, 3),
    (256, 3)
]


def vgg_block(model, filters, n_conv):

    # Add convolutional layers
    for _ in range(n_conv):

        model.add(layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu"
        ))

    # BatchNormalization usually helps neural networks train more smoothly, but for this dataset, turning it off gave better results.
        #model.add(layers.BatchNormalization()) 

    # Reduce feature map size
    model.add(layers.MaxPooling2D(pool_size=2))


def build_model(input_shape, num_classes, blocks=VGG16):

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    # Apply data augmentation during training
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    # Build convolutional blocks
    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv)

    # Flatten extracted features
    model.add(layers.GlobalAveragePooling2D())

    # Add fully connected layers
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))

    # Configure output layer
    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss=(
            "binary_crossentropy"
            if num_classes == 2
            else "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"]
    )

    return model


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    output_dir=None,
    epochs=50,
    batch_size=32,
    blocks=VGG16
):

    # Build DCNN model
    model = build_model(
        X_train.shape[1:],
        num_classes,
        blocks
    )

    model.summary()

    # Configure training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    ]

    # Train model
    print("\nTraining...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save trained model and history
    if output_dir:

        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(
            output_dir,
            "vgg_model.keras"
        )

        model.save(model_path)

        history_path = os.path.join(
            output_dir,
            "history.json"
        )

        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                },
                f,
                indent=4
            )

        print(f"Saved: {model_path}")

    return model, history


def predict_model(model, X_test):

    # Predict class probabilities
    probabilities = model.predict(
        X_test,
        verbose=0
    )

    # Binary classification
    if probabilities.shape[-1] == 1:
        return (
            probabilities.ravel() > 0.5
        ).astype(int)

    # Multi-class classification
    return probabilities.argmax(axis=1)
