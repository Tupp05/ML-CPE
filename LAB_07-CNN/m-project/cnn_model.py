
import json
import os

from tensorflow import keras
from tensorflow.keras import layers


# Build a CNN model with configurable layers, filters, and neurons
def build_model(input_shape, num_classes,
                conv_layers=3, filters=(32, 64, 128),
                dense_neurons=128):
    """
    Build CNN model with configurable convolutional layers,
    filters, and dense neurons.
    """

    # Check if the number of convolutional layers is valid
    if not 1 <= conv_layers <= len(filters):
        raise ValueError(
            "conv_layers must be between 1 and len(filters)"
        )

    # Create a Sequential model and add layers one by one
    model = keras.Sequential()

    # Define the input shape of the images
    model.add(keras.Input(shape=input_shape))

    # Normalize pixel values from 0-255 to 0-1
    model.add(layers.Rescaling(1.0 / 255))

    # Apply data augmentation to increase image variety during training
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    # Add convolutional layers based on the selected configuration
    for i in range(conv_layers):
        model.add(layers.Conv2D(
            filters[i],
            kernel_size=3,
            padding="same",
            activation="relu"
        ))

        # Normalize feature maps and reduce their spatial dimensions
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())

    # Convert feature maps into a feature vector
    model.add(layers.GlobalAveragePooling2D())

    # Apply dropout to reduce overfitting
    model.add(layers.Dropout(0.3))

    # Add a Dense layer with a configurable number of neurons
    model.add(layers.Dense(
        dense_neurons,
        activation="relu"
    ))

    model.add(layers.Dropout(0.3))

    # Define the output layer based on the number of classes
    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    # Compile the model with Adam optimizer, loss function, and accuracy metric
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Build, train, and save the CNN model using the selected configuration
def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32,
                conv_layers=3, filters=(32, 64, 128),
                dense_neurons=128, use_early_stopping=False):

    """Build, train and save the model. Returns (model, history)."""

    # Pass the selected CNN configuration to build_model()
    model = build_model(
        X_train.shape[1:],
        num_classes,
        conv_layers=conv_layers,
        filters=filters,
        dense_neurons=dense_neurons
    )

    # Display the model architecture and parameters
    model.summary()

    # Create a list of callbacks for training
    callbacks = []

    # Enable Early Stopping and learning rate reduction if selected
    if use_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        )

        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5
            )
        )

    # Train the CNN model using training and validation datasets
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the trained model and training history if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Generate the model name based on its configuration
        model_name = (
            f"cnn_conv{conv_layers}"
            f"_dense{dense_neurons}"
            f"_epoch{epochs}"
        )

        model_path = os.path.join(
            output_dir,
            f"{model_name}.keras"
        )

        history_path = os.path.join(
            output_dir,
            f"{model_name}_history.json"
        )

        # Save the trained CNN model as a .keras file
        model.save(model_path)

        # Save training accuracy and loss history as a JSON file
        with open(history_path, "w") as f:
            json.dump(
                {
                    k: [float(v) for v in vs]
                    for k, vs in history.history.items()
                },
                f
            )

        print(f"Saved: {model_path}")

    return model, history


# Predict image classes using the trained CNN model
def predict_model(model, X_test):

    # Generate prediction probabilities for the test dataset
    probabilities = model.predict(X_test, verbose=0)

    # Use a threshold of 0.5 for binary classification
    # or select the class with the highest probability for multiclass
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)
