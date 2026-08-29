import json
import os

from tensorflow import keras
from tensorflow.keras import layers

# Add hidden_layers as a List to allow defining the number of layers and nodes externally
def build_model(input_shape, num_classes, hidden_layers=[256, 128, 64]):
    """Fully-connected neural network (MLP)."""

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    # Normalize 0-255 to 0-1 inside the model
    model.add(layers.Rescaling(1.0 / 255))
    # An MLP takes a flat vector, so the 2D image is unrolled here
    model.add(layers.Flatten())

    # Create Hidden Layers automatically based on the hidden_layers list
    for neurons in hidden_layers:
        model.add(layers.Dense(neurons, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4)) # You can change the Dropout value from here

    # Output Layer: 1 sigmoid output for 2 classes, softmax otherwise
    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Accept hidden_layers and model_name to prevent overwriting saved files
def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32, 
                hidden_layers=[256, 128, 64], model_name="nn_model"):
    """Build, train and save the model. Returns (model, history)."""

    model = build_model(X_train.shape[1:], num_classes, hidden_layers)
    model.summary()

    callbacks = [
        # Stop when validation loss stops improving, keep the best weights
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print(f"\nTraining {model_name}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the saved file name according to model_name
        model_path = os.path.join(output_dir, f"{model_name}.keras")
        history_path = os.path.join(output_dir, f"{model_name}_history.json")

        model.save(model_path)
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {model_path}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    # Binary head outputs one probability, multiclass outputs one per class
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)