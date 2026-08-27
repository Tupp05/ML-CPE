from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Add the parameter kernel="rbf" so that main.py can pass the desired kernel name to this function.
def train_svm(X_train, y_train, kernel="rbf", pca_components=150):
    # Scaler + PCA in one pipeline so test data always gets the same
    # transform. PCA also makes RBF SVM tractable on 10,000 pixel features.
    scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                    whiten=True, random_state=42)),
    ])

    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Create SVM model
    # Change the fixed `kernel="rbf"` to `kernel=kernel` to receive values ​​from the variable.
    model = SVC(
        kernel=kernel, C=10, gamma="scale", cache_size=1000
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_svm(model, scaler, X_test):

    # Apply the same scaling used for training data
    X_test_scaled = scaler.transform(X_test)
    # Predict
    predictions = model.predict(X_test_scaled)

    return predictions