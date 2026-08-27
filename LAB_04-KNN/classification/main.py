import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   

from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# import from file data_loader_class
from data_loader_class import load_and_preprocess_images, CLASS_NAMES
import evaluate
from knn_tf import TFKNNClassifier

OUT_DIR = Path(__file__).resolve().parent / "outputs"

def title(text):
    print("\n" + "--" * 30)
    print(text)

# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1 : load data")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_images()
    class_names = CLASS_NAMES

    # Calculate the total number of pictures
    n_rows = len(X_train) + len(X_val) + len(X_test)
    print(f"data of all : {n_rows} images")
    print(f"classes of predictions : {class_names}")
    print(f"split data : train {len(y_train)} / validation {len(y_val)} / test {len(y_test)}")

    title("STEP 2 : search k for finding the best value")

    k_values = [1, 2, 3, 5, 10, 11, 15, 21, 30]
    scores = []

    for k in k_values:
        model = TFKNNClassifier(k=k)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        scores.append(acc)
        print(f"   k = {k:>2}  ->  validation accuracy = {acc:.4f}")

    best_k = k_values[int(np.argmax(scores))]
    print(f"\n>>> k value is best at {best_k}")

    evaluate.plot_k_curve(k_values, scores, OUT_DIR / "01_k_curve.png")

    title(f"STEP 3 : Training model with k = {best_k} and evaluate on test set")

    model = TFKNNClassifier(k=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float(np.mean(y_pred == y_test))
    print(f"Accuracy on test : {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print("\n the result show in class-wise report:\n")
    evaluate.print_report(y_test, y_pred, class_names)

    cm = evaluate.plot_confusion_matrix(y_test, y_pred, class_names,
                                        OUT_DIR / "02_confusion_matrix.png")
    print("Confusion Matrix (rows = true labels, columns = predicted labels):")
    print(cm)

    title("STEP 4 : check code - compare with scikit-learn ALGORITHM")
  
    sk_model = KNeighborsClassifier(n_neighbors=best_k)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)

    print(f"KNN: result of our (TensorFlow) : {accuracy:.4f}")
    print(f"KNN: result of sklearn          : {np.mean(sk_pred == y_test):.4f}")
    print(f"KNN: matching predictions       : {np.mean(sk_pred == y_pred) * 100:.1f} %")

    title("STEP 5 : Is our model better than guessing? ")

    majority = np.bincount(y_train).argmax()
    baseline = float(np.mean(y_test == majority))

    print(f"Baseline (predict '{class_names[majority]}' every time) : {baseline:.4f}")
    print(f"KNN: our model                         : {accuracy:.4f}")

    if accuracy <= baseline:
        print("\n[summary] KNN DOES NOT win against guessing!")
        print("because: raw pixels are very hard to classify with simple KNN")
        print("         feature not correlated with dog breeds")
        print("         result: That collect, is not code error")
        print("         detect 'not signal in data' is important skill for Data Scientist")
    else:
        print("\n[summary] KNN win than guessing! The model learned something.")

# ---------------------------------------------------------------------------
    title("save predictions to CSV file")

    evaluate.save_predictions(y_test, y_pred, class_names,
                              OUT_DIR / "predictions.csv")

    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()