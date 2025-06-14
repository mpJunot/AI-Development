from mlp import MLP
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == "__main__":
    task = "classification"
    optimizer = "adam"
    weight_init = "he"
    weight_scale = 1.0
    l2_lambda = 0.0001
    l1_lambda = 0.0001
    elastic_alpha = 0.5
    early_stopping = True
    patience = 5
    min_delta = 1e-4
    epochs = 50
    learning_rate = 0.01
    batch_size = 32

    debug_small_set = False

    if task == "classification":
        layer_sizes = [784, 256, 128, 10]
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)
    else:
        data = fetch_california_housing()
        X = data.data
        y = data.target
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
        layer_sizes = [X.shape[1], 64, 32, 1]

    mlp = MLP(
        layer_sizes,
        activation_function="relu",
        learning_rate=learning_rate,
        optimizer=optimizer,
        weight_init=weight_init,
        weight_scale=weight_scale,
        task=task
    )

    mlp.train(
        X_train, y_train, epochs,
        l2_lambda=l2_lambda,
        l1_lambda=l1_lambda,
        elastic_alpha=elastic_alpha,
        validation_data=X_test,
        validation_labels=y_test,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        batch_size=batch_size,
        debug_small_set=debug_small_set,
        learning_rate=learning_rate
    )

    val_loss = mlp.evaluate(X_test, y_test)
    print(f"Validation Loss: {val_loss}")

    if task == "classification":
        mlp.plot_confusion_matrix(X_test, y_test)
        mlp.plot_predictions(X_test, y_test)
    mlp.plot_training_history()
    mlp.save_model("mlp_model.npz")
