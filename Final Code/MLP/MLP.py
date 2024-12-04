import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

def train_and_evaluate(X, y, test_size=0.2, n_splits=5, epochs=100, batch_size=64):
    """
    Step 1: Split the data into training+validation and test sets.
    Step 2: Apply K-fold cross-validation to the training+validation set.
    Step 3: Train an MLP model on each fold, and compute MAE for each fold.
    Step 4: Evaluate the final model on the test set.

    Parameters:
    - X: Features (input data)
    - y: Target (output labels)
    - test_size: Proportion of the dataset to include in the test split
    - n_splits: Number of splits for K-fold cross-validation
    - epochs: Number of epochs for model training
    - batch_size: Batch size for model training

    Returns:
    - mean_mae: The mean MAE from cross-validation
    - test_mae: The MAE on the test set
    - model: The trained model after the final fold
    - X_train_val: The training and validation data (features)
    - y_train_val: The training and validation data (target)
    - X_test: The test data (features)
    - y_test: The test data (target)
    """
    # Step 1: Split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Step 2: Apply K-fold cross-validation on the training+validation set
    kf = KFold(n_splits=n_splits, shuffle=False)

    mae_scores = []  # List to store mean absolute error for each fold

    # 데이터 정규화 (scaling)
    scaler = StandardScaler()

    for train_index, val_index in kf.split(X_train_val):
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        # 데이터 정규화
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # MLP 모델 설계
        mlp_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(1)
        ])

        mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # 학습
        mlp_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0  # Change to 0 to suppress verbose output during training
        )

        # 평가
        val_loss, val_mae = mlp_model.evaluate(X_val_scaled, y_val, verbose=0)
        mae_scores.append(val_mae)  # Store the mean absolute error for each fold

    # 평균 MAE 출력
    mean_mae = np.mean(mae_scores)
    print(f"Cross-validated Mean Absolute Error: {mean_mae}")

    # Step 3: Evaluate on the separate test set
    X_test_scaled = scaler.transform(X_test)
    test_loss, test_mae = mlp_model.evaluate(X_test_scaled, y_test)

    print(f"Test Set Mean Absolute Error: {test_mae}")

    return mean_mae, test_mae, mlp_model, X_train_val, y_train_val, X_test, y_test  # Return all relevant variables
