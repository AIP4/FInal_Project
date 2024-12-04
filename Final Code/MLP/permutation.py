from sklearn.inspection import permutation_importance

def permutation(model, X_train_val, y_train_val, X_test, y_test, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error'):
    """
    Evaluate the model on the test set and calculate permutation importance for each feature.

    Parameters:
    - model: Trained model to evaluate
    - X_train_val: Training and validation features
    - y_train_val: Training and validation target
    - X_test: Test features for permutation importance
    - y_test: Test target for permutation importance
    - n_repeats: Number of repetitions for permutation (default 10)
    - random_state: Random seed for reproducibility (default 42)
    - scoring: Scoring method for permutation importance (default is 'neg_mean_absolute_error')

    Returns:
    - mean_mae: Mean absolute error of the model on the test set
    - low_importance_features: List of features with importance scores <= 0
    - features_with_scores: List of tuples (feature, importance score)
    """
    # Step 1: Evaluate the model
    mean_mae = model.evaluate(X_train_val, y_train_val, verbose=0)[1]  # Return the mean absolute error
    print(f"Model Evaluation - Mean Absolute Error: {mean_mae}")

    # Step 2: Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )

    # Step 3: Extract importance scores
    importance_scores = perm_importance.importances_mean

    # Initialize lists to store features with low importance and their scores
    low_importance_features = []
    features_with_scores = []

    # Step 4: Print and store feature importance scores
    for i, score in enumerate(importance_scores):
        print(f"Feature: {X_train_val.columns[i]}, Importance: {score}")
        features_with_scores.append((X_train_val.columns[i], score))
        
        # Add to low_importance_features if score is less than or equal to 0
        if score <= 0:
            low_importance_features.append(X_train_val.columns[i])
    features_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Select the top 10 features with the highest importance
    top_10_important_features = features_with_scores[:10]

    # Print the top 10 important features (sorted in descending order)
    print("\nTop 10 Important Features (Sorted Descending by Importance Score):")
    for feature, score in top_10_important_features:
        print(f"Feature: {feature}, Importance: {score}")

    return mean_mae, low_importance_features, features_with_scores
