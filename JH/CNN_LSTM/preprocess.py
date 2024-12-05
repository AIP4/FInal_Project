from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

def minmax_scaling(data):
    scaler = MinMaxScaler()
    
    # Select numeric columns excluding 'diff'
    numeric_columns = data.select_dtypes(include=['number']).columns
    numeric_columns = numeric_columns.drop("diff")
    numeric_columns = numeric_columns.drop("PM10")
    
    # Fit MinMaxScaler and transform the numeric columns
    scaled_numeric = scaler.fit_transform(data[numeric_columns])
    
    # Replace the numeric columns with scaled values
    scaled_data = data.copy()
    scaled_data[numeric_columns] = scaled_numeric
    
    return scaled_data, scaler

def standard_scaling(data):
    scaler = StandardScaler()
    
    # Select numeric columns excluding 'diff'
    numeric_columns = data.select_dtypes(include=['number']).columns
    numeric_columns = numeric_columns.drop("diff")
    numeric_columns = numeric_columns.drop("PM10")
    
    # Fit MinMaxScaler and transform the numeric columns
    scaled_numeric = scaler.fit_transform(data[numeric_columns])
    
    # Replace the numeric columns with scaled values
    scaled_data = data.copy()
    scaled_data[numeric_columns] = scaled_numeric
    
    return scaled_data, scaler

def recover_minmax_scaled_with_exclusion(predicted_column, scaler, column_name, excluded_columns):
    # Ensure the column was not excluded
    if column_name in excluded_columns:
        raise ValueError(f"Column '{column_name}' was excluded during scaling and cannot be recovered.")
    
    # Ensure the column name exists in the scaler
    if column_name not in scaler.feature_names_in_:
        raise ValueError(f"Column '{column_name}' was not scaled with the provided scaler.")

    # Get the index of the column in the scaler's feature_names_in_
    column_index = list(scaler.feature_names_in_).index(column_name)

    # Extract only the scaled values for the specific column
    if isinstance(predicted_column, pd.Series):
        predicted_column_values = predicted_column.values.reshape(-1, 1)
    elif isinstance(predicted_column, np.ndarray):
        predicted_column_values = predicted_column.reshape(-1, 1)
    else:
        raise ValueError("Input must be a Pandas Series or NumPy array.")

    # Create a dummy array to hold the scaled values (needed for inverse_transform)
    numeric_columns = list(scaler.feature_names_in_)
    dummy_array = np.zeros((predicted_column_values.shape[0], len(numeric_columns)))
    dummy_array[:, column_index] = predicted_column_values.flatten()

    # Perform inverse transformation using MinMaxScaler
    recovered_values = scaler.inverse_transform(dummy_array)[:, column_index]

    # Return as the same type as input
    if isinstance(predicted_column, pd.Series):
        return pd.Series(recovered_values, index=predicted_column.index, name=column_name)
    return recovered_values

