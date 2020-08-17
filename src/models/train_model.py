from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Paths
proj_path = Path().absolute()
data_raw_path = Path(proj_path, 'data', 'raw')
model_path = Path(proj_path, 'models')

# File names
data_filename = 'datasets_19_420_Iris.csv'
model_name = 'final_model.pkl'

# Variable names
X_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
y_names = ['Species']

# Functions
def process_data(df, X_names, y_names):
    df = df.copy()
    X = df.loc[:, X_names]
    y = df.loc[:, y_names]

    return X, y


if __name__ == "__main__":

    # Read in data
    df = pd.read_csv(Path(data_raw_path, 'datasets_19_420_Iris.csv'))

    # Process data
    X, y = process_data(df, X_names, y_names)

    # Train model
    rf = RandomForestClassifier()
    rf.fit(X, y.values.ravel())

    # Export model using joblib
    joblib.dump(rf, Path(model_path, model_name))
