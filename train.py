import pandas as pd
from models import LSTMModel, RandomForestModel
from data_preprocessing import BikeRentalPreprocessor
from sklearn.model_selection import train_test_split

# Load and preprocess data
preprocessor = BikeRentalPreprocessor()
df = preprocessor.preprocess("data/bike_rentals.csv")

# Split data
X = df.drop('Rented_Bike_Count', axis=1)
y = df['Rented_Bike_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train LSTM
lstm = LSTMModel(input_shape=(X_train.shape[1], 1))
lstm.model.fit(X_train, y_train, epochs=10)

# Train Random Forest
rf = RandomForestModel()
rf.model.fit(X_train, y_train)

# Save models
lstm.model.save("models/lstm_model.h5")
import joblib
joblib.dump(rf.model, "models/rf_model.pkl")
