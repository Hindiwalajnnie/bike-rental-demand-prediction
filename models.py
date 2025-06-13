import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor

class LSTMBikePredictor:
    def __init__(self, sequence_length=24, features=10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = StandardScaler()
    
    def build_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, 
                 input_shape=(self.sequence_length, self.features)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', 
                     loss='mse', 
                     metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_sequences(self, data, target_col):
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:(i + self.sequence_length)])
            targets.append(target_col[i + self.sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=32,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        return history

class RandomForestBikePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.feature_importance = None
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def get_feature_importance(self):
        return self.feature_importance
