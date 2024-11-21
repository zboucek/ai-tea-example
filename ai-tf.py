import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the labels
y_encoded = keras.utils.to_categorical(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Build the model
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    validation_split=0.2, 
    verbose=0
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Accuracy: {test_accuracy:.2%}')

# Prediction function
def predict_iris(features):
    # Scale the input features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction)
    
    return iris.target_names[predicted_class]

# Example prediction
sample_features = [5.1, 3.5, 1.4, 0.2]  # Sample iris measurements
print(f"\nPredicted class for {sample_features}:")
print(predict_iris(sample_features))