import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Data preparation completed!")

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

print("CNN model created!")

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64
)

print("Model training completed!")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_accuracy)

# Save the trained model
model.save("handwritten_character_model.keras")

print("Model saved successfully!")

# Predict one test image
import numpy as np

prediction = model.predict(x_test[0:1], verbose=0)
predicted_digit = np.argmax(prediction)

print("Actual digit:", y_test[0])
print("Predicted digit:", predicted_digit)