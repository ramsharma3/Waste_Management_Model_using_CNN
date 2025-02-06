import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'), 
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)  
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'DATASET/TRAIN',  
    target_size=(64, 64),  
    batch_size=16,  
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'DATASET/TEST',  
    target_size=(64, 64),  
    batch_size=16,  
    class_mode='binary'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5  
)

# save the model
model.save('waste_classification_model_optimized.h5')

# Testing of model 

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('waste_classification_model_optimized.h5')

# Load and preprocess the image
img_path = 'DATASET/TESTING_02.jpg'
img = image.load_img(img_path, target_size=(64, 64))  # Reduced image size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Predict
prediction = model.predict(img_array)
print(f'Prediction: {"Recyclable" if prediction > 0.5 else "Organic"}')

# Analysis of model 

# Example: Print accuracy and loss for each epoch
for epoch in range(len(history.history['accuracy'])):
    print(f"Epoch {epoch + 1}")
    print(f"Training Accuracy: {history.history['accuracy'][epoch]}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][epoch]}")
    print(f"Training Loss: {history.history['loss'][epoch]}")
    print(f"Validation Loss: {history.history['val_loss'][epoch]}")
    print("-----------------------------")

# Graphic Visualisation using matplot library

import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('best_waste_classification_model.h5')
