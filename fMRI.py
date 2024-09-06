import tensorflow as tf from tensorflow.keras models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout from tensorflow.keras-preprocessing-image import ImageDataGenerator from tensorflow.keras.optimizers import Adam
# Define the model
model = Sequential()
# Convolutional layer 1
model-add (Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add (MaxPooling2D(pool_size=(2, 2)))
# input sh
# Convolutional layer 2
model. add (Conv2D(64, (3, 3), activation='relu')) model. add (MaxPooling2D(pool_size=(2, 2)))
# Convolutional layer 3
model. add (Conv2D(128, (3, 3), activation-'relu')) model. add (MaxPooling2D(pool_size=(2, 2)))
# Flatten the layers
model. add (Flatten())
# Fully connected layer
model-add (Dense(128, activation='relu'))
model. add (Dropout (0.5))
# Output layer
model. add (Dense(1, activation='sigmoid')) # Binary classification: cancerous vs non-cance
# Compile the model
model. compile(optimizer-Adam(Ir=0.0001), loss='binary_crossentropy', metrics-['accuracy'])
# Image data generators for data augmentation
train_datagen - ImageDataGenerator(rescale-1./255, rotation_range-20, zoom_range=0.2, hori validation datagen - ImageDataGenerator (rescale-1./255)
# Load and preprocess the MRI/MRI data (Assumed to be in 'train/' and 'val/' directories)
train_generator - train_datagen. flow_from_directory('data/train/', target_size=(128, 128), validation generator - validation_datagen.flow_from _directory('data/val/', target_size=(12
# Train the model
history = model. fit(train_generator, validation data validation generator, epochs=50)
# Save the model
model save('mri_cancer _detection _mode1.h5')
# Evaluate the model
loss, accuracy = model, evaluate(validation _generator)
print(f'Validation Accuracy: {accuracy*100:.2f}%')