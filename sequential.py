import tensorflow as tf
from tensorflow import keras
# import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys
import tracemalloc
import time

sys.path.append(os.path.abspath(os.path.join('..', 'datasets')))
base_dir = "datasets/lfw/"

tracemalloc.start() # starting tracing memory allocation
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2) 
train_dataset = data_generator.flow_from_directory(base_dir, target_size=(100, 100), color_mode='rgb', batch_size=15, subset='training', class_mode="categorical")
validation_dataset = data_generator.flow_from_directory(base_dir, target_size=(100, 100), color_mode='rgb', batch_size=15, subset='validation', class_mode="categorical")
# print(validation_dataset.class_indices)
# print(train_dataset.classes)
IMAGE_SHAPE = (100, 100, 3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation="elu"), 
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="elu"), 
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="elu"), 
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5722, activation="softmax") 
])

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) 
model.summary()

start_training_time = time.time()
# Train the model
model_fit = model.fit(train_dataset, epochs=35, validation_data = validation_dataset) # steps_per_epoch=

end_training_time = time.time()

# Get Memory consuption
first_size, first_peak = tracemalloc.get_traced_memory()

""" Make Predictions based on Validation dataset """
print("Processing predictions")
predictions = model.predict(validation_dataset)
score = model.evaluate(validation_dataset, verbose=0)
print(f'\n\n[PREDICT] Test loss: {score[0]} | Test accuracy: {score[1]}')

accuracy = model_fit.history['accuracy']
val_accuracy = model_fit.history['val_accuracy']
loss = model_fit.history['loss']
val_loss = model_fit.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 2.0])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()

# Saving the Trained Model to the keras h5 format; so in future, if we want to convert again, we don't have to go through the whole process again
saved_model_dir = 'save/LFW_sequential_fine_tuning.h5'
model.save(saved_model_dir)
print("\nModel Saved to save/LFW_sequential_fine_tuning.h5")

# Tracing training time
print("training took {:.4f} seconds".format(end_training_time - start_training_time))
# tracing memory alocation
print(f"Memory Allocation:\nFirst memory size: {first_size=} \nPeak:{first_peak=}")

print("\n[EXIT] - end script.")

exit() # exit the program