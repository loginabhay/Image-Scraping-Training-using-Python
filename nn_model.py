import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd
import tensorflow as tf
from matplotlib import  pyplot as plt

base_dir = r"F:\Assesment_cartesian\ex_2/"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
#test_dir = os.path.join(base_dir, 'test')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100), color_mode='rgb', batch_size=5, class_mode='categorical')
val_generator = train_datagen.flow_from_directory(val_dir, target_size=(100, 100), color_mode='rgb', batch_size=5, class_mode='categorical', shuffle=False)

for data_batch, label_batch in train_generator:
    print('data batch shape:',data_batch.shape)
    print('label batch shape:', label_batch.shape)
    break

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(100,100,3)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(12, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, epochs=50, verbose=2)

# Results - Accuracy
scores = model.evaluate_generator(val_generator, 60)
print("  Test Accuracy = ", scores[1])
print("  Test Loss =", scores[0])

# Evaluation of predicted result and print confusion matrix and classification report
batch_size = 20
# y_pred = model.predict_generator(test_generator, 1300 // batch_size+1)
y_pred = model.predict_generator(val_generator, val_generator.samples // val_generator.batch_size)
y_pred = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(val_generator.classes, y_pred))

print('Classification Report')
num_classes = 9
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(val_generator.classes, y_pred, target_names=target_names))


# Print result in csv files
labels = val_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in y_pred]
filename = val_generator.filenames
results = pd.DataFrame({"Filename": filename, " Predictions ": predictions})
results.to_csv("result_nn_1.csv", index=False)
