import numpy as np
import time
from sklearn import svm
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import tensorflow as tf
from matplotlib import  pyplot as plt

base_dir = r"F:\Assesment_cartesian\ex_2/"
train_dir = os.path.join(base_dir, 'images')
val_dir = os.path.join(base_dir, 'img_val')
#test_dir = os.path.join(base_dir, 'test')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# data preprocessing with the help of flow from directory
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(28, 28), color_mode='rgb', batch_size=5, class_mode='categorical')
validation_generator = train_datagen.flow_from_directory(val_dir, target_size=(28, 28), color_mode='rgb', batch_size=5, class_mode='categorical', shuffle=False)

for data_batch, label_batch in train_generator:
    print('data batch shape:',data_batch.shape)
    print('label batch shape:', label_batch.shape)
    break
# Creating sequential CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=(28,28,3), padding='same'))
model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.20))

model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(9, activation=tf.nn.softmax))


model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
t=time.time()
#
# earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
# checkpoint = tf.keras.callbacks.ModelCheckpoint('model-cnn-epoch{epoch:02d}.h5py', monitor='val_acc', verbose=1,
#                                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)
#
# callbacks_list = [earlystop, checkpoint]
# Training the model
history = model.fit_generator(train_generator, epochs=10, validation_data=validation_generator, verbose=2)
print('Training time: %s' % (t - time.time()))
# model.save('cnn1_epoch.h5py')

accuracy = history.history['acc']
loss = history.history['loss']
val_accuracy = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs,val_accuracy, 'b', label='Validation accuracy')
plt.xlabel('Number of Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('cnnmodel_accuracy3.png')
plt.gcf().clear()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('cnnmodel_loss3.png')

scores = model.evaluate_generator(validation_generator, 45)
print("  Test Accuracy = ", scores[1])
print("  Test Loss =", scores[0])

# Evaluation of predicted result and print confusion matrix and classification report
batch_size = 5

y_pred = model.predict_generator(validation_generator, validation_generator.samples // validation_generator.batch_size)
y_pred = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')
num_classes = 9
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


# Print result in csv files
labels = validation_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in y_pred]
filename = validation_generator.filenames
results = pd.DataFrame({"Filename": filename, " Predictions ": predictions})
results.to_csv("result_cnn2.csv", index=False)