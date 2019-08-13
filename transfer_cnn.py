import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

generator = ImageDataGenerator(rescale=1./255,)
val_datagen = ImageDataGenerator( rescale = 1.0/255. )
base_dir = r"F:\Assesment_cartesian\ex_2/"
val_dir = os.path.join(base_dir, 'img_val')
train = generator.flow_from_directory("./image",target_size=(100,100),class_mode="categorical",batch_size=32)
validation_generator = val_datagen.flow_from_directory( val_dir,
                                                          class_mode  = 'categorical',
                                                          target_size = (100, 100))

pre_model = InceptionV3(input_shape= (100,100,3),weights="imagenet",include_top=False)
for layer in pre_model.layers:
    layer.trainable = False

last_layer = pre_model.get_layer("mixed7")
last_output = last_layer.output
# main_model = tf.keras.models.Model(model.)
x = layers.Flatten()(last_output)
x = layers.Dense(1024,activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1024,activation="relu")(x)
x = layers.Dense(9,activation="softmax")(x)

model = tf.keras.models.Model(pre_model.input,x)
model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics = ['acc'])


history = model.fit_generator(
            train,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 20,
            validation_steps = 3,
            verbose = 2)
model.save('transfer1.h5py')
# accuracy = history.history['acc']
# loss = history.history['loss']
# val_accuracy = history.history['val_acc']
# val_loss = history.history['val_loss']
#
# epochs = range(len(accuracy))
# plt.plot(epochs, accuracy, 'r', label='Training accuracy')
# plt.plot(epochs,val_accuracy, 'b', label='Validation accuracy')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.savefig('transfer_acc_7.png')
# plt.gcf().clear()
#
# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('transfer_loss_7.png')
#
#
# scores = model.evaluate_generator(validation_generator, 45)
# print("  Test Accuracy = ", scores[1])
# print("  Test Loss =", scores[0])
#
# # Evaluation of predicted result and print confusion matrix and classification report
# batch_size = 5
# # y_pred = model.predict_generator(test_generator, 1300 // batch_size+1)
# y_pred = model.predict_generator(validation_generator, validation_generator.samples // validation_generator.batch_size)
# y_pred = np.argmax(y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(validation_generator.classes, y_pred))
#
# print('Classification Report')
# num_classes = 9
# target_names = ["Class {}".format(i) for i in range(num_classes)]
# print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
#
#
# # Print result in csv files
# labels = validation_generator.class_indices
# labels = dict((v, k) for k, v in labels.items())
# predictions = [labels[k] for k in y_pred]
# filename = validation_generator.filenames
# results = pd.DataFrame({"Filename": filename, " Predictions ": predictions})
# results.to_csv("result_7.csv", index=False)