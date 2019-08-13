import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile

# local_weights_file = r'F:\Assesment_cartesian\ex_2\temp\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(100, 100, 3),
                                include_top=False,
                                weights='imagenet')

# pre_trained_model.load_weights("imagenet")

# freezing the layers of the neural network
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final softmax layer for classification
x = layers.Dense(9, activation='softmax')(x)

model = Model( pre_trained_model.input, x)

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics = ['acc'])



base_dir = r"F:\Assesment_cartesian\ex_2/"
train_dir = os.path.join(base_dir, 'images')
val_dir = os.path.join(base_dir, 'img_val')

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 3, color_mode='rgb',
                                                    class_mode = 'categorical',
                                                    target_size = (100, 100))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = val_datagen.flow_from_directory( val_dir,
                                                          batch_size  = 3, color_mode='rgb',
                                                          class_mode  = 'categorical',
                                                          target_size = (100, 100))

history = model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 150,
            validation_steps = 3,
            verbose = 2)

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
plt.savefig('transfer_acc_6.png')
plt.gcf().clear()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('transfer_loss_6.png')


scores = model.evaluate_generator(validation_generator, 45)
print("  Test Accuracy = ", scores[1])
print("  Test Loss =", scores[0])

# Evaluation of predicted result and print confusion matrix and classification report
batch_size = 5
# y_pred = model.predict_generator(test_generator, 1300 // batch_size+1)
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
results.to_csv("result_6.csv", index=False)