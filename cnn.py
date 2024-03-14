from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)

base_model.summary()

base_model.trainable = False

# Adding New Layers

inputs = keras.Input(shape=(224, 224, 3))
# Separately from setting trainable on the model, we set training to False
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)

# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)

n_classes = 3
# A Dense classifier for multi-class classification with softmax activation
outputs = keras.layers.Dense(n_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Let us take a look at the model, now that we have combined the pre-trained model with the new layers.
model.summary()

# Compiling the Model
# model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# Augmenting the Data
'''
# Define a preprocessing function to convert grayscale to RGB
def grayscale_to_rgb(images):
    # images is a numpy array from ImageDataGenerator, shape (batch_size, img_height, img_width, 1)
    # We repeat the grayscale channel 3 times to create an RGB image
    images_rgb = np.repeat(images, 3, axis=-1)
    return images_rgb
'''

# Create a data generator
datagen_train = ImageDataGenerator(
    #preprocessing_function = grayscale_to_rgb,
    #rescale = 1./255,
    samplewise_center=True,  # set each sample mean to 0
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,
)

# No need to augment validation data
datagen_valid = ImageDataGenerator(
    #preprocessing_function = grayscale_to_rgb,
    #rescale = 1./255,
    samplewise_center=True
    )

# Loading the Data

# load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    "/Users/mac/Bladder_cancer/images/train",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=8,
)

# load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
    "/Users/mac/Bladder_cancer/images/test",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=8,
)

# Train the model

# Time to train our model and see how it does. Recall that when using a data generator, we have to explicitly set the number of `steps_per_epoch``:

history = model.fit(train_it, steps_per_epoch=12, validation_data=valid_it, validation_steps=4, epochs=20)