import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to your dataset
dataset_words_dir = r'E:\Asldevhouse\ASL\Dataset_words'

# Define image dimensions
img_width, img_height = 64, 64

# Define batch size
batch_size = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting data into train/validation sets
)

# Train generator for ASL words
train_generator_words = train_datagen.flow_from_directory(
    dataset_words_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'  # Specify training subset
)

validation_generator_words = train_datagen.flow_from_directory(
    dataset_words_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'  # Specify validation subset
)

# Define and train the model for ASL words
model_words = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='softmax')  # 8 classes: words 0 Can, 1 Hello, 2 Help, 3 Me, 4 Nobody, 5 Sad, 6 Understand, 7 Why
])

model_words.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Train the model for ASL words
history_words = model_words.fit(
    train_generator_words,
    steps_per_epoch=train_generator_words.samples // batch_size,
    validation_data=validation_generator_words,
    validation_steps=validation_generator_words.samples // batch_size,
    epochs=10
)

# Save the model for ASL words in .pb format
tf.saved_model.save(model_words, "ASL_word_trained")
