import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to your dataset
dataset_number_dir = r'E:\ASLdevhouse\ISL\Dataset_alphabet'

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

# Train generator for ISL numbers
train_generator_number = train_datagen.flow_from_directory(
    dataset_number_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'  # Specify training subset
)

validation_generator_number = train_datagen.flow_from_directory(
    dataset_number_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'  # Specify validation subset
)

# Define and train the model for ISL numbers
model_numbers = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')  # 6 classes: numbers 0-5
])

model_numbers.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

# Train the model for ISL numbers
history_numbers = model_numbers.fit(
    train_generator_number,
    steps_per_epoch=train_generator_number.samples // batch_size,
    validation_data=validation_generator_number,
    validation_steps=validation_generator_number.samples // batch_size,
    epochs=10
)

# Save the model for ISL numbers in .pb format
tf.saved_model.save(model_numbers, "isl_alpha")
