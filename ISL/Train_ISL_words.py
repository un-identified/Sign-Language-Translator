import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
dataset_words_dir = r'E:\Isldevhouse\ISL\Dataset_words'
img_width, img_height = 64, 64
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  
)


train_generator_words = train_datagen.flow_from_directory(
    dataset_words_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'  
)

validation_generator_words = train_datagen.flow_from_directory(
    dataset_words_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation' 
)


model_words = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='softmax') 
])

model_words.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

history_words = model_words.fit(
    train_generator_words,
    steps_per_epoch=train_generator_words.samples // batch_size,
    validation_data=validation_generator_words,
    validation_steps=validation_generator_words.samples // batch_size,
    epochs=10
)

tf.saved_model.save(model_words, "isl_words")
