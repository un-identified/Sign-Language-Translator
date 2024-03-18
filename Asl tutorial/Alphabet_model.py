import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
import cv2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import skimage
from skimage.transform import resize
import os

batch_size = 32  
image_size = 64  
target_dimensions = (image_size, image_size, 3)  
num_classes = 29  
epochs = 30  

train_len = 87000
train_dir = r'Asl tutorial\asl_alphabet\asl_alphabet_train'

def get_data(folder):
    X = np.empty((train_len, image_size, image_size, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int64)
    cnt = 0  
    label_mapping = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
        'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
        'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
        'nothing': 27, 'space': 28
    }  
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            label = label_mapping.get(folderName, 29)
            for image_filename in os.listdir(folder + '/' + folderName): 
                img_file = cv2.imread(folder + '/' + folderName + '/' + image_filename)
                if img_file is not None:  
                    img_file = skimage.transform.resize(img_file, (image_size, image_size, 3))  
                    img_arr = np.asarray(img_file).reshape((-1, image_size, image_size, 3))  
                    X[cnt] = img_arr  
                    y[cnt] = label  
                    cnt += 1
    print('Done loading')
    return X, y


def splitting(X_train, y_train):
    X_data = X_train
    y_data = y_train


    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42, stratify=y_data)


    y_cat_train = to_categorical(y_train, 29)  
    y_cat_test = to_categorical(y_test, 29)  

    del X_data
    del y_data
    print('Splitting complete')
    return X_train, X_test, y_train, y_test, y_cat_train, y_cat_test

def modelbuilding(X_train, y_cat_train, X_test, y_cat_test):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=target_dimensions))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D((2, 2)))  
    model.add(Conv2D(128, (3, 3)))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3)))  
    model.add(Activation('relu')) 
    model.add(MaxPooling2D((2, 2)))  
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(29, activation='softmax')) 

    model.summary()

   
    early_stop = EarlyStopping(monitor='val_loss', patience=4)


    checkpoint_filepath = os.path.join('checkpoints', 'model.{epoch:03d}-{val_loss:.3f}.h5')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

   
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(X_train, y_cat_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=2,
              validation_data=(X_test, y_cat_test),
              callbacks=[early_stop, model_checkpoint_callback])

    
    model.save('AaaaSL2.h5')
    print('Model saved')
    return model


def eval(model, X_test, y_test, y_cat_test):
    metrics = model.evaluate(X_test, y_cat_test, verbose=0)
    print(f'Test Loss: {metrics[0]}')
    print(f'Test Accuracy: {metrics[1]}')

    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)

    print(classification_report(y_test, predictions))

    plt.figure(figsize=(12, 12))
    sns.heatmap(confusion_matrix(y_test, predictions))
    plt.show()

if __name__ == "__main__":
    X_train, y_train = get_data(train_dir)
    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = splitting(X_train, y_train)
    model = modelbuilding(X_train, y_cat_train, X_test, y_cat_test)
    eval(model, X_test, y_test, y_cat_test)
