import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten, Dense, Dropout, BatchNormalization, LeakyReLU

classes = {0: 'G1 -- Fist', 2: 'G2 -- index finger', 3: 'G3 -- little finger', 4: 'G4 -- index finger and thumb', 5: 'G5 -- Victory!', 6: 'G6 -- Three', 7: 'G7 -- thumb and index finger', 8: 'G8 -- love you', 9: 'G9 -- Five', 1: 'G10 -- index, middle and little finger' }

NUM_CLASSES = 10

def morph_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    img = cv2.inRange(img, (0, 50, 20), (255, 255, 255))
    _, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.morphologyEx(
        img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    img = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    return img

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), padding="same",
               activation="relu", input_shape=(64, 64, 1)),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(),

        Conv2D(64, (3, 3), padding="same", activation="relu"),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), padding="same"),
        LeakyReLU(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.7),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    return model

def predict_live(trained_model):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while cap.isOpened():

        ret, frame = cap.read()
        img = morph_image(frame)

        if not ret:
            continue

        length = len(img)
        height = len(img[0])
        img_fr = img[int(length/2 - 150):int(length/2 + 150), int(height/2 - 150):int(height/2 + 150)]
        cv2.imshow('Cropped', img_fr)

        img_fr = cv2.resize(img_fr, (64, 64))
        img_fr = np.expand_dims(img_fr, axis=0)
        img_fr = np.expand_dims(img_fr, axis=3)
        result = trained_model.predict(img_fr)
        result = result > 0.9

        has_true = np.any(result, axis=1)
        result = result.argmax(axis=1)

        if has_true:
            cv2.putText(
                frame, f"result: {classes[result[0]]}", (
                    10, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA
            )
        cv2.rectangle(frame, (int(height/2 - 150),int(length/2 - 150)), (int(height/2 + 150), int(length/2 + 150)), (255, 0, 0), 1)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':

    model = create_model(NUM_CLASSES)
    model.load_weights('hand_symbols.h5')
    predict_live(model)
