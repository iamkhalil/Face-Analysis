#!/usr/bin/env python3

from tensorflow.keras.preprocessing import image
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import time


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

interpreter = Interpreter(model_path="emotion_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    t1 = time.time()
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        interpreter.set_tensor(input_details[0]['index'], img_pixels)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # Age prediction
        age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

        blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        predicted_age = age_list[age_preds[0].argmax()]

        label = "{} | {}".format(predicted_emotion, predicted_age)
        cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    t2 = time.time()
    print("Time = {}".format(t2 - t1))

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial emotion/age analysis', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
