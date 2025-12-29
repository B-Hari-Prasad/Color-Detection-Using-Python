import numpy as np
import pandas as pd
import cv2
import imutils
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import customtkinter as ctk

# Load pre-trained CNN model for color classification
model = load_model("color_classification_model.h5")

# Load color dataset
index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv("colors.csv", header=None, names=index)

# Function to preprocess color for CNN prediction
def preprocess_color(color):
    return np.array(color, dtype=np.float32).reshape(1, 1, 1, 3) / 255.0

# Function to predict color name using CNN
def predict_color_name(r, g, b):
    input_color = preprocess_color([r, g, b])
    prediction = model.predict(input_color)
    predicted_index = np.argmax(prediction)
    return df.loc[predicted_index, "color_name"], df.loc[predicted_index, "hex"]

# Function to select an image file
def select_image():
    inputfile = filedialog.askopenfilename(title="Select an image file",
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if inputfile:
        img = cv2.imread(inputfile)
        img_copy = img.copy()
        imgHeight, imgWidth = img.shape[:2]

        def getRGBvalue(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                b, g, r = img[y, x]
                cname, hex_code = predict_color_name(r, g, b)
                cv2.putText(img, f'{cname} ({hex_code})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Image", img)

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", getRGBvalue)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to start live video feed
def start_video_feed():
    camera = cv2.VideoCapture(0)
    def identify_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ret, frame = camera.read()
            b, g, r = frame[y, x]
            cname, hex_code = predict_color_name(r, g, b)
            cv2.putText(frame, f'{cname} ({hex_code})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Live Feed", frame)

    cv2.namedWindow("Live Feed")
    cv2.setMouseCallback("Live Feed", identify_color)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

# Create GUI using CustomTkinter
ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("Enhanced Color Detection")
root.geometry("800x600")

frame = ctk.CTkFrame(root)
frame.pack(pady=20, padx=20, fill="both", expand=True)

heading_label = ctk.CTkLabel(frame, text="Color Detection with CNN", font=("Arial", 24))
heading_label.pack(pady=20)

btn_image = ctk.CTkButton(frame, text="Upload Image", command=select_image)
btn_image.pack(pady=10)

btn_video = ctk.CTkButton(frame, text="Open Camera", command=start_video_feed)
btn_video.pack(pady=10)

root.mainloop()
