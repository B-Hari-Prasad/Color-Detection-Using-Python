# Project Overview

This project is a Python-based Color Detection Application that identifies colors from both uploaded images and live webcam feed. It includes a secure login authentication system using SQLite, allowing only registered users to access the application while also supporting new user creation.

# What the Application Does

Users can upload an image and click on any point to detect the color at that pixel. The application displays:

✔️ Color Name

✔️ RGB Values

✔️ HEX Code

It also supports real-time color detection through a live camera interface, allowing instant identification of colors from surrounding objects.

# Smart Color Recognition

A CNN-based color classification model using TensorFlow enhances accuracy by intelligently predicting colors instead of depending only on basic thresholds. A color dataset (colors.csv) is used to map RGB and HEX values precisely.

# User Interface

The graphical user interface is developed using Tkinter and CustomTkinter, providing:

A modern look

Simple navigation

Smooth user interaction

OpenCV is used for image and video processing while Pillow manages image display.

# Secure Login System

The application integrates an SQLite database to:

Store user credentials

Authenticate users

Create new user accounts securely

# Ideal For

Students learning Computer Vision

Designers working with color selection and matching

Developers exploring Machine Learning and GUI development

Anyone interested in intelligent color detection applications

# Key Technologies Used

Python

OpenCV

TensorFlow / Keras

SQLite

Pillow

Pandas & NumPy

Tkinter / CustomTkinter

# Outcome

This project demonstrates real-time image processing, database connectivity, GUI development, and AI-powered color detection in one integrated solution, delivering a smart, efficient, and user-friendly color detection application.
https://color-detection-application/login
