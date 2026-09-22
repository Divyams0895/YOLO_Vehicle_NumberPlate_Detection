# 🚘 AI Vehicle Number Plate Detection using YOLO

A computer vision application that detects vehicle number plates from images, extracts the registration number using OCR, and retrieves the corresponding vehicle record through a Streamlit web interface.

## 📌 Project Overview

This project combines **YOLO object detection**, **OpenCV image processing**, **Tesseract OCR**, **Pandas**, and **Streamlit**.

The system follows this pipeline:

```text
Vehicle Image
      ↓
YOLO Number Plate Detection
      ↓
Plate Cropping
      ↓
Image Preprocessing
      ↓
Tesseract OCR
      ↓
Plate Text Cleaning
      ↓
Vehicle Database Search
      ↓
Vehicle Information
```

## ✨ Features

* 📷 Upload vehicle images
* 🤖 Detect number plates using YOLO
* 🔲 Draw bounding boxes around detected plates
* ✂️ Crop detected plates
* 🖼️ Resize and preprocess plate images
* 🔤 Extract registration numbers using Tesseract OCR
* 🧹 Clean and normalize OCR output
* 🔎 Search registration numbers in a CSV database
* 📋 Display matching vehicle information
* 📊 Display YOLO detection confidence
* 🌐 Streamlit-based web interface

## 🛠️ Technologies

| Technology         | Purpose                       |
| ------------------ | ----------------------------- |
| Python             | Core programming              |
| YOLO / Ultralytics | Number plate detection        |
| OpenCV             | Image processing              |
| Tesseract OCR      | Number plate text recognition |
| Pandas             | Vehicle database handling     |
| Streamlit          | Web interface                 |
| NumPy              | Image/data processing         |



If you found this project interesting, feel free to ⭐ the repository and connect with me on LinkedIn.
