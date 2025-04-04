# Sri Lankan Number Plate Recognition

This project is an Automatic Number Plate Recognition (ANPR) system designed for Sri Lankan vehicles. It uses Python, OpenCV, and PyTesseract to detect and recognize number plates from images. The system processes images of vehicles, identifies the number plate, extracts the text using OCR, and logs the results in a CSV file. It also displays the detected plates with a green bounding box and text overlay.

## Features
- Detects Sri Lankan number plates (yellow or white background with black text).
- Recognizes alphanumeric characters using PyTesseract OCR.
- Logs detected plates with timestamps and provinces in a CSV file.
- Displays images with a green bounding box and detected text for each plate.
- Supports batch processing of multiple images.

## Prerequisites
- **Hardware**: Raspberry Pi (or any computer with a graphical desktop environment).
- **Operating System**: Tested on Raspberry Pi OS (Linux-based).
- **Dependencies**:
  - Python 3.7 or higher
  - OpenCV (`opencv-python`)
  - PyTesseract (`pytesseract`)
  - Tesseract OCR (system dependency)
  - NumPy (`numpy`)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/SriLankan-Number-Plate-Recognition.git
   cd SriLankan-Number-Plate-Recognition
