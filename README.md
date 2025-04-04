# Sri Lankan Number Plate Recognition

This project is an Automatic Number Plate Recognition (ANPR) system designed for Sri Lankan vehicles. It uses Python, OpenCV, and PyTesseract to detect and recognize number plates from both static images and real-time webcam feeds. The system identifies number plates, extracts the text using OCR, logs the results in a CSV file, and displays the detected plates with a green bounding box and text overlay.

## Features
- Detects Sri Lankan number plates (yellow or white background with black text).
- Recognizes alphanumeric characters using PyTesseract OCR.
- Logs detected plates with timestamps and provinces in a CSV file.
- Displays detected plates with a green bounding box and text overlay.
- Supports both batch processing of images and real-time webcam processing.

## Project Structure

- `number_plate_reader_images.py`: Processes static images (`Number_plate_1.jpg` to `Number_plate_5.jpg`) in the `Number_plates/` directory.
- `number_plate_reader_webcam.py`: Processes real-time video from a webcam.
- `vehicle_log.csv`: The CSV file where detected plates are logged with timestamps and provinces.
- `Number_plates/`: Directory for input images (not included in the repository; add your own images).
- `requirements.txt`: Lists the Python dependencies required to run the project.
- `SETUP.txt`: Detailed setup instructions for running the project.

## What This Project Achieves

- **Number Plate Detection**: Identifies Sri Lankan number plates in images or video using color filtering (yellow or white plates) and contour detection.
- **Text Recognition**: Extracts alphanumeric characters from detected plates using PyTesseract OCR.
- **Province Identification**: Maps the first two letters of the plate to a Sri Lankan province (e.g., `WP` → `Western Province`).
- **Logging**: Stores detected plates in a CSV file for record-keeping.
- **Visualization**: Displays images or video frames with a green bounding box and detected text for visual confirmation.
- **Real-Time Processing**: Supports live number plate recognition using a webcam.

## Limitations

- The system may struggle with low-quality images, poor lighting, or angled plates.
- OCR accuracy depends on the clarity of the plate and the preprocessing steps.
- The image script processes a fixed set of images (`Number_plate_1.jpg` to `Number_plate_5.jpg`). Modify the script to handle other images.
- Webcam performance may vary depending on hardware and lighting conditions.

## Future Improvements

- Improve OCR accuracy by training a custom model for Sri Lankan number plates.
- Handle more complex scenarios (e.g., tilted plates, multiple plates in one frame).
- Add a web interface for easier interaction.
- Implement plate tracking to avoid duplicate detections in video.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [OpenCV](https://opencv.org/) for image processing.
- Uses [PyTesseract](https://github.com/madmaze/pytesseract) for OCR.
- Inspired by various ANPR projects on GitHub.
