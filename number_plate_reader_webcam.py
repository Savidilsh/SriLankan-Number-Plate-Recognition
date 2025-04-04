import cv2
import numpy as np
import pytesseract
import datetime
import csv
import os

# Fix Qt error by setting the XCB platform
os.environ["QT_QPA_PLATFORM"] = "xcb"

# CSV file path (relative to the repository root)
log_file = "vehicle_log.csv"

# Initialize CSV if it doesn’t exist
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Plate Number", "Province"])

# Province dictionary
province_dict = {
    "WP": "Western Province", "SP": "Southern Province", "CP": "Central Province",
    "NP": "Northern Province", "EP": "Eastern Province", "NC": "North Central Province",
    "NW": "North Western Province", "SG": "Sabaragamuwa Province", "UP": "Uva Province"
}

# Kernel for dilation
kernel = np.ones((5,5), np.uint8)

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# To avoid logging the same plate multiple times, keep track of recent detections
recent_plates = []
max_recent_plates = 10  # Store up to 10 recent plates to avoid duplicates

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Image preprocessing
    frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=-50)
    
    # Convert to HSV for color filtering (yellow and white plates)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Yellow plate filter
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # White plate filter
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 100000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 6.0:
                plate_roi = frame[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.resize(gray_roi, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                gray_roi = cv2.equalizeHist(gray_roi)
                _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plate_text = pytesseract.image_to_string(thresh_roi, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                plate_text = plate_text.strip()
                if len(plate_text) >= 6:
                    # Check if this plate was recently detected to avoid duplicates
                    if plate_text not in recent_plates:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print(f"Detected Plate: {plate_text}")
                        province_code = plate_text[:2]
                        province = province_dict.get(province_code, "Unknown Province")
                        print(f"Province: {province}")
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(log_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, plate_text, province])
                        # Add to recent plates
                        recent_plates.append(plate_text)
                        if len(recent_plates) > max_recent_plates:
                            recent_plates.pop(0)

    # Display the frame
    cv2.imshow("Number Plate Reader (Webcam)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
