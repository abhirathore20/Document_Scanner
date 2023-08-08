import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Initialize variables
count = 0
scale = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
WIDTH, HEIGHT = 1920, 1080

# Set the webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Define a function for basic image processing
def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold


# Define a function to detect the document within the captured frame
def scan_detection(image):
    global document_contour

# Initialize a default contour representing the whole frame
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    # Draw the detected document contour on the frame
    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)

# Define a function to center text on an image
def center_text(image, text):
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, 2, (255, 0, 255), 5, cv2.LINE_AA)


while True:

    _, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame_copy = frame.copy()

    # Detect the document in the frame and draw its contour
    scan_detection(frame_copy)

    cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))

# Perform perspective transformation (warping) to extract the document
    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    cv2.imshow("Warped", cv2.resize(warped, (int(scale * warped.shape[1]), int(scale * warped.shape[0]))))

# Process the warped image and display the result
    processed = image_processing(warped)
    processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
    cv2.imshow("Processed", cv2.resize(processed, (int(scale * processed.shape[1]),
                                                   int(scale * processed.shape[0]))))
 # Wait for user key press
    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == 27:
        break

    elif pressed_key == ord('s'):
        cv2.imwrite("output/scanned_" + str(count) + ".jpg", processed)
        count += 1

        center_text(frame, "Scan Saved")
        cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))
        cv2.waitKey(500)

    elif pressed_key == ord('o'):
        file = open("output/recognized_" + str(count - 1) + ".txt", "w")
        ocr_text = pytesseract.image_to_string(warped)
        # print(ocr_text)
        file.write(ocr_text)
        file.close()

        center_text(frame, "Text Saved")
        cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))
        cv2.waitKey(500)

cv2.destroyAllWindows()