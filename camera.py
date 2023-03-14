import cv2
import datetime
import os

cap = cv2.VideoCapture(0)

def capture_image():
    ret, frame = cap.read()
    filename = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + '.jpg'
    filepath = os.path.join('/path/to/directory/', filename)
    cv2.imwrite(filepath, frame)

def process_image():
    latest_file = max(os.listdir('/path/to/directory/'), key=os.path.getctime)
    filepath = os.path.join('/path/to/directory/', latest_file)
    image = cv2.imread(filepath)
    # Process the image here

capture_image()
process_image()

cap.release()
cv2.destroyAllWindows()
