import cv2
import numpy as np
import os 
import tkinter as tk
import sys
import tkinter.messagebox as messagebox

# Load pre-trained recognizer and cascade classifiers
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/91989/OneDrive - Krishna Institute of Engineering & Technology/Desktop/project/trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier('C:/Users/91989/OneDrive - Krishna Institute of Engineering & Technology/Desktop/project/Face-Recognition-and-Liveliness-Detection/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:/Users/91989/OneDrive - Krishna Institute of Engineering & Technology/Desktop/project/Face-Recognition-and-Liveliness-Detection/Cascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('C:/Users/91989/OneDrive - Krishna Institute of Engineering & Technology/Desktop/project/Face-Recognition-and-Liveliness-Detection/Cascades/haarcascade_smile.xml')

# Font and names
font = cv2.FONT_HERSHEY_SIMPLEX
names = ['Anushka', 'Ghanu','Aman']  # id = 0 -> Anushka, id = 1 -> Ghanu

# Camera settings
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

blink_counter = 0
blink_threshold = 7
blink_detected = False
Liveliness = False

# Create dataset directory if not exists
dataset_dir = 'dataset'
os.makedirs(dataset_dir, exist_ok=True)

def show_alert(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Liveliness Confirmed", message)
    root.destroy()

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        id, confidence = recognizer.predict(roi_gray)

        if confidence < 50:
            id_name = names[id]
            confidence_text = "  {0}%".format(round(100 - confidence))
            smile_detected = False
            blink = False

            if Liveliness:
                cv2.putText(img, "Smile Detected, Liveliness Confirmed", (5, 15), font, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Smile for Liveliness Detection", (5, 25), font, 1, (255, 0, 0), 2)

                smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15, minSize=(25, 25))
                for (sx, sy, sw, sh) in smiles:
                    smile_detected = True
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                    cv2.putText(img, "Blink for Liveliness Detection", (5, 35), font, 1, (0, 165, 255), 2)

                    eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5, minSize=(5, 5))
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                        _, eye_threshold = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)
                        eye_area = eye_threshold.size
                        eye_white_area = cv2.countNonZero(eye_threshold)
                        eye_ratio = eye_white_area / eye_area

                        if eye_ratio < 0.25 and smile_detected:
                            blink_counter += 1
                            blink_detected = True
                            cv2.putText(roi_color, "Blink Detected", (ex, ey - 10), font, 0.5, (0, 165, 255), 2)
                        else:
                            blink_detected = False

                    if not blink_detected:
                        blink_counter = 0

                    if blink_counter >= blink_threshold:
                        Liveliness = True
                        message = f"Liveliness Detected, User: {id_name}"

                        # Save captured image to dataset
                        img_name = f"{dataset_dir}/{id_name}_{cv2.getTickCount()}.jpg"
                        cv2.imwrite(img_name, img)

                        show_alert(message)
                        cam.release()
                        cv2.destroyAllWindows()
                        sys.exit()

        else:
            id_name = "unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))

        # Display name and confidence
        cv2.putText(img, str(id_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # ESC to exit
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
