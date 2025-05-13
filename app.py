from flask import Flask, render_template, Response, session, redirect, url_for, request
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load secret key from file (Handle FileNotFoundError)
secret_key_path = "C:/Users/91989/secret.pem"
if os.path.exists(secret_key_path):
    with open(secret_key_path, "rb") as f:
        app.secret_key = f.read()
else:
    app.secret_key = os.urandom(24)  # Generate a temporary secret key

# Load face recognition model
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('C:/Users/91989/OneDrive - Krishna Institute of Engineering & Technology/Desktop/project/trainer/trainer.yml')
except Exception as e:
    print(f"Error loading face recognition model: {e}")

# Load Haarcascade classifiers
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Define known user names (adjust based on training data)
names = ['Anushka', 'Ghanu', 'Aman']

# Define liveliness detection variables
blink_counter = 0
blink_threshold = 7
liveliness = False
login_attempt = False  # Track login status

def generate_frames():
    global blink_counter, liveliness, login_attempt
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)  # Set width
    camera.set(4, 480)  # Set height

    minW = 0.1 * camera.get(3)
    minH = 0.1 * camera.get(4)

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(int(minW), int(minH)))

        user_authenticated = False  # Reset authentication status

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            try:
                id, confidence = recognizer.predict(roi_gray)
            except:
                id, confidence = -1, 100  # Handle recognition error

            if confidence < 50:
                name = names[id] if id < len(names) else "Unknown"
                confidence_text = f"{name} ({round(100 - confidence)}%)"
                color = (0, 255, 0)  # Green for recognized faces

                if name == "Unknown":
                    cv2.putText(frame, "Not able to login", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    login_attempt = False
                    continue  # Skip liveliness check

                smile_detected = False
                blink_detected = False

                if liveliness:
                    cv2.putText(frame, "Liveliness Confirmed", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    user_authenticated = True
                else:
                    cv2.putText(frame, "Smile for Liveliness Detection", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15, minSize=(25, 25))

                    for (sx, sy, sw, sh) in smiles:
                        smile_detected = True
                        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

                    if smile_detected:
                        cv2.putText(frame, "Blink for Liveliness Detection", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5, minSize=(5, 5))

                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                            _, eye_threshold = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)
                            eye_height, eye_width = eye_threshold.shape[:2]
                            eye_area = eye_height * eye_width
                            eye_white_area = cv2.countNonZero(eye_threshold)
                            eye_ratio = eye_white_area / eye_area

                            if eye_ratio < 0.25:
                                blink_counter += 1
                                blink_detected = True
                                cv2.putText(roi_color, "Blink Detected", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                            else:
                                blink_detected = False

                        if not blink_detected:
                            blink_counter = 0

                        if blink_counter >= blink_threshold:
                            liveliness = True
                            user_authenticated = True  # User can now log in

            else:
                cv2.putText(frame, "Not able to login", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                color = (0, 0, 255)  # Red for unknown faces
                login_attempt = False
                continue  # Skip liveliness check

            cv2.putText(frame, confidence_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if user_authenticated:
            login_attempt = True  # Allow login

        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()  # Release camera after streaming


from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import openpyxl

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        country = request.form['country']
        book = request.form['book']

        df_path = os.path.join(os.getcwd(), 'user_data.xlsx')
        print("Saving data to:", df_path)

        if os.path.exists(df_path):
            df = pd.read_excel(df_path)
        else:
            df = pd.DataFrame(columns=['Name', 'Email', 'Phone', 'Country', 'Book'])

        new_row = {
            'Name': name,
            'Email': email,
            'Phone': phone,
            'Country': country,
            'Book': book
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        try:
            df.to_excel(df_path, index=False)
            print("Data saved successfully.")
        except Exception as e:
            print("Error saving Excel file:", e)

        return redirect(url_for('login'))  # Redirects to login after registration

    return render_template('register.html', success=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        phone = request.form['phone']

        df_path = os.path.join(os.getcwd(), 'user_data.xlsx')

        if os.path.exists(df_path):
            df = pd.read_excel(df_path)
            # Convert both values to string and strip for safe comparison
            df['Email'] = df['Email'].astype(str).str.strip()
            df['Phone'] = df['Phone'].astype(str).str.strip()

            # Filter by both email and phone
            user_data = df[(df['Email'] == email.strip()) & (df['Phone'] == phone.strip())]

            if not user_data.empty:
                session['email'] = email.strip()  # Save cleaned email to session
                return redirect(url_for('security'))  # Proceed to security question
            else:
                return render_template('login.html', error= "Incorrect email or phone number.")
        else:
            return render_template('login.html', error= "User data file not found.")

    return render_template('login.html')



@app.route('/security', methods=['GET', 'POST'])
def security():
    if 'email' not in session:
        return redirect(url_for('register'))

    if request.method == 'POST':
        user_answer = request.form['security_answer']
        email = session['email']
        df = pd.read_excel('user_data.xlsx')
        user_data = df[df['Email'] == email]

        if not user_data.empty and user_data.iloc[0]['Book'].strip().lower() == user_answer.strip().lower():
            return redirect(url_for('face_login'))  # <-- Fixed here
        else:
            return "Security answer incorrect. <a href='/security'>Try again</a>"

    return render_template('security.html')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/face_login')
def face_login():
    return render_template('face_login.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/authenticate', methods=['POST'])
def authenticate():
    global login_attempt
    if login_attempt:
        session['user'] = names  # Set the logged-in user in the session
        return redirect(url_for('index'))  # This will redirect to /index
    return "Authentication failed! Not able to login."


@app.route('/index')
def index():
    if 'user' in session:
        return render_template('index.html', username=session['user'])  # optional: pass username
    return "Unauthorized access. Please login first.", 401

@app.route('/payment')
def payment_page():
    if 'user' in session:
        return render_template('paymentPage.html')
    return redirect(url_for('authenticate'))  # or a login page

@app.route('/success', methods=['POST'])  # 👈 Add methods=['POST']
def success():
    return render_template('successPage.html')
  # or just return a string/message

if __name__ == '__main__':
    app.run(debug=True)
