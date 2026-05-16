import cv2
import time
import threading
from collections import Counter
from ultralytics import YOLO
import pyttsx3
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Load Model once on startup
model = YOLO("yolov8s.pt")

# Global Variables
camera_active = False
cap = None

def speak_text(text):
    """Runs text-to-speech in a separate thread so the video doesn't freeze."""
    def run_tts():
        engine = pyttsx3.init()
        engine.setProperty('rate', 145)
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=run_tts).start()

def generate_sentence(region_objects):
    """Your original NLP sentence generator logic."""
    sentences = []
    for region, objects in region_objects.items():
        if len(objects) == 0:
            sentences.append(f"no objects detected in {region}")
        else:
            counts = Counter(objects)
            parts = []
            for obj, count in counts.items():
                if count == 1:
                    parts.append(f"one {obj}")
                else:
                    parts.append(f"{count} {obj}s")

            if len(parts) == 1:
                object_text = parts[0]
            else:
                object_text = ", ".join(parts[:-1]) + " and " + parts[-1]
            
            sentences.append(f"{object_text} detected in {region}")
            
    return ". ".join(sentences)

def generate_frames():
    """Captures frames, runs YOLO, and yields them to the web page."""
    global camera_active, cap
    last_spoken_time = time.time()
    speak_interval = 3

    while camera_active and cap is not None:
        start_time = time.time()
        success, frame = cap.read()
        
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        line1 = w // 3
        line2 = 2 * w // 3

        cv2.line(frame, (line1, 0), (line1, h), (0, 255, 0), 3)
        cv2.line(frame, (line2, 0), (line2, h), (0, 255, 0), 3)

        region_objects = {"Region 1": [], "Region 2": [], "Region 3": []}

        results = model(frame, conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                x_center = (x1 + x2) // 2

                if x_center < line1:
                    region = "Region 1"
                elif x_center < line2:
                    region = "Region 2"
                else:
                    region = "Region 3"

                region_objects[region].append(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Handle Speech
        current_time = time.time()
        if current_time - last_spoken_time >= speak_interval:
            sentence = generate_sentence(region_objects)
            print("\nSpeech Output:", sentence)
            speak_text(sentence)
            last_spoken_time = current_time

        # FPS Display
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode frame as JPEG to send to the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, cap
    if not camera_active:
        cap = cv2.VideoCapture(0)
        camera_active = True
    return jsonify({"status": "Camera started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, cap
    if camera_active:
        camera_active = False
        if cap is not None:
            cap.release()
            cap = None
    return jsonify({"status": "Camera stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)