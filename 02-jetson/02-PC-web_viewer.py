import cv2
import zmq
import base64
import numpy as np
import time
from flask import Flask, render_template_string, Response
from datetime import datetime

# --- CONFIGURATION ---
JETSON_IP = "192.168.37.22"
ZMQ_PORT = 5555
THRESHOLD = 0.70

app = Flask(__name__)

PAGE = """
<html>
<head><title>JetsonPilot Dashboard</title></head>
<body style="background: #222; color: #eee; text-align: center; font-family: monospace;">
    <h1>JetsonPilot Real-Time View</h1>
    <div style="margin-bottom: 10px; color: #aaa;">Data source: Jetson Nano (ResNet18)</div>
    <img src="/video_feed" style="border: 2px solid #444; width: 672px; image-rendering: pixelated;"/>
</body>
</html>
"""

class VideoProcessor:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        # Low Conflate means: "Give me only the newest message, drop old ones"
        # Great to avoid the "250 FPS buffer read" issue you had
        self.socket.setsockopt(zmq.CONFLATE, 1)

        print(f"[Connect] Connecting to Jetson at {JETSON_IP}:{ZMQ_PORT}...")
        self.socket.connect(f"tcp://{JETSON_IP}:{ZMQ_PORT}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Recording setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"jetson_recording_{timestamp}.avi"
        self.writer = None
        print(f"[Record] Saving to {self.filename}")

    def process_stream(self):
        while True:
            try:
                # 1. Receive Data
                # Since we use CONFLATE, this always gives the absolute latest frame
                data = self.socket.recv_json()

                prob = data['prob_target']
                fps = data['jetson_fps'] # Retrieve the REAL FPS from Jetson
                jpg_original = base64.b64decode(data['image_b64'])

                # 2. Decode
                np_arr = np.frombuffer(jpg_original, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None: continue

                # 3. Draw Overlay
                if prob > THRESHOLD:
                    label = f"CIBLE:{prob:.0%}"
                    color = (0, 255, 0)
                    bg_color = (0, 50, 0)
                else:
                    label = f"NOCIBLE:{prob:.0%}"
                    color = (0, 0, 255)
                    bg_color = (0, 0, 50)

                # Top Bar
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 20), bg_color, -1)

                # Info Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Target Info
                cv2.putText(frame, label, (5, 15), font, 0.4, color, 1)

                # FPS Info (From Jetson)
                fps_text = f"INF:{fps:.1f} FPS"
                (w, _), _ = cv2.getTextSize(fps_text, font, 0.4, 1)
                cv2.putText(frame, fps_text, (frame.shape[1] - w - 5, 15), font, 0.4, (200, 200, 200), 1)

                # 4. Save to Disk
                if self.writer is None:
                    h, w_img = frame.shape[:2]
                    self.writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (w_img, h))
                self.writer.write(frame)

                # 5. Browser Stream
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            except Exception as e:
                # print(f"Error: {e}")
                pass

processor = VideoProcessor()

@app.route('/')
def index():
    return render_template_string(PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(processor.process_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
