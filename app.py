import os
import cv2
import torch
import time
import datetime
from flask import Flask, render_template, Response, jsonify, send_file, request
from ultralytics import YOLO

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

# =============================
# Load Model (Render Compatible)
# =============================
MODEL_PATH = "model/best.pt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ Model file not found in /model folder")

print("✅ Loading YOLO model...")
model = YOLO(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# Global Variables
# =============================
stats = {
    "fps": 0,
    "detections": 0,
    "alarm": False,
    "condition": "GOOD"
}

road_history = []
road_name = "Not Specified"
video_path = None


# =============================
# Routes
# =============================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path

    file = request.files['video']
    video_path = "uploaded_video.mp4"
    file.save(video_path)

    return jsonify({"status": "Video uploaded successfully"})


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    return jsonify(stats)


@app.route('/set_road', methods=['POST'])
def set_road():
    global road_name
    road_name = request.json.get("road_name", "Not Specified")
    return jsonify({"status": "Road name updated"})


@app.route('/export_pdf')
def export_pdf():

    file_path = "road_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)

    elements = []
    styles = getSampleStyleSheet()

    danger = sum(1 for r in road_history if r["condition"] == "DANGEROUS")
    moderate = sum(1 for r in road_history if r["condition"] == "MODERATE")

    if danger > 5:
        overall = "DANGEROUS"
    elif moderate > 5:
        overall = "MODERATE"
    else:
        overall = "GOOD"

    elements.append(Paragraph("Road Hazard Detection Report", styles['Title']))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(f"Road Name: {road_name}", styles['Normal']))
    elements.append(Paragraph(f"Overall Condition: {overall}", styles['Normal']))
    elements.append(Paragraph(f"Generated On: {datetime.datetime.now()}", styles['Normal']))
    elements.append(Spacer(1, 20))

    data = [["Time", "Detections", "Condition"]]

    for record in road_history:
        data.append([record["time"],
                     str(record["detections"]),
                     record["condition"]])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(table)
    doc.build(elements)

    return send_file(file_path, as_attachment=True)


# =============================
# Frame Generator
# =============================
def generate_frames():

    global video_path

    if video_path is None:
        return

    cap = cv2.VideoCapture(video_path)
    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 360))

        results = model(frame, device=device, conf=0.45, verbose=False)

        detection_count = 0
        hazard_detected = False

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < 0.45:
                    continue

                detection_count += 1
                hazard_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)

                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)

        if detection_count == 0:
            condition = "GOOD"
        elif detection_count <= 2:
            condition = "MODERATE"
        else:
            condition = "DANGEROUS"

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        road_history.append({
            "time": timestamp,
            "detections": detection_count,
            "condition": condition
        })

        if len(road_history) > 50:
            road_history.pop(0)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        stats["fps"] = int(fps)
        stats["detections"] = detection_count
        stats["alarm"] = hazard_detected
        stats["condition"] = condition

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes +
               b'\r\n')

    cap.release()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)