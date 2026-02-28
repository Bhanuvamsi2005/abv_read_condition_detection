import os
import cv2
import torch
import datetime
from flask import Flask, render_template, jsonify, send_file, request
from ultralytics import YOLO

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

# =============================
# MODEL LOAD (SAFE VERSION)
# =============================

MODEL_PATH = "model/best.pt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ model/best.pt not found")

print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✅ Model loaded")
print("📦 Model classes:", model.names)

# =============================
# GLOBALS
# =============================

stats = {
    "detections": 0,
    "alarm": False,
    "condition": "GOOD"
}

road_history = []
road_name = "Not Specified"

UPLOAD_PATH = "uploaded_video.mp4"
OUTPUT_PATH = "processed_video.mp4"


# =============================
# ROUTES
# =============================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():

    global stats, road_history

    road_history.clear()

    file = request.files['video']
    file.save(UPLOAD_PATH)

    print("📥 Video uploaded")

    cap = cv2.VideoCapture(UPLOAD_PATH)

    if not cap.isOpened():
        return jsonify({"error": "Failed to open video"})

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    total_frames = 0
    total_detections = 0

    print("🚀 Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # 🔥 LOWER CONF FOR DEBUG
        results = model(frame, device=device, conf=0.20, verbose=False)

        frame_detections = 0
        hazard_detected = False

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                print("Detected:", model.names[cls], "Conf:", conf)

                if conf < 0.20:
                    continue

                frame_detections += 1
                hazard_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)

                cv2.putText(frame,
                            f"{model.names[cls]} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)

        total_detections += frame_detections

        # CONDITION
        if frame_detections == 0:
            condition = "GOOD"
        elif frame_detections <= 2:
            condition = "MODERATE"
        else:
            condition = "DANGEROUS"

        stats["detections"] = frame_detections
        stats["alarm"] = hazard_detected
        stats["condition"] = condition

        road_history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "detections": frame_detections,
            "condition": condition
        })

        out.write(frame)

    cap.release()
    out.release()

    print("✅ Processing finished")
    print("Total frames:", total_frames)
    print("Total detections:", total_detections)

    return jsonify({
        "status": "Processing complete",
        "frames": total_frames,
        "detections": total_detections
    })


@app.route('/download_video')
def download_video():
    if not os.path.exists(OUTPUT_PATH):
        return "No processed video found"
    return send_file(OUTPUT_PATH, as_attachment=True)


@app.route('/stats')
def get_stats():
    return jsonify(stats)


@app.route('/export_pdf')
def export_pdf():

    if len(road_history) == 0:
        return "No detection data available"

    file_path = "road_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)

    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Road Hazard Detection Report", styles['Title']))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(f"Road Name: {road_name}", styles['Normal']))
    elements.append(Spacer(1, 20))

    data = [["Time", "Detections", "Condition"]]

    for record in road_history:
        data.append([
            record["time"],
            str(record["detections"]),
            record["condition"]
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(table)
    doc.build(elements)

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
