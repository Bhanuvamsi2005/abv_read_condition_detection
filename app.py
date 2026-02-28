import os
import cv2
import torch
import datetime
import numpy as np
from flask import Flask, render_template, jsonify, send_file, request, Response
from ultralytics import YOLO
import logging

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =============================
# CONFIGURATION
# =============================
MODEL_PATH = "model/best.pt"
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# =============================
# MODEL LOADING WITH ERROR HANDLING
# =============================
def load_model():
    """Load YOLO model with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        logger.info("🔄 Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        logger.info(f"✅ Model loaded successfully on {device}")
        logger.info(f"📦 Model classes: {model.names}")
        
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

# Load model at startup
model = load_model()

# =============================
# GLOBALS
# =============================
stats = {
    "detections": 0,
    "alarm": False,
    "condition": "GOOD",
    "processing": False
}

road_history = []
road_name = "Not Specified"

# Class-specific thresholds (from your detect.py)
CLASS_THRESHOLDS = {
    0: 0.40,  # pothole
    1: 0.40   # crack
}

COLORS = {
    0: (0, 0, 255),      # pothole - red
    1: (255, 0, 0)       # crack - blue
}

# =============================
# HELPER FUNCTIONS
# =============================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_detections(frame, results):
    """Draw detections on frame (adapted from detect.py)"""
    hazard_detected = False
    detection_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Apply class-specific threshold
            threshold = CLASS_THRESHOLDS.get(cls, 0.35)
            if conf < threshold:
                continue

            hazard_detected = True
            detection_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[cls]} {conf:.2f}"

            color = COLORS.get(cls, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2)

    return hazard_detected, detection_count

def process_video(input_path, output_path):
    """Process video and return statistics"""
    global stats, road_history
    
    road_history.clear()
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise Exception("Failed to open video")
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # Ensure fps is at least 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # If dimensions are 0, set default values
    if width == 0 or height == 0:
        width, height = 640, 480
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = 0
    total_detections = 0
    detection_history = []  # For smoothing
    history_size = 5
    
    logger.info("🚀 Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        
        # Run inference
        results = model(frame, device=model.device, conf=0.20, verbose=False)
        
        # Draw detections
        hazard_detected, frame_detections = draw_detections(frame, results)
        total_detections += frame_detections
        
        # Detection smoothing (from detect.py)
        detection_history.append(hazard_detected)
        if len(detection_history) > history_size:
            detection_history.pop(0)
        
        smoothed_hazard = sum(detection_history) >= 3
        
        # Determine condition
        if frame_detections == 0:
            condition = "GOOD"
        elif frame_detections <= 2:
            condition = "MODERATE"
        else:
            condition = "DANGEROUS"
        
        # Update stats
        stats["detections"] = frame_detections
        stats["alarm"] = smoothed_hazard
        stats["condition"] = condition
        
        # Add overlay panel (from detect.py)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        status = "HAZARD DETECTED" if smoothed_hazard else "NORMAL"
        status_color = (0, 0, 255) if smoothed_hazard else (0, 255, 0)
        
        cv2.putText(frame, f"Status: {status}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(frame, f"Detections: {frame_detections}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Condition: {condition}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add to history
        road_history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "detections": frame_detections,
            "condition": condition,
            "hazard": smoothed_hazard
        })
        
        out.write(frame)
        
        # Log progress every 100 frames
        if total_frames % 100 == 0:
            logger.info(f"Processed {total_frames} frames...")
    
    cap.release()
    out.release()
    
    logger.info(f"✅ Processing finished: {total_frames} frames, {total_detections} detections")
    
    return {
        "status": "success",
        "frames": total_frames,
        "detections": total_detections
    }

# =============================
# ROUTES
# =============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        stats["processing"] = True
        
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Generate unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = os.path.join(UPLOAD_FOLDER, f"input_{timestamp}.mp4")
        output_path = os.path.join(PROCESSED_FOLDER, f"output_{timestamp}.mp4")
        
        # Save uploaded file
        file.save(input_path)
        logger.info(f"📥 Video uploaded: {input_path}")
        
        # Process video
        result = process_video(input_path, output_path)
        
        # Clean up input file
        os.remove(input_path)
        
        stats["processing"] = False
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        stats["processing"] = False
        return jsonify({"error": str(e)}), 500

@app.route('/download_video')
def download_video():
    """Download the most recently processed video"""
    try:
        # Get the most recent processed video
        processed_files = sorted([f for f in os.listdir(PROCESSED_FOLDER) if f.startswith('output_')])
        
        if not processed_files:
            return "No processed video found", 404
        
        latest_video = os.path.join(PROCESSED_FOLDER, processed_files[-1])
        
        if not os.path.exists(latest_video):
            return "Processed video not found", 404
        
        return send_file(latest_video, as_attachment=True, download_name="processed_video.mp4")
    
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return str(e), 500

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    return jsonify(stats)

@app.route('/export_pdf')
def export_pdf():
    """Export detection history as PDF"""
    try:
        if len(road_history) == 0:
            return "No detection data available", 404
        
        # Generate PDF with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(PROCESSED_FOLDER, f"report_{timestamp}.pdf")
        
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        elements.append(Paragraph("Road Hazard Detection Report", styles['Title']))
        elements.append(Spacer(1, 15))
        
        # Metadata
        elements.append(Paragraph(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Paragraph(f"Road Name: {road_name}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Summary
        total_detections = sum(r['detections'] for r in road_history)
        hazard_frames = sum(1 for r in road_history if r.get('hazard', False))
        
        elements.append(Paragraph(f"Total Frames Analyzed: {len(road_history)}", styles['Normal']))
        elements.append(Paragraph(f"Total Detections: {total_detections}", styles['Normal']))
        elements.append(Paragraph(f"Frames with Hazards: {hazard_frames}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Data table
        data = [["Time", "Detections", "Condition", "Hazard"]]
        
        for record in road_history[-100:]:  # Last 100 records to avoid huge PDF
            data.append([
                record["time"],
                str(record["detections"]),
                record["condition"],
                "Yes" if record.get('hazard', False) else "No"
            ])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        return send_file(file_path, as_attachment=True, download_name="road_hazard_report.pdf")
    
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return str(e), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear detection history"""
    global road_history
    road_history.clear()
    return jsonify({"status": "success"})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": model.device.type if model else None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
