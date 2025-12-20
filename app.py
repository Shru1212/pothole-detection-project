import os
import csv
from datetime import datetime

from flask import Flask, render_template, request
import cv2
import numpy as np

# ---------------------------------------------
# Flask app initialization
# ---------------------------------------------
app = Flask(__name__)

# Ensure static folder exists for saving result images
os.makedirs("static", exist_ok=True)

REPORTS_FILE = "reports.csv"


# ---------------------------------------------
# Helper: initialise reports CSV file
# ---------------------------------------------
def init_reports_file():
    if not os.path.exists(REPORTS_FILE):
        with open(REPORTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "location", "severity", "potholes", "image_path"]
            )


def load_reports():
    """Read reports from CSV as list of dicts."""
    if not os.path.exists(REPORTS_FILE):
        return []
    with open(REPORTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


init_reports_file()


# ---------------------------------------------
# Pothole detection (handles small & big holes)
# ---------------------------------------------
def detect_potholes(image_bgr):
    """
    Classical computer-vision based pothole detector.
    - Focus on lower road region
    - Threshold + light morphology
    - Filters by area and shape so that both
      small and big potholes are detected.
    """
    # Resize to fixed size for stability
    img = cv2.resize(image_bgr, (640, 480))

    # 1) Focus on road area (bottom ~70%)
    h, w, _ = img.shape
    roi = img[int(h * 0.3):h, 0:w]  # region of interest
    roi_h, roi_w, _ = roi.shape
    roi_area = roi_h * roi_w

    # 2) Grayscale + blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Threshold – dark regions become white
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4) Light morphology to reduce noise but not merge everything
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) Find contours
    contours, _ = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pothole_count = 0
    annotated = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ---- TUNING PARAMETERS ----
        min_area = 200                # ignore tiny blobs
        max_area = 0.65 * roi_area    # allow big potholes, but not whole road

        if area < min_area or area > max_area:
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = w_box / float(h_box)

        # Allow a wide range so big wide potholes are not removed
        if aspect_ratio < 0.2 or aspect_ratio > 6.0:
            continue

        # Convert ROI coordinates to full-image coordinates
        y_full = int(h * 0.3) + y

        pothole_count += 1
        cv2.rectangle(
            annotated,
            (x, y_full),
            (x + w_box, y_full + h_box),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            "Pothole",
            (x, y_full - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated, pothole_count


# ---------------------------------------------
# Flask route
# ---------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    pothole_count = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        location = request.form.get("location", "").strip()
        severity = request.form.get("severity", "Medium")

        if not file or file.filename == "":
            error = "Please select an image first."
        else:
            # Convert uploaded file to OpenCV image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                error = "Invalid image file."
            else:
                # Run detection
                annotated, count = detect_potholes(img)

                # Save result image (overwrite each time)
                output_path = os.path.join("static", "output.jpg")
                cv2.imwrite(output_path, annotated)

                result_image = "static/output.jpg"
                pothole_count = count

                # --- Save report to CSV ---
                with open(REPORTS_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            location if location else "Not specified",
                            severity,
                            pothole_count,
                            result_image,
                        ]
                    )

    # Load reports for display
    reports = load_reports()

    return render_template(
        "index.html",
        result_image=result_image,
        pothole_count=pothole_count,
        error=error,
        reports=reports,
    )


# ---------------------------------------------
# Main entry point
# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)