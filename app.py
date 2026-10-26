import os
import re
import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

#Extracts File ID from a Google Drive URL and downloads the image.
def download_image_from_gdrive(drive_url, save_path):
  file_id_match = re.search(r'[-\w]{25,}', drive_url)
  if not file_id_match:
    return False

  file_id = file_id_match.group(0)
  download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

  try:
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
      with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
          if chunk:
            f.write(chunk)
      return True
  except Exception as e:
    print(f'Drive API Download Error: {e}')

  return False


def analyze_traffic_violations(detected_vehicles, total_count):
  violations = []

  if total_count >= 2:
    violations.append("- Failure to maintain safe following distance (Rear-end collision pattern)")
    violations.append("- Potential delayed reaction time / distracted driving")
    violations.append("- Intersection right-of-way violation or improper lane merging")
  elif total_count == 1:
    violations.append("- Single-vehicle loss of control")
    violations.append("- Driving too fast for road/weather conditions")
    violations.append("- Potential fixed-object or roadside hazard collision")
  else:
    violations.append("- Insufficient vehicle telemetry detected in frame")

  heavy_damages = [v for v in detected_vehicles if v['condition'] == 'Heavy Damage']
  moderate_damages = [v for v in detected_vehicles if v['condition'] == 'Moderate Damage']

  if len(heavy_damages) > 0:
    violations.append("- High-velocity impact indicator (Severe structural deformation)")
    violations.append("- Excessive speeding or failure to yield right-of-way")
  elif len(moderate_damages) > 0:
    violations.append("- Moderate kinetic transfer (Failure to brake adequately)")
  else:
    violations.append("- Low-impact or minor contact friction indicator")

  return violations


def assess_damage_opencv(vehicle_crop):
  if vehicle_crop.size == 0:
    return {'condition': 'Unknown', 'density': 0.0}

  vehicle_crop = cv2.resize(vehicle_crop, (200, 200))
  gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

  h, w = gray.shape
  mask = np.zeros_like(gray)
  margin_h = int(h * 0.15)
  margin_w = int(w * 0.15)
  mask[margin_h : h - margin_h, margin_w : w - margin_w] = 255

  masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
  blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
  edges = cv2.Canny(blurred, 80, 200)

  active_pixels = np.sum(mask > 0)
  edge_density = np.sum(edges > 0) / active_pixels if active_pixels > 0 else 0

  car_body = gray[margin_h : h - margin_h, margin_w : w - margin_w]
  bh, bw = car_body.shape
  if bh > 0 and bw > 1:
    left_half = car_body[:, : bw // 2]
    right_half = cv2.flip(car_body[:, bw // 2 :], 1)
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w > 0:
      left_half = left_half[:, :min_w]
      right_half = right_half[:, :min_w]
      diff = cv2.absdiff(left_half, right_half)
      asymmetry_score = np.mean(diff) / 255.0
    else:
      asymmetry_score = 0.0
  else:
    asymmetry_score = 0.0

  damage_score = (edge_density * 1.2) + (asymmetry_score * 0.6)

  if damage_score > 0.14:
    condition = 'Heavy Damage'
  elif damage_score > 0.08:
    condition = 'Moderate Damage'
  else:
    condition = 'Minor / Intact'

  return {'condition': condition, 'density': round(damage_score, 3)}


def process_accident_image(filepath, filename):
  image = cv2.imread(filepath)
  if image is None:
    return None, []

  H, W, _ = image.shape
  vehicle_classes = [2, 3, 5, 7]

  results = model(image, conf=0.4, iou=0.45)
  detected_vehicles = []
  raw_detections = []

  for r in results:
    for box in r.boxes:
      cls_id = int(box.cls[0])
      if cls_id in vehicle_classes:
        class_name = model.names[cls_id].capitalize()
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if (x2 - x1) < 40 or (y2 - y1) < 40:
          continue

        raw_detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name})

  raw_detections = sorted(raw_detections, key=lambda d: d['x1'])

  vehicle_count = 0
  for det in raw_detections:
    vehicle_count += 1
    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
    class_name = det['class']

    vehicle_crop = image[y1:y2, x1:x2]
    damage_info = assess_damage_opencv(vehicle_crop)

    vehicle_data = {
        'id': vehicle_count,
        'type': class_name,
        'condition': damage_info['condition'],
        'density': damage_info['density'],
    }
    detected_vehicles.append(vehicle_data)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    label = f'Vehicle {vehicle_count}: {class_name}'
    cv2.putText(
        image, label, (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )

  annotated_filename = 'analyzed_' + filename
  annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
  cv2.imwrite(annotated_path, image)

  return annotated_filename, detected_vehicles


@app.route('/')
def index():
  return render_template('index.html')


# Unified route handling both local file upload and Google Drive link pasting
@app.route('/analyze', methods=['POST'])
def analyze():
  filename = None
  save_path = None

  uploaded_file = request.files.get('photo')
  drive_url = request.form.get('gdrive_url')

  if uploaded_file and uploaded_file.filename != '':
    filename = uploaded_file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(save_path)

  elif drive_url and drive_url.strip() != '':
    filename = 'gdrive_fetched_scene.jpg'
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    success = download_image_from_gdrive(drive_url, save_path)
    if not success:
      return (
          'Error: Could not download image from the provided Google Drive link. '
          'Make sure link permissions are set to "Anyone with the link can view". '
          '<br><br><a href="/">Go Back</a>'
      )
  else:
    return redirect(url_for('index'))

  annotated_filename, vehicles = process_accident_image(save_path, filename)
  violations = analyze_traffic_violations(vehicles, len(vehicles))

  return render_template(
      'result.html',
      image_url=url_for('static_uploads', filename=annotated_filename),
      vehicles=vehicles,
      total=len(vehicles),
      violations=violations,
  )


@app.route('/uploads/<filename>')
def static_uploads(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
  app.run(debug=True, port=5000)