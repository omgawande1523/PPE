import cv2
from ultralytics import YOLO
from twilio.rest import Client
import os
import time
import csv
from datetime import datetime

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "best.pt"
VIDEO_SOURCE = 0  # webcam
CONF_THRESHOLD = 0.3
REQUIRED_PPE = ["helmet", "vest", "gloves", "boots", "goggles"]
ALERT_COOLDOWN = 10  # seconds between alerts
LOG_FILE = "ppe_log.csv"

# Twilio config (set these in your .env or environment variables)
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

# Colors for bounding boxes
COLORS = {
    "person": (0, 255, 0),
    "helmet": (0, 0, 255),
    "vest": (0, 165, 255),
    "gloves": (255, 0, 0),
    "boots": (128, 0, 128),
    "goggles": (255, 255, 0)
}

# ------------------------------
# Initialize
# ------------------------------
model = YOLO(MODEL_PATH)
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
cap = cv2.VideoCapture(VIDEO_SOURCE)
last_alert_time = 0

# Initialize CSV log
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Total People", "People Missing PPE Details"])

# ------------------------------
# Utility Functions
# ------------------------------
def iou(box1, box2):
    """Calculate Intersection over Union (for future person-PPE mapping logic)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2-x1) * max(0, y2-y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    return inter_area / box1_area if box1_area > 0 else 0

def send_alert(total_people, violations):
    """Send Twilio WhatsApp text alert."""
    if not violations:
        return

    message_text = (
        f"⚠️ PPE Violation Alert ⚠️\n\n"
        f"Total people detected: {total_people}\n"
        f"People missing PPE: {len(violations)}\n\n"
    )
    for idx, v in enumerate(violations, 1):
        message_text += f"👤 Person {idx} missing: {', '.join(v['missing'])}\n"

    twilio_client.messages.create(
        from_="whatsapp:+14155238886",
        to="whatsapp:+918262963069",  # replace with your number
        body=message_text
    )

    print("📨 Twilio alert sent successfully (text only).")

def log_violations(total_people, violations):
    """Log PPE violations into CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    violation_text = "; ".join([f"Person {i+1}: {', '.join(v['missing'])}" for i, v in enumerate(violations)])
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, total_people, violation_text])

# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Detect persons
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    persons = []
    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            cls_name = model.names[int(cls_id)]
            if cls_name == "person":
                persons.append(box.cpu().numpy())
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["person"], 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["person"], 2)

    # Step 2: PPE detection for each person
    violations = []
    for person_box in persons:
        x1, y1, x2, y2 = map(int, person_box)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        ppe_results = model.predict(person_crop, conf=CONF_THRESHOLD, verbose=False)
        present_ppe = set()

        for pr in ppe_results:
            for pbox, cls_id, conf in zip(pr.boxes.xyxy, pr.boxes.cls, pr.boxes.conf):
                cls_name = model.names[int(cls_id)]
                if cls_name in REQUIRED_PPE:
                    present_ppe.add(cls_name)
                    px1, py1, px2, py2 = map(int, pbox)
                    cv2.rectangle(frame, (x1+px1, y1+py1), (x1+px2, y1+py2), COLORS[cls_name], 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1+px1, y1+py1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls_name], 2)

        missing_ppe = list(set(REQUIRED_PPE) - present_ppe)
        if missing_ppe:
            violations.append({"missing": missing_ppe})

    # Step 3: Console display
    os.system('cls' if os.name == 'nt' else 'clear')
    print("================ PPE Detection ================")
    print(f"Total people detected: {len(persons)}")
    print(f"People missing PPE: {len(violations)}")
    for idx, v in enumerate(violations):
        print(f"Person {idx+1} missing: {', '.join(v['missing'])}")
    print("===============================================\n")

    # Step 4: Twilio alert (text only)
    current_time = time.time()
    if violations and (current_time - last_alert_time > ALERT_COOLDOWN):
        send_alert(len(persons), violations)
        last_alert_time = current_time

    # Step 5: Log to CSV
    if violations:
        log_violations(len(persons), violations)

    # Step 6: Display frame
    cv2.imshow("PPE Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
