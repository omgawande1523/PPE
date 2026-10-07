import cv2
from ultralytics import YOLO
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "best.pt"  # Your trained YOLO11s model
VIDEO_SOURCE = 0        # Webcam
CONF_THRESHOLD = 0.3
REQUIRED_PPE = ["helmet", "vest", "gloves", "boots", "goggles"]

# Colors for bounding boxes
COLORS = {
    "person": (0, 255, 0),
    "helmet": (0, 0, 255),
    "vest": (0, 165, 255),
    "gloves": (255, 0, 0),
    "boots": (128, 0, 128),
    "goggles": (255, 255, 0)
}

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# ------------------------------
# Utility Functions
# ------------------------------
def iou(box1, box2):
    """Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2-x1) * max(0, y2-y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    return inter_area / box1_area if box1_area > 0 else 0

# ------------------------------
# Webcam Loop
# ------------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Detect persons in the frame
    results = model.predict(frame, conf=CONF_THRESHOLD)
    persons = []
    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            cls_name = model.names[int(cls_id)]
            if cls_name == "person":
                persons.append(box.cpu().numpy())
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["person"], 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["person"], 2)

    # Step 2: For each person, crop and detect PPE
    violations = []
    for idx, person_box in enumerate(persons):
        x1, y1, x2, y2 = map(int, person_box)
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        # Detect PPE inside the person's bounding box
        ppe_results = model.predict(person_crop, conf=CONF_THRESHOLD)
        present_ppe = []
        for pr in ppe_results:
            for pbox, cls_id, conf in zip(pr.boxes.xyxy, pr.boxes.cls, pr.boxes.conf):
                cls_name = model.names[int(cls_id)]
                if cls_name in REQUIRED_PPE:
                    present_ppe.append(cls_name)
                    # Draw PPE boxes relative to full frame
                    px1, py1, px2, py2 = map(int, pbox.cpu().numpy())
                    cv2.rectangle(frame, (x1+px1, y1+py1), (x1+px2, y1+py2), COLORS[cls_name], 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1+px1, y1+py1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls_name], 2)

        # Determine missing PPE
        missing_ppe = list(set(REQUIRED_PPE) - set(present_ppe))
        if missing_ppe:
            violations.append(missing_ppe)

    # ------------------------------
    # Terminal Output
    # ------------------------------
    print("\033[H\033[J")  # Clear terminal
    print("================ PPE Detection ================")
    print(f"Total people detected: {len(persons)}")
    print(f"People missing PPE: {len(violations)}")
    for idx, missing in enumerate(violations):
        print(f"Person {idx+1} missing: {', '.join(missing)}")
    print("===============================================\n")

    # Show frame with bounding boxes
    cv2.imshow("Person-First PPE Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()