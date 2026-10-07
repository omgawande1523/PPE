import cv2
import os
from ultralytics import YOLO

# Load model
model_path = "best.pt"
model = YOLO(model_path)

# Read image
image_path = r"D:\Personal\PPE kit Detection final\test_images\10.jpg"
test_image = cv2.imread(image_path)

# Colors for drawing boxes
COLORS = {
    "person": (0, 255, 0),
    "helmet": (0, 0, 255),
    "vest": (0, 165, 255),
    "gloves": (255, 0, 0),
    "boots": (128, 0, 128),
    "goggles": (255, 255, 0)
}

# Utility function
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2-x1) * max(0, y2-y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    return inter_area / box1_area if box1_area > 0 else 0

# Run prediction
results = model.predict(test_image)
class_names = model.names

# Draw bounding boxes
for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
    x1, y1, x2, y2 = map(int, box)
    class_name = class_names[int(cls)]
    color = COLORS.get(class_name, (255, 255, 255))
    cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(test_image, class_name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Create 'results' directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Define output path
output_path = os.path.join("results", os.path.basename(image_path))

# Save the output image
cv2.imwrite(output_path, test_image)
print(f"⚔️ Output image saved at: {output_path}")

# Display the image
cv2.imshow("PPE Detection", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
