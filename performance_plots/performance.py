from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_PATH = "best.pt"           # Your trained model
SAVE_DIR = "performance_plots"   # Directory to save visualizations
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# ---------------------------
# LOAD MODEL
# ---------------------------
model = YOLO(MODEL_PATH)

# Create directory if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# EVALUATE MODEL
# ---------------------------
print("\nEvaluating model performance... ⚙️")
metrics = model.val(conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, plots=True)

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
conf_matrix = metrics.confusion_matrix
labels = list(model.names.values())

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - YOLO Model", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.show()

# ---------------------------
# PRECISION-RECALL CURVE
# ---------------------------
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "precision_recall_curve.png"))
plt.show()

# ---------------------------
# CLASS-WISE F1 SCORES
# ---------------------------
class_f1 = metrics.results_dict['metrics/f1(B)']
plt.figure(figsize=(10, 6))
sns.barplot(x=labels, y=class_f1, palette="viridis")
plt.title("Class-wise F1 Scores")
plt.xlabel("Classes")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "classwise_f1_scores.png"))
plt.show()

# ---------------------------
# SUMMARY
# ---------------------------
print("\n📊 Performance Summary:")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {np.mean(precision):.4f}")
print(f"Recall: {np.mean(recall):.4f}")
print(f"Plots saved to: {SAVE_DIR}")
