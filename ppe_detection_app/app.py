from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import base64
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from deduplication import ViolationDeduplicator, ViolationState
from deduplication.config import ALERT_COOLDOWN_SECONDS, ENABLE_DEDUPLICATION, MAX_VIOLATION_HISTORY

# Twilio Alerts (Phase 1)
try:
    from alerts import TwilioAlertService
    import twilio_config
    from twilio_config import ENABLE_TWILIO_ALERTS, validate_config
    TWILIO_AVAILABLE = True
except ImportError as e:
    TWILIO_AVAILABLE = False
    print(f"Twilio alerts not available: {e}")
except Exception as e:
    TWILIO_AVAILABLE = False
    print(f"Twilio configuration error: {e}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLO model
try:
    model = YOLO(r"D:\Projects\PPE kit Detection final\best.pt")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize camera
def init_camera():
    """Initialize camera with error handling"""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Warning: Could not open camera. Trying to reinitialize...")
        cam.release()
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Failed to open camera after retry")
            return None
    # Set camera properties for better performance
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    print("Camera initialized successfully")
    return cam

camera = init_camera()

violations = []
detection_history = []

# Initialize deduplicator if enabled
if ENABLE_DEDUPLICATION:
    deduplicator = ViolationDeduplicator(
        cooldown_seconds=ALERT_COOLDOWN_SECONDS,
        max_active_violations=MAX_VIOLATION_HISTORY
    )
    print(f"Deduplication enabled with {ALERT_COOLDOWN_SECONDS}s cooldown period")
else:
    deduplicator = None
    print("Deduplication disabled")

# Initialize Twilio alert service if enabled
twilio_service = None
if TWILIO_AVAILABLE and ENABLE_TWILIO_ALERTS:
    try:
        # Validate configuration first
        is_valid, error_msg = validate_config()
        if is_valid:
            twilio_service = TwilioAlertService()
            print("Twilio alerts enabled")
            # Test connection
            test_result = twilio_service.test_connection()
            if test_result['success']:
                print(f"Twilio connection verified - Account: {test_result.get('account_name', 'N/A')}")
            else:
                print(f"Warning: Twilio connection test failed: {test_result.get('error', 'Unknown error')}")
        else:
            print(f"Twilio alerts disabled: {error_msg}")
    except Exception as e:
        print(f"Failed to initialize Twilio alerts: {e}")
        twilio_service = None
else:
    if not TWILIO_AVAILABLE:
        print("Twilio alerts not available (SDK not installed or import error)")
    elif not ENABLE_TWILIO_ALERTS:
        print("Twilio alerts disabled (ENABLE_TWILIO_ALERTS=false)")

# Expected PPE classes (adjust based on your model's classes)
# Update this list to match the class names in your best.pt model
# You can check model.names after loading to see available classes
PPE_CLASSES = ['helmet', 'vest', 'gloves', 'safety_glasses', 'boots']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_violations(results, frame):
    """Extract violations from YOLO detection results"""
    detected_classes = set()
    missing_ppe = []
    has_detections = False
    
    if results and len(results) > 0:
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                has_detections = True
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls] if model else f"class_{cls}"
                    
                    if conf > 0.5:  # Confidence threshold
                        detected_classes.add(class_name.lower())
        
        # Only check for missing PPE if we have detections (activity in frame)
        if has_detections:
            # Check for missing PPE - only flag if we expect it but don't see it
            # Adjust PPE_CLASSES based on your model's actual class names
            for ppe in PPE_CLASSES:
                # Normalize class name for comparison
                ppe_normalized = ppe.lower().replace('_', ' ').replace('-', ' ')
                found = False
                for detected in detected_classes:
                    detected_normalized = detected.lower().replace('_', ' ').replace('-', ' ')
                    if ppe_normalized in detected_normalized or detected_normalized in ppe_normalized:
                        found = True
                        break
                if not found:
                    missing_ppe.append(ppe)
    
    return detected_classes, missing_ppe

@app.route('/')
def index():
    return render_template('safety kit detection/safety kit detection/new_index.html')

def gen_frames():
    global violations, detection_history, camera
    
    # Track seen violations in this frame cycle for resolution tracking (Phase 2)
    seen_signatures = set()
    frame_count = 0
    
    # Reinitialize camera if needed
    if camera is None or not camera.isOpened():
        print("Reinitializing camera...")
        camera = init_camera()
        if camera is None:
            print("Error: Camera not available")
            return
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            # Try to reinitialize camera
            camera.release()
            camera = init_camera()
            if camera is None:
                break
            continue

        try:
            if model is not None:
                # Run inference
                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()
                
                # Detect violations
                detected_classes, missing_ppe = detect_violations(results, frame)
                
                # Phase 2: Track seen violations
                seen_signatures.clear()
                
                # Update violations if there are missing PPE items
                if missing_ppe:
                    # Track signature for resolution checking
                    if deduplicator:
                        signature = deduplicator.create_signature(missing_ppe)
                        seen_signatures.add(signature)
                    
                    # Check if this is a duplicate violation (if deduplication enabled)
                    is_duplicate = False
                    if deduplicator:
                        is_duplicate = deduplicator.is_duplicate(missing_ppe)
                    
                    # Only create new violation entry if not duplicate
                    if not is_duplicate:
                        violation_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "missing": missing_ppe,
                            "detected": list(detected_classes),
                            "person_count": len(results[0].boxes) if results[0].boxes is not None else 0
                        }
                        
                        # Get violation info for Phase 2 (state information)
                        if deduplicator:
                            violation_info = deduplicator.get_violation_info(missing_ppe)
                            if violation_info:
                                violation_entry["state"] = violation_info.get('state', 'new')
                                violation_entry["alert_count"] = violation_info.get('alert_count', 1)
                            
                            deduplicator.register_violation(missing_ppe)
                        
                        # Send Twilio alert (only for new violations, respects deduplication)
                        if twilio_service:
                            try:
                                alert_result = twilio_service.send_violation_alert({
                                    'missing': missing_ppe,
                                    'timestamp': violation_entry['timestamp'],
                                    'detected': list(detected_classes),
                                    'person_count': violation_entry['person_count'],
                                    'location': 'Camera Feed'  # Can be customized
                                })
                                
                                if alert_result.get('success'):
                                    message_sid = alert_result.get('message_sid', 'N/A')
                                    status = alert_result.get('message_status', 'unknown')
                                    warnings = alert_result.get('warnings')
                                    
                                    print(f"✅ Twilio SMS sent - Violation: {', '.join(missing_ppe)}")
                                    print(f"   📱 To: {twilio_config.ALERT_RECIPIENT_PHONE}")
                                    print(f"   📞 From: {twilio_config.TWILIO_PHONE_NUMBER}")
                                    print(f"   🆔 Message SID: {message_sid}")
                                    print(f"   📊 Status: {status}")
                                    
                                    if warnings:
                                        for warning in warnings:
                                            print(f"   ⚠️  {warning}")
                                elif alert_result.get('skipped'):
                                    reason = alert_result.get('reason', 'Unknown')
                                    print(f"⏭️  Twilio alert skipped: {reason}")
                                else:
                                    error = alert_result.get('error', 'Unknown error')
                                    details = alert_result.get('details', '')
                                    print(f"❌ Twilio alert FAILED: {error}")
                                    if details:
                                        print(f"   Details: {details}")
                                    print(f"   📱 Attempted to: {twilio_config.ALERT_RECIPIENT_PHONE}")
                                    print(f"   📞 From: {twilio_config.TWILIO_PHONE_NUMBER}")
                                    print(f"   💡 Run 'python check_twilio_status.py' for troubleshooting")
                            except Exception as e:
                                print(f"❌ Exception sending Twilio alert: {e}")
                                logger.error(f"Error sending Twilio alert: {e}", exc_info=True)
                        
                        # Add to history (keep last 10)
                        detection_history.append({
                            "detected": list(detected_classes),
                            "missing": missing_ppe,
                            "timestamp": datetime.now().isoformat()
                        })
                        if len(detection_history) > 10:
                            detection_history.pop(0)
                        
                        violations = [violation_entry]
                    else:
                        # Duplicate violation - update timestamp of existing violation if it exists
                        # but don't create a new alert
                        if deduplicator:
                            deduplicator.update_violation_timestamp(missing_ppe)
                        # Keep existing violation in the list (don't replace it)
                else:
                    # No violations detected - clear violations list
                    violations = []
                
                # Phase 2: Mark unseen violations as resolved (check every 30 frames to avoid overhead)
                frame_count += 1
                if deduplicator and frame_count % 30 == 0:  # Check every ~1 second (at 30fps)
                    active_violations = deduplicator.get_active_violations()
                    for sig, violation_data in active_violations.items():
                        # If violation not seen in this cycle and it's active/expired (not already resolved)
                        if sig not in seen_signatures and violation_data['state'] != ViolationState.RESOLVED:
                            # Check if it's been a while since last seen (more than cooldown period)
                            time_since_last_seen = (datetime.now() - violation_data['last_seen']).total_seconds()
                            if time_since_last_seen > ALERT_COOLDOWN_SECONDS:
                                # Convert signature back to missing PPE list
                                missing_items = sig.split('|')
                                deduplicator.mark_violation_resolved(missing_items)
            else:
                annotated_frame = frame
                
        except Exception as e:
            print(f"Error in frame processing: {e}")
            annotated_frame = frame

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violations')
def get_violations():
    return jsonify({
        "violations": violations,
        "history": detection_history[-5:]  # Return last 5 detections
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image/video upload and run detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the image
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Could not read image file'}), 400
            
            # Run detection
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            results = model(image, verbose=False)
            
            # Process results
            detected_classes = set()
            missing_ppe = []
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        if conf > 0.5:
                            detected_classes.add(class_name.lower())
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                "class": class_name,
                                "confidence": round(conf, 2),
                                "bbox": [float(x1), float(y1), float(x2), float(y2)]
                            })
                
                # Check for missing PPE - only if we have detections
                if len(detected_classes) > 0:
                    for ppe in PPE_CLASSES:
                        ppe_normalized = ppe.lower().replace('_', ' ').replace('-', ' ')
                        found = False
                        for detected in detected_classes:
                            detected_normalized = detected.lower().replace('_', ' ').replace('-', ' ')
                            if ppe_normalized in detected_normalized or detected_normalized in ppe_normalized:
                                found = True
                                break
                        if not found:
                            missing_ppe.append(ppe)
                
                # Create annotated image
                annotated_image = result.plot()
                annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
                cv2.imwrite(annotated_path, annotated_image)
                
                # Convert to base64 for response
                _, buffer = cv2.imencode('.jpg', annotated_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return jsonify({
                    'success': True,
                    'detected': list(detected_classes),
                    'missing': missing_ppe,
                    'detections': detections,
                    'annotated_image': 'data:image/jpeg;base64,' + img_base64,
                    'has_violations': len(missing_ppe) > 0
                })
            else:
                return jsonify({
                    'success': True,
                    'detected': [],
                    'missing': PPE_CLASSES,
                    'detections': [],
                    'has_violations': True,
                    'message': 'No detections found'
                })
                
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/detection_status')
def detection_status():
    """Get current detection status"""
    global camera
    camera_status = False
    if camera is not None:
        camera_status = camera.isOpened()
        if camera_status:
            # Try to read a test frame
            ret, _ = camera.read()
            camera_status = ret
    
    status = {
        "model_loaded": model is not None,
        "camera_active": camera_status,
        "camera_available": camera is not None,
        "recent_violations": len(violations)
    }
    
    # Add deduplication stats if enabled
    if deduplicator:
        status["deduplication"] = {
            "enabled": True,
            "cooldown_seconds": ALERT_COOLDOWN_SECONDS,
            "stats": deduplicator.get_stats()
        }
    else:
        status["deduplication"] = {"enabled": False}
    
    # Add Twilio alert stats if enabled
    if twilio_service:
        status["twilio_alerts"] = twilio_service.get_stats()
    else:
        status["twilio_alerts"] = {
            "enabled": False,
            "available": TWILIO_AVAILABLE
        }
    
    return jsonify(status)

@app.route('/deduplication_stats')
def deduplication_stats():
    """Get deduplication statistics (Phase 2: Enhanced with state information)"""
    if not deduplicator:
        return jsonify({
            "enabled": False,
            "message": "Deduplication is disabled"
        })
    
    stats = deduplicator.get_stats()
    
    return jsonify({
        "enabled": True,
        "phase": 2,
        "config": {
            "cooldown_seconds": ALERT_COOLDOWN_SECONDS,
            "max_violations": MAX_VIOLATION_HISTORY
        },
        "stats": stats,
        "active_violations": len(deduplicator.get_active_violations())
    })

@app.route('/violations_by_state/<state>')
def violations_by_state(state):
    """Get violations by state (Phase 2 feature)
    
    States: new, active, expired, resolved
    """
    if not deduplicator:
        return jsonify({"error": "Deduplication is disabled"}), 400
    
    try:
        state_enum = ViolationState(state.lower())
    except ValueError:
        return jsonify({
            "error": f"Invalid state. Valid states: {[s.value for s in ViolationState]}"
        }), 400
    
    violations_by_state = deduplicator.get_violations_by_state(state_enum)
    
    # Convert to JSON-serializable format
    result = []
    for sig, violation_data in violations_by_state.items():
        violation_dict = violation_data.copy()
        violation_dict['signature'] = sig
        violation_dict['state'] = violation_dict['state'].value
        violation_dict['missing_ppe'] = sig.split('|')
        
        # Convert datetime to ISO strings
        for key in ['timestamp', 'first_seen', 'last_seen']:
            if key in violation_dict and isinstance(violation_dict[key], datetime):
                violation_dict[key] = violation_dict[key].isoformat()
        
        result.append(violation_dict)
    
    return jsonify({
        "state": state,
        "count": len(result),
        "violations": result
    })

@app.route('/twilio_stats')
def twilio_stats():
    """Get Twilio alert statistics"""
    if not twilio_service:
        return jsonify({
            "enabled": False,
            "available": TWILIO_AVAILABLE,
            "message": "Twilio alerts not enabled or not available"
        })
    
    stats = twilio_service.get_stats()
    return jsonify(stats)

@app.route('/twilio_test')
def twilio_test():
    """Test Twilio connection"""
    if not twilio_service:
        return jsonify({
            "success": False,
            "error": "Twilio service not initialized"
        }), 400
    
    result = twilio_service.test_connection()
    return jsonify(result)

@app.route('/twilio_send_test')
def twilio_send_test():
    """Send a test SMS alert"""
    if not twilio_service:
        return jsonify({
            "success": False,
            "error": "Twilio service not initialized"
        }), 400
    
    # Send a test violation alert
    test_violation = {
        'missing': ['helmet', 'vest'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'detected': ['gloves'],
        'person_count': 1,
        'location': 'Test Alert'
    }
    
    result = twilio_service.send_violation_alert(test_violation)
    
    return jsonify({
        "test_alert": True,
        "result": result,
        "config": {
            "twilio_phone": twilio_config.TWILIO_PHONE_NUMBER,
            "recipient_phone": twilio_config.ALERT_RECIPIENT_PHONE
        }
    })

@app.route('/violation_info')
def violation_info():
    """Get information about a specific violation (Phase 2 feature)
    
    Query params: missing (comma-separated list of missing PPE items)
    Example: /violation_info?missing=helmet,vest
    """
    if not deduplicator:
        return jsonify({"error": "Deduplication is disabled"}), 400
    
    missing_param = request.args.get('missing', '')
    if not missing_param:
        return jsonify({"error": "Missing 'missing' query parameter"}), 400
    
    missing_ppe = [item.strip() for item in missing_param.split(',')]
    info = deduplicator.get_violation_info(missing_ppe)
    
    if info is None:
        return jsonify({
            "found": False,
            "message": "Violation not found in tracking"
        })
    
    # Convert datetime to ISO strings
    for key in ['timestamp', 'first_seen', 'last_seen']:
        if key in info and isinstance(info[key], datetime):
            info[key] = info[key].isoformat()
    
    return jsonify({
        "found": True,
        "violation": info
    })

@app.route('/team_images/<filename>')
def team_images(filename):
    """Serve team images from templates folder"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'safety kit detection', 'safety kit detection')
    return send_from_directory(template_dir, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
