"""Flask dashboard. All detection goes through ppe.infer.Pipeline.

Run from the repository root:
    python -m app.app                         # configs/infer_default.yaml, webcam from its 'source'
    python -m app.app --config configs/x.yaml --port 5000

Endpoints are the same as the old ppe_detection_app/app.py. In /violations
and /upload, "missing" now lists the items a person was seen violating
(a no_* detection); items with no evidence either way are listed under
"unknown" instead of being reported as violations.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from ppe.alerts import ViolationDeduplicator, ViolationState
from ppe.alerts import config as alert_config
from ppe.associate import UNKNOWN, VIOLATION
from ppe.infer import DEFAULT_CONFIG, REPO_ROOT, FrameResult, Pipeline, draw, load_config
from ppe.logging import RunLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = REPO_ROOT / "runs" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Set in main()
pipe: Pipeline | None = None
run_log: RunLogger | None = None
camera_source: int | str = 0
camera = None
deduplicator: ViolationDeduplicator | None = None
twilio_service = None

violations: list[dict] = []
detection_history: list[dict] = []


def summarise(fr: FrameResult) -> dict:
    """Frame-level view for the UI, built from the per-person decisions."""
    violated, unknown = set(), set()
    for p in fr.persons:
        violated.update(k for k, v in p.items.items() if v == VIOLATION)
        unknown.update(k for k, v in p.items.items() if v == UNKNOWN)
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "missing": sorted(violated),
        "unknown": sorted(unknown - violated),
        "detected": sorted({b.name for b in fr.boxes}),
        "person_count": len(fr.persons),
        "persons": [
            {"track_id": p.person.track_id, "status": p.status, "items": p.items} for p in fr.persons
        ],
    }


def init_camera():
    cam = cv2.VideoCapture(camera_source)
    if not cam.isOpened():
        logger.error("Could not open camera source %r", camera_source)
        return None
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam


def handle_violations(fr: FrameResult, summary: dict) -> None:
    """Dedup and alert once per violating person's item pattern."""
    global violations
    if not summary["missing"]:
        violations = []
        return
    for p in fr.persons:
        if p.status != VIOLATION:
            continue
        items = sorted(k for k, v in p.items.items() if v == VIOLATION)
        if deduplicator and deduplicator.is_duplicate(items):
            deduplicator.update_violation_timestamp(items)
            continue
        if deduplicator:
            deduplicator.register_violation(items)
        if twilio_service:
            result = twilio_service.send_violation_alert({
                "missing": items,
                "timestamp": summary["timestamp"],
                "detected": summary["detected"],
                "person_count": summary["person_count"],
                "location": alert_config.ALERT_LOCATION,
            })
            logger.info("Twilio alert for %s: %s", items, result)
        detection_history.append({"detected": summary["detected"], "missing": items,
                                  "timestamp": datetime.now().isoformat()})
        del detection_history[:-10]
    violations = [summary]


def resolve_stale() -> None:
    if not deduplicator:
        return
    for sig, data in deduplicator.get_active_violations().items():
        if data["state"] != ViolationState.RESOLVED:
            if (datetime.now() - data["last_seen"]).total_seconds() > alert_config.ALERT_COOLDOWN_SECONDS:
                deduplicator.mark_violation_resolved(sig.split("|"))


def gen_frames():
    global camera
    if camera is None or not camera.isOpened():
        camera = init_camera()
        if camera is None:
            return
    frame_idx = 0
    while True:
        ok, frame = camera.read()
        if not ok:
            logger.warning("Failed to read frame; reopening camera")
            camera.release()
            camera = init_camera()
            if camera is None:
                break
            continue
        fr = pipe.process(frame, frame_idx=frame_idx, track=True)
        run_log.frame(fr.frame_idx, f"camera:{camera_source}", frame.shape[:2], fr.boxes, fr.persons,
                      fr.orphans, fr.infer_ms)
        handle_violations(fr, summarise(fr))
        frame_idx += 1
        if frame_idx % 30 == 0:
            resolve_stale()
        ok, buf = cv2.imencode(".jpg", draw(fr), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/violations")
def get_violations():
    return jsonify({"violations": violations, "history": detection_history[-5:]})


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    if file.filename.rsplit(".", 1)[-1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Invalid file type"}), 400
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / secure_filename(file.filename)
    file.save(path)
    image = cv2.imread(str(path))
    if image is None:
        return jsonify({"error": "Could not read image file"}), 400
    fr = pipe.process(image, track=False)
    run_log.frame(0, f"upload:{path.name}", image.shape[:2], fr.boxes, fr.persons, fr.orphans, fr.infer_ms)
    summary = summarise(fr)
    annotated = draw(fr)
    cv2.imwrite(str(UPLOAD_DIR / f"annotated_{path.name}"), annotated)
    _, buf = cv2.imencode(".jpg", annotated)
    return jsonify({
        "success": True,
        **summary,
        "detections": [b.to_dict() for b in fr.boxes],
        "annotated_image": "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8"),
        "has_violations": bool(summary["missing"]),
    })


@app.route("/detection_status")
def detection_status():
    status = {
        "model_loaded": pipe is not None,
        "camera_active": bool(camera is not None and camera.isOpened()),
        "camera_available": camera is not None,
        "recent_violations": len(violations),
        "run_log": str(run_log.path.relative_to(REPO_ROOT)) if run_log else None,
        "deduplication": {"enabled": False},
        "twilio_alerts": twilio_service.get_stats() if twilio_service else {"enabled": False},
    }
    if deduplicator:
        status["deduplication"] = {"enabled": True, "cooldown_seconds": alert_config.ALERT_COOLDOWN_SECONDS,
                                   "stats": deduplicator.get_stats()}
    return jsonify(status)


@app.route("/deduplication_stats")
def deduplication_stats():
    if not deduplicator:
        return jsonify({"enabled": False, "message": "Deduplication is disabled"})
    return jsonify({
        "enabled": True,
        "config": {"cooldown_seconds": alert_config.ALERT_COOLDOWN_SECONDS,
                   "max_violations": alert_config.MAX_VIOLATION_HISTORY},
        "stats": deduplicator.get_stats(),
        "active_violations": len(deduplicator.get_active_violations()),
    })


@app.route("/violations_by_state/<state>")
def violations_by_state(state):
    if not deduplicator:
        return jsonify({"error": "Deduplication is disabled"}), 400
    try:
        state_enum = ViolationState(state.lower())
    except ValueError:
        return jsonify({"error": f"Invalid state. Valid states: {[s.value for s in ViolationState]}"}), 400
    out = []
    for sig, data in deduplicator.get_violations_by_state(state_enum).items():
        d = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in data.items()}
        d.update(signature=sig, state=data["state"].value, missing_ppe=sig.split("|"))
        out.append(d)
    return jsonify({"state": state, "count": len(out), "violations": out})


@app.route("/twilio_stats")
def twilio_stats():
    if not twilio_service:
        return jsonify({"enabled": False, "message": "Twilio alerts not enabled or not available"})
    return jsonify(twilio_service.get_stats())


@app.route("/twilio_test")
def twilio_test():
    if not twilio_service:
        return jsonify({"success": False, "error": "Twilio service not initialized"}), 400
    return jsonify(twilio_service.test_connection())


@app.route("/twilio_send_test")
def twilio_send_test():
    if not twilio_service:
        return jsonify({"success": False, "error": "Twilio service not initialized"}), 400
    result = twilio_service.send_violation_alert({
        "missing": ["helmet"], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detected": ["no_helmet"], "person_count": 1, "location": "Test Alert",
    })
    return jsonify({"test_alert": True, "result": result})


@app.route("/violation_info")
def violation_info():
    if not deduplicator:
        return jsonify({"error": "Deduplication is disabled"}), 400
    missing = [m.strip() for m in request.args.get("missing", "").split(",") if m.strip()]
    if not missing:
        return jsonify({"error": "Missing 'missing' query parameter"}), 400
    info = deduplicator.get_violation_info(missing)
    if info is None:
        return jsonify({"found": False, "message": "Violation not found in tracking"})
    info = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in info.items()}
    return jsonify({"found": True, "violation": info})


@app.route("/team_images/<path:filename>")
def team_images(filename):
    """Team photos are not committed; drop them into app/templates/team/ locally."""
    return send_from_directory(APP_DIR / "templates" / "team", filename)


def init_alerts() -> None:
    global deduplicator, twilio_service
    if alert_config.ENABLE_DEDUPLICATION:
        deduplicator = ViolationDeduplicator(cooldown_seconds=alert_config.ALERT_COOLDOWN_SECONDS,
                                             max_active_violations=alert_config.MAX_VIOLATION_HISTORY)
    if not alert_config.ENABLE_TWILIO_ALERTS:
        logger.info("Twilio alerts disabled (ENABLE_TWILIO_ALERTS is not true in .env)")
        return
    ok, msg = alert_config.validate_config()
    if not ok:
        logger.warning("Twilio alerts disabled: %s", msg)
        return
    from ppe.alerts import TwilioAlertService

    twilio_service = TwilioAlertService()
    logger.info("Twilio connection test: %s", twilio_service.test_connection())


def main(argv: list[str] | None = None) -> None:
    global pipe, run_log, camera_source
    ap = argparse.ArgumentParser(prog="python -m app.app")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    src = str(cfg.get("source", "0"))
    camera_source = int(src) if src.isdigit() else src
    pipe = Pipeline(cfg)
    init_alerts()
    meta = {"config": cfg.get("config_path"), "weights": Path(cfg["weights"]).name, "source": "flask_app",
            "conf": pipe.predict_kwargs["conf"], "iou": pipe.predict_kwargs["iou"],
            "imgsz": pipe.predict_kwargs["imgsz"], "required_items": pipe.required_items,
            "class_names": pipe.names}
    log_dir = REPO_ROOT / cfg.get("log_dir", "runs/logs")
    with RunLogger(log_dir, meta) as run_log:
        app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
