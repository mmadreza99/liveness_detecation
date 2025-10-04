import random
import base64
import json
import re
import numpy as np
import cv2
import mediapipe as mp
import face_recognition
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from datetime import timedelta

import os
import uuid
import io
import base64

# Directory to save debug/annotated images (ensure it's writable)
DEBUG_DIR = "/tmp/liveness_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Initialize MediaPipe FaceMesh globally
mp_face_mesh = mp.solutions.face_mesh
global_face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Cache keys
LIVENESS_STATE_KEY = "liveness_state"
ATTEMPTS_KEY = "liveness_attempts"

# چالش‌های تشخیص زنده‌بودن
CHALLENGES = [
    {
        "instruction": "سر خود را به چپ و راست بچرخانید",
        "type": "head_turn",
        "threshold": 15.0
    },
    {
        "instruction": "چند بار پلک بزنید",
        "type": "blink",
        "threshold": 2
    },
    {
        "instruction": "سر خود را بالا و پایین تکان دهید",
        "type": "head_nod",
        "threshold": 10.0
    },
    {
        "instruction": "ابروهای خود را بالا ببرید",
        "type": "raise_eyebrows",
        "threshold": 0.05
    }
]


def get_liveness_state():
    """Get or initialize liveness state with attempts tracking"""
    state = cache.get(LIVENESS_STATE_KEY, {
        "last_check": timezone.now() - timedelta(seconds=11),
        "blink_count": 0,
        "recent_poses": [],
        "current_challenge": None,
        "challenge_start": None,
        "challenge_data": {},
        "attempts_remaining": 3
    })

    # Initialize attempts if not present
    if "attempts_remaining" not in state:
        state["attempts_remaining"] = cache.get(ATTEMPTS_KEY, 3)

    # Select new challenge if none is active
    if state["current_challenge"] is None:
        state["current_challenge"] = random.choice(CHALLENGES)
        state["challenge_start"] = timezone.now()
        state["challenge_data"] = {"blinks": 0, "pose_changes": []}

    return state


def update_liveness_state(state):
    """Update liveness state in cache"""
    cache.set(LIVENESS_STATE_KEY, state, timeout=120)  # 2 minutes TTL
    cache.set(ATTEMPTS_KEY, state["attempts_remaining"], timeout=120)


def base64_to_image(b64str):
    """Convert base64 string to OpenCV image (RGB)"""
    try:
        match = re.search(r'base64,(.*)', b64str)
        if not match:
            raise ValueError("Invalid base64 header")
        img_bytes = base64.b64decode(match.group(1))
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"Image decoding failed: {str(e)}")


def eye_aspect_ratio(landmarks, eye_indices):
    """Compute Eye Aspect Ratio (EAR) from Mediapipe landmarks"""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    A = np.hypot(p2.x - p6.x, p2.y - p6.y)
    B = np.hypot(p3.x - p5.x, p3.y - p5.y)
    C = np.hypot(p1.x - p4.x, p1.y - p4.y)
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def estimate_head_pose(landmarks, img_w, img_h):
    """Estimate head pose (pitch, yaw, roll) in degrees"""
    try:
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),  # Nose tip
            (landmarks[152].x * img_w, landmarks[152].y * img_h),  # Chin
            (landmarks[33].x * img_w, landmarks[33].y * img_h),  # Left eye corner
            (landmarks[263].x * img_w, landmarks[263].y * img_h),  # Right eye corner
            (landmarks[61].x * img_w, landmarks[61].y * img_h),  # Left mouth
            (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26.0),  # Left eye left corner
            (43.3, 32.7, -26.0),  # Right eye right corner
            (-28.9, -28.9, -24.1),  # Left mouth
            (28.9, -28.9, -24.1)  # Right mouth
        ])

        focal_length = img_w
        camera_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return (0, 0, 0)

        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        if sy < 1e-6:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0
        else:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        return (np.degrees(x), np.degrees(y), np.degrees(z))
    except Exception:
        return (0, 0, 0)


def detect_smile(landmarks):
    """Detect smile based on mouth landmarks"""
    left = landmarks[61]
    right = landmarks[291]
    top = landmarks[78]
    bottom = landmarks[308]
    width = np.hypot(right.x - left.x, right.y - left.y)
    height = np.hypot(bottom.x - top.x, bottom.y - top.y)
    smile_ratio = height / width if width != 0 else 0
    return smile_ratio < 0.2


def detect_tongue(landmarks):
    """Detect if tongue is out"""
    tongue_tip = landmarks[13]
    chin = landmarks[152]
    distance = abs(tongue_tip.y - chin.y)
    return distance > 0.1


def detect_raised_eyebrows(landmarks):
    """Detect raised eyebrows"""
    left_brow = landmarks[70].y
    right_brow = landmarks[300].y
    nose = landmarks[1].y
    avg_brow_height = (left_brow + right_brow) / 2
    return abs(avg_brow_height - nose) < 0.15


def annotate_landmarks_and_encode(rgb_img, landmarks=None, boxes=None, labels=None, feature_name=None):
    """
    Draw landmarks, boxes and labels on a copy of rgb_img and return base64 and filepath.
    - landmarks: list of mediapipe landmark objects (or None)
    - boxes: list of tuples (top,left,bottom,right) in pixel coords
    - labels: list of strings to write near boxes
    Returns: {"b64": ..., "path": "..."}
    """
    # copy and convert to BGR for OpenCV drawing
    vis = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_RGB2BGR)

    h, w = vis.shape[:2]
    # draw boxes
    if boxes:
        for i, box in enumerate(boxes):
            top, left, bottom, right = box
            # draw rectangle
            cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 0), 2)
            if labels and i < len(labels):
                cv2.putText(vis, labels[i], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # draw landmarks (simple circles)
    if landmarks:
        for lm in landmarks:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            cv2.circle(vis, (x_px, y_px), 1, (0, 0, 255), 2)

    # label feature name
    if feature_name:
        cv2.putText(vis, feature_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # encode to jpg
    success, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise RuntimeError("Failed to encode debug image")

    # create unique filename
    fname = f"{feature_name or 'feat'}_{uuid.uuid4().hex[:8]}.jpg"
    fpath = os.path.join(DEBUG_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(buf.tobytes())

    # also produce base64 (data URI)
    b64str = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')

    return {"b64": b64str, "path": fpath}


@csrf_exempt
def check_frame(request):
    """Endpoint to process video frames for liveness detection"""
    if request.method != "POST":
        return JsonResponse({"error": "This endpoint accepts POST requests"}, status=405)

    try:
        data = json.loads(request.body)

        if "get_instruction" in data:
            state = get_liveness_state()
            return JsonResponse({
                "instruction": state["current_challenge"]["instruction"],
                "attempts_remaining": state["attempts_remaining"]
            })

        img_data = data.get("image", "")
        if not img_data:
            return JsonResponse({"error": "No image provided"}, status=400)

        # Convert base64 to image
        rgb = base64_to_image(img_data)
        h, w = rgb.shape[:2]

        # Resize if too large for optimization
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        # Get liveness state
        state = get_liveness_state()
        last_check = state["last_check"]
        recent_poses = state["recent_poses"]
        challenge = state["current_challenge"]
        challenge_data = state["challenge_data"]
        attempts_remaining = state["attempts_remaining"]

        # Process frame with FaceMesh
        results = global_face_mesh.process(rgb)
        now = timezone.now()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Calculate EAR for blink detection
            ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
            ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0

            # Estimate head pose and store
            pose = estimate_head_pose(landmarks, rgb.shape[1], rgb.shape[0])
            recent_poses.append(pose)

            # ... after computing ear, pose, etc (inside results.multi_face_landmarks)
            boxes = []
            labels = []
            # optionally get face bounding box from landmarks (simple min/max)
            xs = [int(p.x * rgb.shape[1]) for p in landmarks]
            ys = [int(p.y * rgb.shape[0]) for p in landmarks]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            boxes.append((top, left, bottom, right))
            labels.append("face")

            # prepare a list to collect debug images for this frame
            debug_images = []

            if challenge["type"] == "blink":
                if ear < 0.25:
                    challenge_data["blinks"] = challenge_data.get("blinks", 0) + 1
                    # create annotated image for blink event
                    ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, boxes=boxes, labels=labels,
                                                        feature_name="blink")
                    debug_images.append({"event": "blink", **ann})
            elif challenge["type"] == "head_turn":
                yaw = abs(pose[1])
                challenge_data["pose_changes"] = challenge_data.get("pose_changes", [])
                challenge_data["pose_changes"].append(yaw)
                # save annotated image with yaw value as label
                labels[0] = f"yaw: {yaw:.1f}"
                ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, boxes=boxes, labels=labels,
                                                    feature_name="head_turn")
                debug_images.append({"event": "head_turn", "yaw": yaw, **ann})
            # ... similarly for other types
            elif challenge["type"] == "raise_eyebrows":
                if detect_raised_eyebrows(landmarks):
                    challenge_data["detections"] = challenge_data.get("detections", 0) + 1
                    ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, boxes=boxes, labels=labels,
                                                        feature_name="raise_eyebrows")
                    debug_images.append({"event": "raise_eyebrows", **ann})
            # store debug images into state (limit number to avoid bloat)
            existing_dbg = state.get("debug_images", [])
            existing_dbg.extend(debug_images)
            # limit to latest 5 images
            state["debug_images"] = existing_dbg[-5:]
            state["challenge_data"] = challenge_data

        # Update state
        state.update({"recent_poses": recent_poses})

        # Check liveness every 10 seconds
        if (now - last_check).seconds >= 10:
            success = False
            new_instruction = ""

            if challenge["type"] == "blink":
                success = challenge_data.get("blinks", 0) >= challenge["threshold"]
            elif challenge["type"] == "head_turn":
                yaw_changes = challenge_data.get("pose_changes", [])
                success = len(yaw_changes) >= 2 and (max(yaw_changes) - min(yaw_changes)) > challenge["threshold"]
            elif challenge["type"] == "smile":
                success = challenge_data.get("smiles_detected", 0) > 0
            elif challenge["type"] == "head_nod":
                pitch_changes = challenge_data.get("pose_changes", [])
                success = len(pitch_changes) >= 2 and (max(pitch_changes) - min(pitch_changes)) > challenge["threshold"]
            elif challenge["type"] in ["tongue_out", "raise_eyebrows", "close_left_eye"]:
                success = challenge_data.get("detections", 0) >= challenge["threshold"]

            if success:
                status = "✅ Alive - Challenge completed successfully"
                # Select new challenge
                state["current_challenge"] = random.choice(CHALLENGES)
                state["challenge_start"] = now
                state["challenge_data"] = {"blinks": 0, "pose_changes": []}
                new_instruction = state["current_challenge"]["instruction"]
            else:
                attempts_remaining -= 1
                status = f"❌ Challenge failed - {attempts_remaining} attempts remaining"
                if attempts_remaining <= 0:
                    status = "❌ Maximum attempts exceeded - Please try again later"

            # Reset state
            state.update({
                "last_check": now,
                "recent_poses": [],
                "attempts_remaining": attempts_remaining
            })
            update_liveness_state(state)

            return JsonResponse({
                "status": status,
                "new_instruction": new_instruction,
                "attempts_remaining": attempts_remaining
            })

        # Update state for next frame
        update_liveness_state(state)
        return JsonResponse({"status": "Processing...", "attempts_remaining": attempts_remaining})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def upload_passport_and_verify(request):
    """Endpoint to verify identity by comparing passport and live images"""
    if request.method != "POST":
        return JsonResponse({"error": "This endpoint accepts POST requests"}, status=405)

    try:
        data = json.loads(request.body)
        passport_b64 = data.get("passport_image")
        live_b64 = data.get("live_image")

        if not passport_b64 or not live_b64:
            return JsonResponse({"error": "Both passport_image and live_image are required"}, status=400)

        # Convert images
        passport_img = base64_to_image(passport_b64)
        live_img = base64_to_image(live_b64)

        # Face recognition
        passport_locations = face_recognition.face_locations(passport_img)
        live_locations = face_recognition.face_locations(live_img)

        if not passport_locations:
            return JsonResponse({"error": "No face found in passport image"}, status=400)
        if not live_locations:
            return JsonResponse({"error": "No face found in live image"}, status=400)

        passport_enc = face_recognition.face_encodings(passport_img, known_face_locations=passport_locations)[0]
        live_enc = face_recognition.face_encodings(live_img, known_face_locations=live_locations)[0]

        # Compute similarity
        dist = np.linalg.norm(passport_enc - live_enc)
        SIMILARITY_THRESHOLD = 0.55
        match = dist < SIMILARITY_THRESHOLD

        # Single-frame liveness check
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6
        ) as fm:
            res = fm.process(live_img)
            liveness_hint = "Face detected (single-shot) - recommend multi-frame liveness check" if res.multi_face_landmarks else "No face detected"

        return JsonResponse({
            "distance": float(dist),
            "match": bool(match),
            "threshold": SIMILARITY_THRESHOLD,
            "liveness_hint": liveness_hint,
            "confidence": f"{(1 - dist / SIMILARITY_THRESHOLD) * 100:.1f}%" if match else "0%"
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def liveness_test(request):
    """Render HTML page for liveness test"""
    return render(request, "test_liveness.html")


def reset_attempts(request):
    """Reset attempts counter (for testing)"""
    cache.delete(LIVENESS_STATE_KEY)
    cache.delete(ATTEMPTS_KEY)
    return JsonResponse({"status": "Attempts reset successfully"})