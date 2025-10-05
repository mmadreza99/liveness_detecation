import random
import base64
import json
import re
import numpy as np
import cv2
import mediapipe as mp
import face_recognition
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache

import textwrap
import os
import uuid

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
        # "instruction": "سر خود را به چپ و راست بچرخانید",
        "instruction": "head_turn",
        "type": "head_turn",
        "threshold": 25.0
    },
    # {
    #     # "instruction": "چند بار پلک بزنید",
    #     "instruction": "blink",
    #     "type": "blink",
    #     "threshold": 3
    # },
    # {
    #     "instruction": "سر خود را بالا و پایین تکان دهید",
    #     "type": "head_nod",
    #     "threshold": 10.0
    # },
    # {
    #     "instruction": "ابروهای خود را بالا ببرید",
    #     "type": "raise_eyebrows",
    #     "threshold": 0.05
    # }
]


def get_liveness_state():
    """Get or initialize liveness state (no attempt limit)"""
    state = cache.get(LIVENESS_STATE_KEY, {
        "last_check": timezone.now(),
        "blink_count": 0,
        "recent_poses": [],
        "current_challenge": None,
        "challenge_start": None,
        "challenge_data": {},
        "debug_images": []
    })

    # Select a new challenge if none active
    if state["current_challenge"] is None:
        state["current_challenge"] = random.choice(CHALLENGES)
        state["challenge_start"] = timezone.now()
        state["challenge_data"] = {"blinks": 0, "pose_changes": []}

    return state

def update_liveness_state(state):
    """Update liveness state in cache"""
    cache.set(LIVENESS_STATE_KEY, state, timeout=120)


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
    """Process video frames for optimized liveness detection"""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests allowed"}, status=405)

    try:
        data = json.loads(request.body)
        print('first touch')
        # Get instruction
        if "get_instruction" in data:
            state = get_liveness_state()
            return JsonResponse({"instruction": state["current_challenge"]["instruction"]})
        print('if "get_instruction" in data')

        # Convert base64 → image
        img_data = data.get("image", "")
        if not img_data:
            return JsonResponse({"error": "No image provided"}, status=400)
        print('img_data')

        rgb = base64_to_image(img_data)
        h, w = rgb.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        state = get_liveness_state()
        challenge = state["current_challenge"]
        challenge_data = state["challenge_data"]
        recent_poses = state["recent_poses"]

        results = global_face_mesh.process(rgb)
        now = timezone.now()

        if results.multi_face_landmarks:
            print(f'find face : ')
            landmarks = results.multi_face_landmarks[0].landmark
            if challenge["type"] == "blink":
                # Calculate blink
                ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
                       eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0
                print('ear', ear)

                if ear < 0.22:
                    challenge_data["blinks"] = challenge_data.get("blinks", 0) + 1
                    print('challenge_data["blinks"]', challenge_data["blinks"])
                elif ear > 0.25:
                    challenge_data["blinks_open"] = challenge_data.get("blinks_open", 0) + 1
                    print('challenge_data["blinks_open"]', challenge_data["blinks_open"])
                ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, feature_name="blink")
                state["debug_images"].append(ann)
            elif challenge["type"] == "head_turn":
                # Head pose estimation
                pose = estimate_head_pose(landmarks, rgb.shape[1], rgb.shape[0])
                recent_poses.append(pose)
                print('poses', pose)

                yaw = abs(pose[1])
                labels = [f"yaw:{yaw:.1f}"]
                ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, labels=labels,
                                                    feature_name="head_turn")
                state["debug_images"].append(ann)
            elif challenge["type"] == "raise_eyebrows":
                pose = estimate_head_pose(landmarks, rgb.shape[1], rgb.shape[0])
                yaw = abs(pose[1])
                labels = [f"yaw:{yaw:.1f}"]
                ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, labels=labels,
                                                    feature_name="head_turn")
                state["debug_images"].append(ann)
        else:
            print(f'not find face : ')

        # Every 10s check success
        if (now - state["last_check"]).seconds >= 10:
            success = False
            if challenge["type"] == "blink":
                success = (challenge_data.get("blinks", 0) >= challenge["threshold"]
                           and challenge_data.get("blinks_open", 0) >= challenge["threshold"])
                print('result blink: ', challenge_data.get("blinks", 0), challenge_data.get("blinks_open", 0))
            elif challenge["type"] == "head_turn":
                yaw_changes = [p[1] for p in recent_poses]
                if len(yaw_changes) >= 2:
                    success = (max(yaw_changes) - min(yaw_changes)) > challenge["threshold"]
                print(f'result head_turn: , len {len(yaw_changes)},threshold {max(yaw_changes) - min(yaw_changes)}')
            elif challenge["type"] == "raise_eyebrows":
                success = detect_raised_eyebrows(landmarks)

            if success:
                status = "✅ Alive - Challenge completed"
                state["current_challenge"] = random.choice(CHALLENGES)
                state["challenge_data"] = {"blinks": 0, "pose_changes": []}
                new_instruction = state["current_challenge"]["instruction"]
            else:
                status = "⚠️ Challenge not detected - Try again"
                new_instruction = challenge["instruction"]

            state.update({
                "last_check": now,
                "recent_poses": [],
                "challenge_data": challenge_data
            })
            update_liveness_state(state)

            # --- build and save a summary image with steps and "percentage" info ---
            try:
                # create lines from state and challenge_data
                lines = build_step_lines_from_state(state)

                # If we have some crude confidence/percentage we can calculate:
                # example: for blink challenge compute percent = min(100, blinks/threshold*100)
                perc_line = ""
                if challenge["type"] == "blink":
                    blinks = challenge_data.get("blinks", 0)
                    pct = min(100, int((blinks / max(1, challenge["threshold"])) * 100))
                    perc_line = f"Blink progress: {blinks}/{challenge['threshold']} ({pct}%)"
                elif challenge["type"] == "head_turn":
                    pcs = challenge_data.get("pose_changes", [])
                    if pcs:
                        # estimate percent by yaw delta vs threshold
                        delta = (max(pcs) - min(pcs))
                        pct = min(100, int((delta / max(1e-6, challenge["threshold"])) * 100))
                        perc_line = f"Yaw delta: {delta:.1f}deg ({pct}%)"
                if perc_line:
                    lines.append(perc_line)

                # pick an image to overlay (prefer last debug image if any)
                img_for_overlay = None
                if state.get("debug_images"):
                    # load image from path if available
                    last_dbg = state["debug_images"][-1]
                    imgp = last_dbg.get("path")
                    if imgp and os.path.exists(imgp):
                        img_bgr = cv2.imread(imgp)
                        if img_bgr is not None:
                            img_for_overlay = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # fallback to current frame rgb
                if img_for_overlay is None:
                    img_for_overlay = rgb

                summary = save_summary_image(img_for_overlay, lines)
                summary_path = summary["path"]
            except Exception as ex:
                # don't break main flow on failure to save debug image
                summary_path = None


            return JsonResponse({"status": status, "new_instruction": new_instruction})
        print('Processing . . . ')

        update_liveness_state(state)
        return JsonResponse({"status": "Processing..."})

    except Exception as e:
        print('error :', e)
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


@csrf_exempt
def reset_attempts(request):
    """Reset attempts counter (for testing)"""
    cache.delete(LIVENESS_STATE_KEY)
    cache.delete(ATTEMPTS_KEY)
    return JsonResponse({"status": "Attempts reset successfully"})



def save_summary_image(rgb_img, lines, filename=None):
    """
    Save a summary image with multiple text lines overlayed.
    Returns: {"path": ..., "b64": ...}
    """
    # English inline comments as requested
    # Convert to BGR for OpenCV drawing
    vis = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]

    # Choose font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w / 800.0, 1.2))
    thickness = 1 if w < 1000 else 2

    # Prepare background rectangle for text for readability
    # compute text block height
    line_height = int(24 * font_scale) + 6
    block_height = line_height * len(lines) + 12
    block_width = int(w * 0.9)
    x0 = int(w * 0.05)
    y0 = h - block_height - 10

    # Draw semi-transparent rectangle
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 6), (x0 + block_width + 6, y0 + block_height + 6), (0,0,0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # Draw each line of text
    y = y0 + 24
    for i, line in enumerate(lines):
        # wrap long lines
        wrapped = textwrap.wrap(line, width=80)
        for j, sub in enumerate(wrapped):
            text_pos = (x0 + 6, y + j * line_height)
            cv2.putText(vis, sub, text_pos, font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        y += len(wrapped) * line_height

    # unique filename
    fname = filename or f"summary_{uuid.uuid4().hex[:8]}.jpg"
    fpath = os.path.join(DEBUG_DIR, fname)

    # encode and save
    success, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise RuntimeError("Failed to encode summary image")
    with open(fpath, "wb") as f:
        f.write(buf.tobytes())

    b64str = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')
    return {"path": fpath, "b64": b64str}


def build_step_lines_from_state(state):
    """
    Build a list of human-readable lines describing debug events + confidence percents
    state: the liveness state dict (may contain 'debug_images' items saved earlier)
    """
    lines = []
    # Add timestamp
    lines.append(f"Time: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Attempts
    lines.append(f"Attempts remaining: {state.get('attempts_remaining', 'N/A')}")
    # Challenge
    ch = state.get("current_challenge") or {}
    lines.append(f"Challenge: {ch.get('instruction','-')} ({ch.get('type','-')})")

    # Add per-event info from debug_images
    dbg = state.get("debug_images", [])
    if dbg:
        lines.append("Events:")
        for d in dbg[-5:]:
            # try to include event and any yaw/conf/confidence keys
            ev = d.get("event", "event")
            info_parts = [ev]
            if "yaw" in d:
                info_parts.append(f"yaw={d['yaw']:.1f}")
            if "confidence" in d:
                info_parts.append(f"conf={d['confidence']}")
            lines.append(" - " + ", ".join(info_parts))
    else:
        lines.append("Events: none captured")

    # Add a short overall placeholder (more can be provided by callers)
    return lines
