import os
import uuid
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
        "threshold": 25.0
    },
    {
        "instruction": "چند بار پلک بزنید",
        "type": "blink",
        "threshold": 3
    },
    {
        "instruction": "ابروهای خود را بالا ببرید",
        "type": "raise_eyebrows",
        "threshold": 0.03
    }
]

class MyObject:
    def __init__(self, x, y ):
        self.x = x
        self.y = y


def get_liveness_state():
    """Get or initialize liveness state (no attempt limit)"""
    state = cache.get(LIVENESS_STATE_KEY, {
        "last_check": timezone.now(),
        "blink_count": 0,
        "recent_poses": [],
        "current_challenge": None,
        "challenge_start": None,
        "challenge_data": {},
        "debug_images": [],
        "result_raise_eyebrows": []
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
    """Detect a smile based on mouth landmarks"""
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
    """
    Detect if eyebrows are raised.

    Parameters:
        landmarks: list of mediapipe face landmarks

    Returns:
        bool: True if eyebrows are raised
    """
    # Indices of key points for left/right eyebrow and eyes
    left_eyebrow_indices = [55, 65, 52]   # sample points along the left eyebrow
    right_eyebrow_indices = [285, 295, 282]  # sample points along the right eyebrow
    left_eye_indices = [33, 133]   # approximate top/bottom of left eye
    right_eye_indices = [362, 263] # approximate top/bottom of right eye

    # Helper to calculate average y
    def avg_y(indices):
        return sum(landmarks[i].y for i in indices) / len(indices)

    # Helper to calculate average y
    def avg_x(indices):
        return sum(landmarks[i].x for i in indices) / len(indices)

     # Compute average Y positions
    left_eyebrow_y = avg_y(left_eyebrow_indices)
    right_eyebrow_y = avg_y(right_eyebrow_indices)
    left_eye_y = avg_y(left_eye_indices)
    right_eye_y = avg_y(right_eye_indices)

     # Compute average x positions
    left_eyebrow_x = avg_x(left_eyebrow_indices)
    right_eyebrow_x = avg_x(right_eyebrow_indices)
    left_eye_x = avg_x(left_eye_indices)
    right_eye_x = avg_x(right_eye_indices)

    new_landmark = (
        MyObject(left_eyebrow_x, left_eyebrow_y),
        MyObject(right_eyebrow_x, right_eyebrow_y),
        MyObject(left_eye_x, left_eye_y),
        MyObject(right_eye_x, right_eye_y),
    )

    # Normalized eyebrow height relative to eyes
    avg_distance = ((left_eye_y - left_eyebrow_y) + (right_eye_y - right_eyebrow_y)) / 2.0
    return avg_distance ,new_landmark


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
        result_raise_eyebrows = state["result_raise_eyebrows"]

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
                    ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, feature_name=f"blink_close_{ear}")
                elif ear >= 0.23:
                    challenge_data["blinks_open"] = challenge_data.get("blinks_open", 0) + 1
                    print('challenge_data["blinks_open"]', challenge_data["blinks_open"])
                    ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, feature_name=f"blink_open_{ear}")
                state["debug_images"].append(ann)
            elif challenge["type"] == "head_turn":
                # Head pose estimation
                pose = estimate_head_pose(landmarks, rgb.shape[1], rgb.shape[0])
                recent_poses.append(pose)
                print('poses', pose, 'len', len(recent_poses))

                yaw = abs(pose[1])
                labels = [f"yaw:{yaw:.1f}"]
                ann = annotate_landmarks_and_encode(rgb, landmarks=landmarks, labels=labels,
                                                    feature_name="head_turn")
                state["debug_images"].append(ann)
            elif challenge["type"] == "raise_eyebrows":
                avg_distance ,new_landmark = detect_raised_eyebrows(landmarks)
                result_raise_eyebrows.append(avg_distance)

                labels = [f"raise_eyebrows: {avg_distance}"]
                ann = annotate_landmarks_and_encode(rgb,  landmarks=new_landmark ,labels=labels,
                                                    feature_name=f"raise_eyebrows-{avg_distance}")
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
                min_val = min(result_raise_eyebrows)
                max_val = max(result_raise_eyebrows)
                diff = max_val - min_val

                success = diff > challenge["threshold"]
                print(f'result {success} diff {diff:.4f} min_val-{min_val:.4f}-max_val_{max_val:.4f}')

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
