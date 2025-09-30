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

# Initialize MediaPipe FaceMesh globally to avoid repeated instantiation
mp_face_mesh = mp.solutions.face_mesh
global_face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Eye landmark indices for left and right eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Cache keys for state management
LIVENESS_STATE_KEY = "liveness_state"

def get_liveness_state():
    """Retrieve or initialize liveness state from cache."""
    state = cache.get(LIVENESS_STATE_KEY, {
        "last_check": timezone.now() - timedelta(seconds=11),
        "blink_count": 0,
        "recent_poses": []
    })
    return state

def update_liveness_state(state):
    """Update liveness state in cache with a TTL of 1 minute."""
    cache.set(LIVENESS_STATE_KEY, state, timeout=60)

def base64_to_image(b64str):
    """
    Convert base64 string to OpenCV image (RGB).
    Args:
        b64str (str): Base64-encoded image string.
    Returns:
        np.ndarray: Decoded image in RGB format, or None if invalid.
    """
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
    """
    Compute Eye Aspect Ratio (EAR) from Mediapipe landmarks.
    Args:
        landmarks: List of Mediapipe landmarks.
        eye_indices (list): Indices of eye landmarks.
    Returns:
        float: EAR value for the eye.
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    A = np.hypot(p2.x - p6.x, p2.y - p6.y)
    B = np.hypot(p3.x - p5.x, p3.y - p5.y)
    C = np.hypot(p1.x - p4.x, p1.y - p4.y)
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def estimate_head_pose(landmarks, img_w, img_h):
    """
    Estimate head pose (pitch, yaw, roll) in degrees using 2D-3D correspondences.
    Args:
        landmarks: List of Mediapipe landmarks.
        img_w (int): Image width.
        img_h (int): Image height.
    Returns:
        tuple: (pitch, yaw, roll) in degrees, or (0,0,0) if estimation fails.
    """
    try:
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),   # Nose tip
            (landmarks[152].x * img_w, landmarks[152].y * img_h),  # Chin
            (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Left eye corner
            (landmarks[263].x * img_w, landmarks[263].y * img_h),  # Right eye corner
            (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left mouth
            (landmarks[291].x * img_w, landmarks[291].y * img_h)   # Right mouth
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -63.6, -12.5),     # Chin
            (-43.3, 32.7, -26.0),    # Left eye left corner
            (43.3, 32.7, -26.0),     # Right eye right corner
            (-28.9, -28.9, -24.1),   # Left mouth
            (28.9, -28.9, -24.1)     # Right mouth
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
        sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        if sy < 1e-6:
            x = np.arctan2(-rmat[1,2], rmat[1,1])
            y = np.arctan2(-rmat[2,0], sy)
            z = 0
        else:
            x = np.arctan2(rmat[2,1], rmat[2,2])
            y = np.arctan2(-rmat[2,0], sy)
            z = np.arctan2(rmat[1,0], rmat[0,0])
        return (np.degrees(x), np.degrees(y), np.degrees(z))
    except Exception:
        return (0, 0, 0)

@csrf_exempt
def check_frame(request):
    """
    Endpoint to process video frames for liveness detection (blink and head movement).
    Expects JSON with 'image' (base64-encoded frame).
    Returns status every 10 seconds or 'Processing...' for intermediate frames.
    """
    if request.method != "POST":
        return HttpResponse("This endpoint accepts POST with JSON {'image': 'data:image/...;base64,...'}")

    try:
        data = json.loads(request.body)
        img_data = data.get("image", "")
        if not img_data:
            return JsonResponse({"error": "No image provided"}, status=400)

        # Convert base64 to image
        rgb = base64_to_image(img_data)
        h, w = rgb.shape[:2]

        # Resize if too large to optimize processing
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        # Get liveness state
        state = get_liveness_state()
        last_check = state["last_check"]
        blink_count = state["blink_count"]
        recent_poses = state["recent_poses"]

        # Process frame with FaceMesh
        results = global_face_mesh.process(rgb)
        now = timezone.now()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Calculate EAR for blink detection
            ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
            ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0
            if ear < 0.25:  # EAR threshold for blink
                blink_count += 1

            # Estimate head pose and store
            pose = estimate_head_pose(landmarks, rgb.shape[1], rgb.shape[0])
            recent_poses.append(pose)
            if len(recent_poses) > 6:
                recent_poses.pop(0)

        # Update state
        state.update({"blink_count": blink_count, "recent_poses": recent_poses})

        # Check liveness every 10 seconds
        if (now - last_check).seconds > 10:
            yaw_vals = [abs(p[1]) for p in recent_poses if p is not None]
            movement_ok = len(yaw_vals) >= 2 and (max(yaw_vals) - min(yaw_vals)) > 1.5

            if blink_count > 0 and movement_ok:
                status = "✅ Alive"
            elif blink_count > 0:
                status = "⚠️ Weak (blink-only) - further checks needed"
            else:
                status = "❌ Spoofing suspected"

            # Reset state and update cache
            state.update({"last_check": now, "blink_count": 0, "recent_poses": []})
            update_liveness_state(state)
            return JsonResponse({"status": status})

        # Update state for next frame
        update_liveness_state(state)
        return JsonResponse({"status": "Processing..."})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def upload_passport_and_verify(request):
    """
    Endpoint to verify identity by comparing passport and live images.
    Expects JSON with 'passport_image' and 'live_image' (base64-encoded).
    Returns match result and liveness hint.
    """
    if request.method != "POST":
        return HttpResponse("This endpoint accepts POST with JSON {'passport_image': '...', 'live_image': '...'}")

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
            liveness_hint = "face_detected (single-shot) - recommend multi-frame liveness check" if res.multi_face_landmarks else "no_face_detected"

        return JsonResponse({
            "distance": float(dist),
            "match": bool(match),
            "threshold": SIMILARITY_THRESHOLD,
            "liveness_hint": liveness_hint
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def liveness_test(request):
    """Render HTML page for liveness test."""
    return render(request, "test_liveness.html")