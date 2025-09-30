import base64
import json
import re

import cv2
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

import mediapipe as mp

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# اندیس چشم‌ها در FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

last_check = timezone.now()
blink_count = 0


import re
import io
import json
import base64
import numpy as np
import cv2
import face_recognition
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import mediapipe as mp
from datetime import timedelta

# -- Global reused mediapipe objects to avoid expensive re-creation per request --
mp_face_mesh = mp.solutions.face_mesh
# create one FaceMesh object for streaming-like usage; static_image_mode=False (video)
global_face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Eye indices (example for Mediapipe's 468-landmark mesh indices for left/right)
# Replace LEFT_EYE, RIGHT_EYE with your actual index lists if different.
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # example Mediapipe indices
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# persistent state
last_check = timezone.now() - timedelta(seconds=11)
blink_count = 0
recent_poses = []  # store last few pose values for movement check

# ---------------- helpers -----------------
def eye_aspect_ratio(landmarks, eye_indices):
    """Compute EAR from mediapipe landmarks (expects normalized x,y)."""
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    A = np.hypot(p2.x - p6.x, p2.y - p6.y)
    B = np.hypot(p3.x - p5.x, p3.y - p5.y)
    C = np.hypot(p1.x - p4.x, p1.y - p4.y)
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return float(ear)

def get_landmark_coords(landmarks, img_w, img_h):
    """Return list of (x,y) pixel coords from normalized landmarks."""
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    return pts

def estimate_head_pose(landmarks, img_w, img_h):
    """
    Simple head-pose estimation using 2D-3D correspondences (approx).
    Returns (pitch, yaw, roll) in degrees approximated.
    If fails, returns (0,0,0).
    """
    try:
        # use a few landmark indices: nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),   # example - adjust indices
            (landmarks[152].x * img_w, landmarks[152].y * img_h),
            (landmarks[33].x * img_w, landmarks[33].y * img_h),
            (landmarks[263].x * img_w, landmarks[263].y * img_h),
            (landmarks[61].x * img_w, landmarks[61].y * img_h),
            (landmarks[291].x * img_w, landmarks[291].y * img_h)
        ], dtype="double")

        # approximate 3D model points of a generic face (units are arbitrary)
        model_points = np.array([
            (0.0, 0.0, 0.0),         # nose tip
            (0.0, -63.6, -12.5),     # chin
            (-43.3, 32.7, -26.0),    # left eye left corner
            (43.3, 32.7, -26.0),     # right eye right corner
            (-28.9, -28.9, -24.1),   # left mouth
            (28.9, -28.9, -24.1)     # right mouth
        ])

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))  # assume no lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return (0,0,0)
        # convert rotation vector to Euler angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rmat[2,1], rmat[2,2])
            y = np.arctan2(-rmat[2,0], sy)
            z = np.arctan2(rmat[1,0], rmat[0,0])
        else:
            x = np.arctan2(-rmat[1,2], rmat[1,1])
            y = np.arctan2(-rmat[2,0], sy)
            z = 0
        # convert to degrees
        return (np.degrees(x), np.degrees(y), np.degrees(z))
    except Exception:
        return (0,0,0)

# ---------------- endpoints -----------------
@csrf_exempt
def check_frame(request):
    """Endpoint to receive base64 image frames, run liveness checks (blink + head movement),
       and optionally return a status every 10 seconds."""
    global last_check, blink_count, recent_poses, global_face_mesh

    if request.method != "POST":
        # return the page if GET
        return HttpResponse("This endpoint accepts POST with JSON {'image': 'data:image/...;base64,...'}")

    try:
        data = json.loads(request.body)
        img_data = data.get("image", "")
        if not img_data:
            return JsonResponse({"error": "no image provided"}, status=400)

        m = re.search(r'base64,(.*)', img_data)
        if not m:
            return JsonResponse({"error": "invalid base64 header"}, status=400)
        img_str = m.group(1)
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JsonResponse({"error": "could not decode image"}, status=400)

        # resize if too large to speed up processing (optional)
        h, w = frame.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process with the global face mesh (reuse)
        results = global_face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # EAR (eye aspect ratio)
            ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
            ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0

            # count blink when EAR drops below threshold
            EAR_THRESHOLD = 0.25
            if ear < EAR_THRESHOLD:
                blink_count += 1

            # head pose: store last few poses and measure variance (movement)
            pose = estimate_head_pose(landmarks, frame.shape[1], frame.shape[0])
            recent_poses.append(pose)
            if len(recent_poses) > 6:
                recent_poses.pop(0)

        # every 10 seconds return a consolidated status
        now = timezone.now()
        if (now - last_check).seconds > 10:
            # check movement (simple): calculate variance of yaw over recent poses
            yaw_vals = [abs(p[1]) for p in recent_poses if p is not None]
            movement_ok = False
            if len(yaw_vals) >= 2 and (max(yaw_vals) - min(yaw_vals)) > 1.5:
                movement_ok = True

            # final liveness decision: require at least one blink AND some head movement
            if blink_count > 0 and movement_ok:
                status = "✅ Alive"
            elif blink_count > 0 and not movement_ok:
                # blinking alone is weaker, might be replayed video; mark as suspicious
                status = "⚠️ Weak (blink-only) - further checks needed"
            else:
                status = "❌ Spoofing suspected"

            # reset counters
            last_check = now
            blink_count = 0
            recent_poses = []
            return JsonResponse({"status": status})

        return JsonResponse({"status": "Processing..."})
    except Exception as e:
        # return error for debugging; in production log instead
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def upload_passport_and_verify(request):
    """
    Accepts a passport image (base64) and a recent live frame (base64) OR we can store passport first,
    then compare later. Here we accept both in same request for simplicity:
    JSON: { "passport_image": "...base64...", "live_image": "...base64..." }
    """
    try:
        data = json.loads(request.body)
        passport_b64 = data.get("passport_image")
        live_b64 = data.get("live_image")
        if not passport_b64 or not live_b64:
            return JsonResponse({"error": "passport_image and live_image are required"}, status=400)

        def b64_to_rgb(b64str):
            m = re.search(r'base64,(.*)', b64str)
            if not m:
                raise ValueError("invalid base64 header")
            b = base64.b64decode(m.group(1))
            arr = np.frombuffer(b, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("could not decode image")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        passport_img = b64_to_rgb(passport_b64)
        live_img = b64_to_rgb(live_b64)

        # Use face_recognition to get embeddings (128-d)
        # convert to RGB already done
        passport_locations = face_recognition.face_locations(passport_img)
        live_locations = face_recognition.face_locations(live_img)

        if len(passport_locations) == 0:
            return JsonResponse({"error": "no face found in passport image"}, status=400)
        if len(live_locations) == 0:
            return JsonResponse({"error": "no face found in live image"}, status=400)

        passport_enc = face_recognition.face_encodings(passport_img, known_face_locations=passport_locations)[0]
        live_enc = face_recognition.face_encodings(live_img, known_face_locations=live_locations)[0]

        # compute Euclidean distance
        dist = np.linalg.norm(passport_enc - live_enc)
        # common thresholds: 0.4-0.6 depending on model/tolerance; choose 0.55
        SIMILARITY_THRESHOLD = 0.55

        match = dist < SIMILARITY_THRESHOLD

        # optionally: also run simple liveness on live_img using mediapipe single-shot
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6
        ) as fm:
            res = fm.process(live_img)
            liveness_hint = "unknown"
            if res.multi_face_landmarks:
                # simple EAR check on single image won't detect blink, so mark as 'needs video' or rely on previous endpoint
                liveness_hint = "face_detected (single-shot) - recommend multi-frame liveness check"
            else:
                liveness_hint = "no_face_detected"

        return JsonResponse({
            "distance": float(dist),
            "match": bool(match),
            "threshold": SIMILARITY_THRESHOLD,
            "liveness_hint": liveness_hint
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def liveness_test(request):
    return render(request, "test_liveness.html")
