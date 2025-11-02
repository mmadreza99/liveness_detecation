import os
import uuid
import random
import base64
import json
import re
from datetime import timedelta

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

# تعریف بازه‌ی مجاز برای حالت روبه‌رو
FRONT_YAW_RANGE = (-4, 4)    # حدود ۸ درجه چپ و راست مجاز
FRONT_PITCH_RANGE = (-4, 4)  # حدود ۸ درجه بالا و پایین مجاز

# چالش‌های تشخیص زنده‌بودن
COMBO_CHALLENGES = [
    {
        "instruction": "سر خود را به چپ و راست بچرخانید",
        "type": ["head_turn"],
        "thresholds": [25.0]
    },
    {
        "instruction": "چند بار پلک بزنید.(به لنز نگاه کنید)",
        "type": ["blink"],
        "thresholds": [3]
    },
    {
        "instruction": "ابروهای خود را بالا ببرید",
        "type": ["raise_eyebrows"],
        "thresholds": [0.020]
    },
    {
        "instruction": "در حالی که سرت را به چپ و راست می‌چرخانی چند بار پلک بزن",
        "type": ["head_turn", "blink"],  # ✅ دو نوع همزمان
        "thresholds": [25.0, 3]
    },
    {
        "instruction": "در حالی که سرت را به چپ و راست می‌چرخانی ابروهای خود را بالا ,پایین ببرید",
        "type": ["head_turn", "raise_eyebrows"],  # ✅ دو نوع همزمان
        "thresholds": [25.0, 0.020]
    },
    {
        "instruction": "در حالی که ابروهای خود را بالا ,پایین ببرید, چند بار پلک بزنید.(به لنز نگاه کنید)",
        "type": ["blink", "raise_eyebrows"],  # ✅ دو نوع همزمان
        "thresholds": [3, 0.020]
    },
]


class MyObject:
    def __init__(self, x, y ):
        self.x = x
        self.y = y


def create_challenge():
    nonce = uuid.uuid4().hex
    challenge = random.choice(COMBO_CHALLENGES)
    expires = timezone.now() + timedelta(seconds=15)  # کوتاه!
    state = {
        "nonce": nonce,
        "challenge": challenge,
        "expires": expires,
        "created": timezone.now(),
        "attempts": 0,
        "challenge_data":{},
        "recent_poses": [],
        "result_raise_eyebrows": [],
        "len_frame": 0,
    }
    cache.set(f"liveness_ch_{nonce}", state, timeout=30)
    return state


def get_instruction(request=None):
    st = create_challenge()
    return {"instruction": st["challenge"]["instruction"],
            "nonce": st["nonce"],
            "expires": st["expires"].isoformat(),
            }


def update_liveness_state(state):
    """Update liveness state in cache"""
    cache.set(f"liveness_ch_{state['nonce']}", state, timeout=120)


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
    a = np.hypot(p2.x - p6.x, p2.y - p6.y)
    b = np.hypot(p3.x - p5.x, p3.y - p5.y)
    c = np.hypot(p1.x - p4.x, p1.y - p4.y)
    return (a + b) / (2.0 * c) if c != 0 else 0.0


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


def annotate_landmarks_and_encode(rgb_img, landmarks=None, boxes=None, labels=None, feature_name=None, folder=None):
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
        overlay = vis.copy()
        alpha = 0.5
        for lm in landmarks:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            cv2.circle(overlay, (x_px, y_px), 2, (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # label feature name
    if feature_name:
        cv2.putText(vis, feature_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # encode to jpg
    success, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise RuntimeError("Failed to encode debug image")


    # 🟢 Create per-user/request folder (based on nonce or fallback)
    folder = folder or "unknown_request"
    folder_path = os.path.join(DEBUG_DIR, folder)
    os.makedirs(folder_path, exist_ok=True)

    # 🟢 Generate unique file name
    fname = f"{feature_name or 'feat'}_{uuid.uuid4().hex[:8]}.jpg"
    fpath = os.path.join(folder_path, fname)

    # Save the image
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
            state = get_instruction(request)
            print(f'if "get_instruction" in data  : {state["instruction"]}')
            return JsonResponse(state)

        # Convert base64 → image
        img_data = data.get("image", "")
        if not img_data:
            return JsonResponse({"error": "No image provided"}, status=400)
        print('img_data')

        nonce = data.get("nonce")
        if not nonce:
            print('no nonce')
            return JsonResponse({"error": "nonce required"}, status=400)

        state = cache.get(f"liveness_ch_{nonce}")
        if not state:
            return JsonResponse({"error": "invalid or expired nonce"}, status=400)
        if timezone.now() > state["expires"]:
            print('expire')
            return JsonResponse({"error": "nonce expired"}, status=400)

        rgb = base64_to_image(img_data)
        h, w = rgb.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        challenge = state["challenge"]
        challenge_data = state["challenge_data"]
        recent_poses = state["recent_poses"]
        result_raise_eyebrows = state["result_raise_eyebrows"]
        print(f'check instruction  : {state["challenge"]["instruction"]}')

        results = global_face_mesh.process(rgb)
        now = timezone.now()

        if results.multi_face_landmarks:
            print(f'find face : ')
            state["len_frame"] += 1
            count = state['len_frame']
            print(f'len _ frame : {count}')
            landmarks = results.multi_face_landmarks[0].landmark
            if "blink" in challenge["type"]:
                # Calculate blink
                ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
                       eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0
                print('ear', ear)

                if ear < 0.19:
                    challenge_data["blinks"] = challenge_data.get("blinks", 0) + 1
                    print('challenge_data["blinks"]', challenge_data["blinks"])
                    annotate_landmarks_and_encode(rgb, landmarks=landmarks,
                                                  feature_name=f"blink_close_{count}_{ear}", folder=nonce)
                else:
                    challenge_data["blinks_open"] = challenge_data.get("blinks_open", 0) + 1
                    print('challenge_data["blinks_open"]', challenge_data["blinks_open"])
                    annotate_landmarks_and_encode(rgb, landmarks=landmarks, feature_name=f"blink_open_{count}_{ear}",
                                                  folder=nonce)
                    if "blink_front_frame" not in state or abs(ear) > abs(state["ear"]):
                        state["ear"] = ear
                        # تصویر را به base64 ذخیره کن تا در حافظه بماند
                        success, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        if success:
                            b64_image = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')
                            state["blink_front_frame"] = b64_image
                            print("✅ [blink_front_frame] Front-facing frame captured in memory")
            if "head_turn" in challenge["type"]:
                # Head pose estimation
                pose = estimate_head_pose(landmarks, rgb.shape[1], rgb.shape[0])
                recent_poses.append(pose)
                print('poses', pose, 'len', len(recent_poses))

                yaw = abs(pose[1])
                labels = [f"yaw:{yaw:.1f}"]
                annotate_landmarks_and_encode(rgb, landmarks=landmarks, labels=labels,
                                              feature_name=f"head_turn__{count}", folder=nonce)

                if (FRONT_YAW_RANGE[0] < yaw < FRONT_YAW_RANGE[1]
                        and FRONT_PITCH_RANGE[0] < pose[2] < FRONT_PITCH_RANGE[1]) :
                    if "front_frame" not in state or abs(yaw) < abs(state["front_pose"][1]):
                        state["front_pose"] = pose
                        annotate_landmarks_and_encode(rgb, feature_name=f"front_frame_{count}_pose:{pose}", folder=nonce)

                        # تصویر را به base64 ذخیره کن تا در حافظه بماند
                        success, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        if success:
                            b64_image = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')
                            state["front_frame"] = b64_image
                            print("✅ Front-facing frame captured in memory")
            if "raise_eyebrows" in challenge["type"]:
                avg_distance ,new_landmark = detect_raised_eyebrows(landmarks)
                result_raise_eyebrows.append(avg_distance)

                labels = [f"raise_eyebrows: {avg_distance}"]
                ann = annotate_landmarks_and_encode(rgb,  landmarks=new_landmark ,labels=labels,
                                                    feature_name=f"raise_eyebrows-{count}-{avg_distance}", folder=nonce)
        else:
            annotate_landmarks_and_encode(rgb, feature_name=f"not_find_face", folder=nonce)
            print(f'not find face : ')

        # Every 10s check success
        if (now - state["created"]).seconds >= 10:
            types = challenge["type"]
            thresholds = challenge["thresholds"]
            print(f'types {types}')
            results = []

            for t in types:
                if t == "blink":
                    blinks = challenge_data.get("blinks", 0)
                    blinks_open = challenge_data.get("blinks_open", 0)
                    results.append(blinks >= thresholds[types.index(t)] and blinks_open >= thresholds[types.index(t)])
                    print(f'blink    c={blinks}, o={blinks_open}, {thresholds[types.index(t)]}')
                elif t == "head_turn":
                    yaw_changes = [p[1] for p in recent_poses]
                    if len(yaw_changes) >= 2:
                        diff = max(yaw_changes) - min(yaw_changes)
                        results.append(diff > thresholds[types.index(t)])
                        print(f'head_turn    diff={diff},  {thresholds[types.index(t)]}')

                elif t == "raise_eyebrows":
                    if result_raise_eyebrows:
                        min_val = min(result_raise_eyebrows)
                        max_val = max(result_raise_eyebrows)
                        diff = max_val - min_val
                        results.append(diff > thresholds[types.index(t)])
                        print(f'raise_eyebrows  diff={diff} | min_val={min_val}, max_val={max_val},  {thresholds[types.index(t)]}')

            print(f'result: {results} ')
            success = all(results)
            if success:
                status = "✅ Alive - Challenge completed"
            else:
                status = "⚠️ Challenge not detected - Try again"

            new_challenge = random.choice(COMBO_CHALLENGES)
            state["challenge"] = new_challenge

            state.update({
                "created": now,
                "recent_poses": [],
                "challenge_data": {"blinks": 0, "pose_changes": []}
            })
            update_liveness_state(state)
            return JsonResponse({"status": status,
                                 "new_instruction": new_challenge['instruction'],
                                 })
        print('Processing . . . ')

        update_liveness_state(state)
        return JsonResponse({"status": "Processing...",
                             "instruction": state["challenge"]["instruction"],})

    except Exception as e:
        print('error :', e)
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def upload_passport_and_verify(request):
    """
    Verify identity by comparing passport photo with the best front-facing live frame.
    Improved version:
      - uses nonce to fetch the correct liveness session (front frame)
      - returns debug comparison image path (saved under DEBUG_DIR/<nonce>/)
      - does basic face alignment using face_recognition landmarks (center crop)
    """
    if request.method != "POST":
        return JsonResponse({"error": "This endpoint accepts POST requests only"}, status=405)

    try:
        data = json.loads(request.body)
        passport_b64 = data.get("passport_image")
        nonce = data.get("nonce")  # ask frontend to pass the same nonce used in liveness

        if not passport_b64:
            return JsonResponse({"error": "passport_image is required"}, status=400)
        if not nonce:
            return JsonResponse({"error": "nonce is required"}, status=400)

        # Load the liveness session by nonce (use same cache key pattern as check_frame)
        state = cache.get(f"liveness_ch_{nonce}")
        if not state:
            return JsonResponse({"error": "invalid or expired nonce"}, status=400)

        # Prefer front_frame or blink_front_frame captured during liveness
        front_b64 = state.get("front_frame") or state.get("blink_front_frame")
        if not front_b64:
            return JsonResponse({
                "error": "No front-facing frame captured yet. Please complete liveness check first."
            }, status=400)

        # Decode both images (base64 → OpenCV RGB)
        passport_img = base64_to_image(passport_b64)
        live_img = base64_to_image(front_b64)

        # convert to RGB numpy arrays expected by face_recognition (they are already RGB)
        # detect faces in both images
        passport_locations = face_recognition.face_locations(passport_img)
        live_locations = face_recognition.face_locations(live_img)

        if not passport_locations:
            return JsonResponse({"error": "No face detected in passport image"}, status=400)
        if not live_locations:
            return JsonResponse({"error": "No face detected in live (front) frame"}, status=400)

        # Choose the largest face (in case of multiple)
        def pick_largest(locations):
            # locations: list of (top, right, bottom, left)
            areas = [(loc[2]-loc[0])*(loc[1]-loc[3]) for loc in locations]
            return locations[int(np.argmax(areas))]

        passport_loc = pick_largest(passport_locations)
        live_loc = pick_largest(live_locations)

        # Optionally: crop and align faces a bit to improve encoder stability
        def crop_with_margin(img, loc, margin=0.35):
            top, right, bottom, left = loc
            h, w = img.shape[:2]
            height = bottom - top
            width = right - left
            top_m = max(0, int(top - margin * height))
            bottom_m = min(h, int(bottom + margin * height))
            left_m = max(0, int(left - margin * width))
            right_m = min(w, int(right + margin * width))
            return img[top_m:bottom_m, left_m:right_m]

        passport_crop = crop_with_margin(passport_img, passport_loc)
        live_crop = crop_with_margin(live_img, live_loc)

        # Compute face encodings (use cropped images to focus on face)
        # Note: face_recognition.face_encodings expects whole-image coords or known locations; we give a single face.
        passport_encs = face_recognition.face_encodings(passport_crop)
        live_encs = face_recognition.face_encodings(live_crop)

        if not passport_encs:
            return JsonResponse({"error": "Failed to compute encoding for passport face"}, status=500)
        if not live_encs:
            return JsonResponse({"error": "Failed to compute encoding for live face"}, status=500)

        passport_enc = passport_encs[0]
        live_enc = live_encs[0]

        # Compare faces using Euclidean distance
        dist = float(np.linalg.norm(passport_enc - live_enc))

        # Threshold tuning: 0.50-0.6 is common; choose 0.55 default (same as original)
        SIMILARITY_THRESHOLD = 0.55
        match = dist < SIMILARITY_THRESHOLD

        # Compute a simple "confidence" metric (not probabilistic)
        if match:
            confidence = max(0.0, min(100.0, (1.0 - dist / SIMILARITY_THRESHOLD) * 100.0))
        else:
            # give a small score even if not match, scaled
            confidence = max(0.0, min(100.0, (1.0 - dist / (SIMILARITY_THRESHOLD * 2)) * 100.0))

        # Build a side-by-side debug image and save it in DEBUG_DIR/<nonce>/
        def make_side_by_side_and_save(img_a, img_b, label_a="passport", label_b="live", folder=nonce):
            # Convert RGB -> BGR for OpenCV drawing/saving
            a_bgr = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
            b_bgr = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)

            # Resize to same height
            ha, wa = a_bgr.shape[:2]
            hb, wb = b_bgr.shape[:2]
            target_h = max(ha, hb)
            # keep aspect ratio
            def resize_to_height(img, h):
                H, W = img.shape[:2]
                scale = h / H
                return cv2.resize(img, (int(W*scale), h))

            a_r = resize_to_height(a_bgr, target_h)
            b_r = resize_to_height(b_bgr, target_h)

            # pad widths to be equal heights (already equal)
            spacer = 10
            combined = np.concatenate([a_r, np.full((target_h, spacer, 3), 255, dtype=np.uint8), b_r], axis=1)

            # draw rectangles around detected faces on each crop (optional)
            # We don't have face locations in crop space here, skip that complex mapping.

            # annotate text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, label_a, (10, 30), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, label_b, (a_r.shape[1] + spacer + 10, 30), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            # ensure folder exists
            folder_path = os.path.join(DEBUG_DIR, folder)
            os.makedirs(folder_path, exist_ok=True)
            fname = f"compare_{uuid.uuid4().hex[:8]}.jpg"
            fpath = os.path.join(folder_path, fname)
            cv2.imwrite(fpath, combined, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return fpath

        debug_path = make_side_by_side_and_save(passport_crop, live_crop, folder=nonce)

        # Return JSON with result and debug path
        return JsonResponse({
            "match": bool(match),
            "distance": dist,
            "threshold": SIMILARITY_THRESHOLD,
            "confidence": f"{confidence:.1f}%",
            "debug_image_path": debug_path,
            "message": "✅ Identity verified" if match else "⚠️ Faces do not match",
        })

    except Exception as e:
        print("Error in upload_passport_and_verify:", str(e))
        return JsonResponse({"error": str(e)}, status=500)


def liveness_test(request):
    """Render HTML page for liveness test"""
    return render(request, "test_liveness.html")


@csrf_exempt
def reset_attempts(request):
    """Reset attempts counter (for testing)"""
    data = json.loads(request.body)
    nonce = data.get("nonce")
    if not nonce:
        print('no nonce')
        return JsonResponse({"error": "nonce required"}, status=400)

    cache.delete(f"liveness_ch_{nonce}")
    return JsonResponse({"status": "Attempts reset successfully"})
