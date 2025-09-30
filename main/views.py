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


# محاسبه نسبت چشم (Eye Aspect Ratio)
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    A = ((p2.x - p6.x) ** 2 + (p2.y - p6.y) ** 2) ** 0.5
    B = ((p3.x - p5.x) ** 2 + (p3.y - p5.y) ** 2) ** 0.5
    C = ((p1.x - p4.x) ** 2 + (p1.y - p4.y) ** 2) ** 0.5

    ear = (A + B) / (2.0 * C)
    return ear


@csrf_exempt
def check_frame(request):
    global last_check, blink_count

    if request.method == "POST":
        data = json.loads(request.body)
        img_data = data.get("image")

        # تبدیل base64 به تصویر
        img_str = re.search(r'base64,(.*)', img_data).group(1)
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
        ) as face_mesh:

            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
                ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (ear_left + ear_right) / 2.0

                # آستانه پلک زدن
                if ear < 0.25:
                    blink_count += 1

        # هر 10 ثانیه وضعیت برگردون
        now = timezone.now()
        if (now - last_check).seconds > 10:
            status = "✅ Alive" if blink_count > 0 else "❌ Spoofing suspected"
            last_check = now
            blink_count = 0
            return JsonResponse({"status": status})

        return JsonResponse({"status": f"Processing... blink_count:{blink_count} "})
    return JsonResponse(render(request, 'liveness.html'))


def liveness_test(request):
    return render(request, "test_liveness.html")
