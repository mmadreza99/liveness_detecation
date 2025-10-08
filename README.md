---

# مستند فنی پروژه Liveness Detection

## مقدمه

این پروژه یک سیستم تشخیص زنده‌بودن و تطبیق چهره است که شامل موارد زیر است:

* تشخیص زنده‌بودن با استفاده از چالش‌های مختلف (پلک زدن، چرخش سر، بالا بردن ابرو)
* تطبیق تصویر پرسنلی (Passport) با تصویر زنده
* نمایش نتایج و وضعیت‌ها در یک صفحه HTML
* استفاده از Django، MediaPipe، OpenCV و face_recognition
* مدیریت وضعیت‌ها با Cache و لایونِس state

---

## ساختار پروژه

```
project/
├── main/
│   ├── views.py              # منطق اصلی لایونِس و تطبیق چهره
│   ├── tests.py              # تست‌های واحد با unittest و Django
│   └── templates/
│       └── test_liveness.html # صفحه HTML برای تست
```

---

## views.py

### وارد کردن کتابخانه‌ها

```python
import os, uuid, random, base64, json, re
import numpy as np
import cv2
import mediapipe as mp
import face_recognition
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
```

### تنظیمات اولیه

* دایرکتوری ذخیره تصاویر Debug:

```python
DEBUG_DIR = "/tmp/liveness_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)
```

* MediaPipe FaceMesh:

```python
mp_face_mesh = mp.solutions.face_mesh
global_face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
```

* شاخص نقاط چشم:

```python
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
```

### مدیریت state و چالش‌ها

```python
LIVENESS_STATE_KEY = "liveness_state"
CHALLENGES = [
    {"instruction": "سر خود را به چپ و راست بچرخانید", "type": "head_turn", "threshold": 25.0},
    {"instruction": "چند بار پلک بزنید", "type": "blink", "threshold": 3},
    {"instruction": "ابروهای خود را بالا ببرید", "type": "raise_eyebrows", "threshold": 0.03}
]
```

* کلاس کمکی برای نقاط:

```python
class MyObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

### توابع کلیدی

#### مدیریت وضعیت لایونِس

```python
def get_liveness_state():
    """Get or initialize liveness state"""
    state = cache.get(LIVENESS_STATE_KEY, {...})
    if state["current_challenge"] is None:
        state["current_challenge"] = random.choice(CHALLENGES)
        state["challenge_start"] = timezone.now()
        state["challenge_data"] = {"blinks": 0, "pose_changes": []}
    return state

def update_liveness_state(state):
    """Update liveness state in cache"""
    cache.set(LIVENESS_STATE_KEY, state, timeout=120)
```

#### پردازش تصویر

* تبدیل Base64 به OpenCV image:

```python
def base64_to_image(b64str):
    ...
```

* محاسبه نسبت چشم (EAR):

```python
def eye_aspect_ratio(landmarks, eye_indices):
    ...
```

* تخمین وضعیت سر (Pitch/Yaw/Roll):

```python
def estimate_head_pose(landmarks, img_w, img_h):
    ...
```

* تشخیص لبخند:

```python
def detect_smile(landmarks):
    ...
```

* تشخیص زبان بیرون:

```python
def detect_tongue(landmarks):
    ...
```

* تشخیص بالا رفتن ابرو:

```python
def detect_raised_eyebrows(landmarks):
    ...
```

* رسم نقاط و برچسب‌ها:

```python
def annotate_landmarks_and_encode(rgb_img, landmarks=None, boxes=None, labels=None, feature_name=None):
    ...
```

### Viewها

#### چک کردن فریم‌ها

```python
@csrf_exempt
def check_frame(request):
    """Process video frames for optimized liveness detection"""
    ...
```

* دریافت تصویر Base64 و پردازش آن
* اجرای چالش‌ها و ثبت وضعیت‌ها
* برگشت JSON با وضعیت فعلی یا نتیجه

#### تطبیق پاسپورت

```python
@csrf_exempt
def upload_passport_and_verify(request):
    """Endpoint to verify identity by comparing passport and live images"""
    ...
```

* تطبیق چهره با استفاده از face_recognition
* بررسی لایونِس تک فریم
* خروجی JSON شامل match، distance و confidence

#### صفحه تست HTML

```python
def liveness_test(request):
    return render(request, "test_liveness.html")
```

#### ریست کردن تلاش‌ها

```python
@csrf_exempt
def reset_attempts(request):
    cache.delete(LIVENESS_STATE_KEY)
    cache.delete(ATTEMPTS_KEY)
    return JsonResponse({"status": "Attempts reset successfully"})
```

---

## tests.py

* تست عملکرد تبدیل Base64 به تصویر، محاسبات EAR، تخمین سر، تشخیص لبخند/زبان/ابرو، و مدیریت state.
* استفاده از `unittest` و `django.test.TestCase`
* شبیه‌سازی landmarks با `Mock` و `numpy.random`
* تست viewهای Django با `Client` و `patch` کردن کتابخانه‌های خارجی

نمونه:

```python
class Base64ToImageTest(TestCase):
    def test_base64_to_image_success(self):
        result = base64_to_image(self.valid_data_url)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (100, 100, 3))
```

---

## test_liveness.html

* صفحه HTML برای تست لایونِس
* نمایش ویدئو، دستور چالش، وضعیت و پیشرفت
* دکمه شروع و بازنشانی
* ارسال فریم‌ها به سرور با AJAX و fetch
* تایمر شمارش معکوس

نمونه کد JS برای شروع:

```javascript
startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    resetBtn.disabled = false;

    const webcamStarted = await startWebcam();
    if (!webcamStarted) {
        startBtn.disabled = false;
        return;
    }

    const instruction = await getInstruction();
    if (!instruction) {
        startBtn.disabled = false;
        return;
    }

    remainingTime = 10;
    updateCountdown();
    countdownInterval = setInterval(updateCountdown, 1000);
    startSendingFrames();
});
```

---

## نکات فنی

1. **MediaPipe FaceMesh** برای استخراج نقاط کلیدی چهره استفاده شده است.
2. **OpenCV** برای پردازش تصویر و رسم نقاط و bounding boxها استفاده می‌شود.
3. **face_recognition** برای تطبیق چهره بین پاسپورت و تصویر زنده.
4. **Cache** برای ذخیره وضعیت چالش‌ها و جلوگیری از تلاش‌های همزمان.
5. **HTML و JS** برای تعامل کاربر و ارسال فریم‌ها به سرور.
6. **تست‌ها** پوشش کامل توابع اصلی و viewها را دارند و می‌توانند روی فایل واقعی تست شوند.

---

اگر بخواهید، می‌توانم **نسخه بروزشده‌ی `tests.py`** را هم آماده کنم تا از فایل واقعی تصویر استفاده کند و unittestهای Django کاملاً عملیاتی باشند.

آیا مایل هستید این کار انجام شود؟
