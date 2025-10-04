import os
import base64
import json
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse
from django.core.cache import cache
from unittest.mock import Mock, patch
import cv2
from django.utils import timezone

from .views import (
    base64_to_image,
    eye_aspect_ratio,
    estimate_head_pose,
    detect_smile,
    detect_tongue,
    detect_raised_eyebrows,
    get_liveness_state,
    update_liveness_state,
    CHALLENGES
)


class Base64ToImageTest(TestCase):
    """Test cases for base64_to_image function"""

    def setUp(self):
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', self.test_image)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        self.valid_data_url = f"data:image/jpeg;base64,{b64_str}"

    def test_base64_to_image_success(self):
        """Test successful base64 to image conversion"""
        result = base64_to_image(self.valid_data_url)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_base64_to_image_invalid_format(self):
        """Test base64 conversion with invalid format"""
        with self.assertRaises(ValueError) as context:
            base64_to_image("invalid_string")

        self.assertIn("Invalid base64 header", str(context.exception))

    def test_base64_to_image_corrupted_data(self):
        """Test base64 conversion with corrupted data"""
        with self.assertRaises(ValueError):
            base64_to_image("data:image/jpeg;base64,invalid_base64_data")


class EyeDetectionTest(TestCase):
    """Test cases for eye detection functions"""

    def setUp(self):
        # Create mock landmarks for testing
        self.mock_landmarks = []
        for i in range(478):
            landmark = Mock()
            landmark.x = np.random.uniform(0.1, 0.9)
            landmark.y = np.random.uniform(0.1, 0.9)
            landmark.z = np.random.uniform(-0.5, 0.5)
            self.mock_landmarks.append(landmark)

        # Set specific landmarks for reliable testing
        # Left eye landmarks
        self.mock_landmarks[33].x, self.mock_landmarks[33].y = 0.3, 0.4
        self.mock_landmarks[133].x, self.mock_landmarks[133].y = 0.4, 0.4
        self.mock_landmarks[160].x, self.mock_landmarks[160].y = 0.35, 0.35
        self.mock_landmarks[158].x, self.mock_landmarks[158].y = 0.35, 0.45
        self.mock_landmarks[153].x, self.mock_landmarks[153].y = 0.4, 0.35
        self.mock_landmarks[144].x, self.mock_landmarks[144].y = 0.4, 0.45

        # Right eye landmarks
        self.mock_landmarks[362].x, self.mock_landmarks[362].y = 0.6, 0.4
        self.mock_landmarks[263].x, self.mock_landmarks[263].y = 0.7, 0.4
        self.mock_landmarks[385].x, self.mock_landmarks[385].y = 0.65, 0.35
        self.mock_landmarks[387].x, self.mock_landmarks[387].y = 0.65, 0.45
        self.mock_landmarks[373].x, self.mock_landmarks[373].y = 0.7, 0.35
        self.mock_landmarks[380].x, self.mock_landmarks[380].y = 0.7, 0.45

    def test_eye_aspect_ratio_normal_eye(self):
        """Test EAR calculation for normal open eye"""
        left_eye_indices = [33, 160, 158, 133, 153, 144]

        ear = eye_aspect_ratio(self.mock_landmarks, left_eye_indices)

        # EAR for open eye should be relatively high
        self.assertGreater(ear, 0.2)
        self.assertIsInstance(ear, float)

    def test_eye_aspect_ratio_closed_eye(self):
        """Test EAR calculation for closed eye"""
        # Modify landmarks to simulate closed eye
        closed_eye_landmarks = self.mock_landmarks.copy()
        closed_eye_landmarks[160].y = 0.4
        closed_eye_landmarks[158].y = 0.4
        closed_eye_landmarks[153].y = 0.4
        closed_eye_landmarks[144].y = 0.4

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        ear = eye_aspect_ratio(closed_eye_landmarks, left_eye_indices)

        # EAR for closed eye should be low
        self.assertLess(ear, 0.2)


class HeadPoseTest(TestCase):
    """Test cases for head pose estimation"""

    def setUp(self):
        self.mock_landmarks = []
        for i in range(478):
            landmark = Mock()
            landmark.x = np.random.uniform(0.1, 0.9)
            landmark.y = np.random.uniform(0.1, 0.9)
            self.mock_landmarks.append(landmark)

    def test_estimate_head_pose_success(self):
        """Test successful head pose estimation"""
        img_w, img_h = 640, 480

        pitch, yaw, roll = estimate_head_pose(self.mock_landmarks, img_w, img_h)

        self.assertIsInstance(pitch, float)
        self.assertIsInstance(yaw, float)
        self.assertIsInstance(roll, float)

        # Pose should be within reasonable range
        self.assertGreaterEqual(pitch, -90)
        self.assertLessEqual(pitch, 90)
        self.assertGreaterEqual(yaw, -90)
        self.assertLessEqual(yaw, 90)
        self.assertGreaterEqual(roll, -90)
        self.assertLessEqual(roll, 90)

    def test_estimate_head_pose_failure(self):
        """Test head pose estimation with invalid landmarks"""
        invalid_landmarks = [Mock() for _ in range(10)]  # Not enough landmarks

        pitch, yaw, roll = estimate_head_pose(invalid_landmarks, 640, 480)

        # Should return zeros on failure
        self.assertEqual(pitch, 0)
        self.assertEqual(yaw, 0)
        self.assertEqual(roll, 0)


class FacialExpressionTest(TestCase):
    """Test cases for facial expression detection"""

    def setUp(self):
        self.mock_landmarks = []
        for i in range(478):
            landmark = Mock()
            landmark.x = np.random.uniform(0.1, 0.9)
            landmark.y = np.random.uniform(0.1, 0.9)
            self.mock_landmarks.append(landmark)

        # Set specific landmarks for expression detection
        # Mouth landmarks
        self.mock_landmarks[61].x, self.mock_landmarks[61].y = 0.4, 0.6
        self.mock_landmarks[291].x, self.mock_landmarks[291].y = 0.6, 0.6
        self.mock_landmarks[78].x, self.mock_landmarks[78].y = 0.5, 0.55
        self.mock_landmarks[308].x, self.mock_landmarks[308].y = 0.5, 0.65

        # Tongue landmarks
        self.mock_landmarks[13].x, self.mock_landmarks[13].y = 0.5, 0.7
        self.mock_landmarks[152].x, self.mock_landmarks[152].y = 0.5, 0.8

        # Eyebrow landmarks
        self.mock_landmarks[70].x, self.mock_landmarks[70].y = 0.35, 0.3
        self.mock_landmarks[300].x, self.mock_landmarks[300].y = 0.65, 0.3
        self.mock_landmarks[1].x, self.mock_landmarks[1].y = 0.5, 0.4

    def test_detect_smile_positive(self):
        """Test smile detection for smiling face"""
        # Modify landmarks for smile
        smile_landmarks = self.mock_landmarks.copy()
        smile_landmarks[61].x = 0.3
        smile_landmarks[291].x = 0.7
        smile_landmarks[78].y = 0.56
        smile_landmarks[308].y = 0.64

        result = detect_smile(smile_landmarks)

        self.assertTrue(result)

    def test_detect_smile_negative(self):
        """Test smile detection for non-smiling face"""
        # Modify landmarks for non-smile
        nonsmile_landmarks = self.mock_landmarks.copy()
        nonsmile_landmarks[61].x = 0.45
        nonsmile_landmarks[291].x = 0.55
        nonsmile_landmarks[78].y = 0.52
        nonsmile_landmarks[308].y = 0.68

        result = detect_smile(nonsmile_landmarks)

        self.assertFalse(result)

    def test_detect_tongue_positive(self):
        """Test tongue detection when tongue is out"""
        tongue_landmarks = self.mock_landmarks.copy()
        tongue_landmarks[13].y = 0.75

        result = detect_tongue(tongue_landmarks)

        self.assertTrue(result)

    def test_detect_tongue_negative(self):
        """Test tongue detection when tongue is not out"""
        notongue_landmarks = self.mock_landmarks.copy()
        notongue_landmarks[13].y = 0.79

        result = detect_tongue(notongue_landmarks)

        self.assertFalse(result)

    def test_detect_raised_eyebrows_positive(self):
        """Test raised eyebrows detection"""
        raised_landmarks = self.mock_landmarks.copy()
        raised_landmarks[70].y = 0.25
        raised_landmarks[300].y = 0.25

        result = detect_raised_eyebrows(raised_landmarks)

        self.assertTrue(result)

    def test_detect_raised_eyebrows_negative(self):
        """Test non-raised eyebrows detection"""
        normal_landmarks = self.mock_landmarks.copy()
        normal_landmarks[70].y = 0.35
        normal_landmarks[300].y = 0.35

        result = detect_raised_eyebrows(normal_landmarks)

        self.assertFalse(result)


class LivenessStateTest(TestCase):
    """Test cases for liveness state management"""

    def setUp(self):
        cache.clear()

    def test_get_liveness_state_initial(self):
        """Test initial liveness state creation"""
        state = get_liveness_state()

        self.assertIsNotNone(state)
        self.assertIn('current_challenge', state)
        self.assertIn('attempts_remaining', state)
        self.assertEqual(state['attempts_remaining'], 3)
        self.assertIsNotNone(state['current_challenge'])

    def test_update_liveness_state(self):
        """Test liveness state update"""
        initial_state = get_liveness_state()
        initial_state['attempts_remaining'] = 2

        update_liveness_state(initial_state)
        updated_state = get_liveness_state()

        self.assertEqual(updated_state['attempts_remaining'], 2)

    def test_challenge_selection(self):
        """Test that challenges are selected randomly"""
        challenges_seen = set()

        for _ in range(10):
            state = get_liveness_state()
            challenge_type = state['current_challenge']['type']
            challenges_seen.add(challenge_type)
            cache.clear()  # Clear cache to force new challenge selection

        # Should have seen multiple different challenges
        self.assertGreater(len(challenges_seen), 1)


class ViewTest(TestCase):
    """Test cases for Django views"""

    def setUp(self):
        self.client = Client()
        cache.clear()

    def test_liveness_test_view(self):
        """Test the liveness test page view"""
        response = self.client.get(reverse('liveness_test'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Liveness Detection')

    @patch('main.views.global_face_mesh.process')
    def test_check_frame_get_instruction(self, mock_process):
        """Test getting instruction from check_frame"""
        response = self.client.post(
            reverse('check_frame'),
            data=json.dumps({'get_instruction': True}),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('instruction', data)
        self.assertIn(data['instruction'], [challenge['instruction'] for challenge in CHALLENGES])

    @patch('main.views.global_face_mesh.process')
    def test_check_frame_with_face_detection(self, mock_process):
        """Test check_frame with successful face detection"""
        # Mock face detection results
        mock_results = Mock()
        mock_results.multi_face_landmarks = [Mock()]

        # Create realistic landmarks
        landmarks = []
        for i in range(478):
            landmark = Mock()
            landmark.x = np.random.uniform(0.1, 0.9)
            landmark.y = np.random.uniform(0.1, 0.9)
            landmarks.append(landmark)

        mock_results.multi_face_landmarks[0].landmark = landmarks
        mock_process.return_value = mock_results

        # Create test image data
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_image)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64_str}"

        response = self.client.post(
            reverse('check_frame'),
            data=json.dumps({'image': data_url}),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)

    def test_check_frame_no_image_data(self):
        """Test check_frame with no image data"""
        response = self.client.post(
            reverse('check_frame'),
            data=json.dumps({}),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    @patch('main.views.face_recognition.face_locations')
    @patch('main.views.face_recognition.face_encodings')
    @patch('main.views.mp_face_mesh.FaceMesh')
    def test_upload_passport_and_verify_success(self, mock_facemesh, mock_encodings, mock_locations):
        """Test successful passport verification"""
        # Mock face recognition results
        mock_locations.side_effect = [[(10, 20, 30, 40)], [(10, 20, 30, 40)]]
        mock_encodings.side_effect = [
            [np.random.rand(128)],
            [np.random.rand(128)]
        ]

        # Mock face mesh results
        mock_facemesh_instance = Mock()
        mock_facemesh.return_value.__enter__.return_value = mock_facemesh_instance
        mock_facemesh_instance.process.return_value.multi_face_landmarks = [Mock()]

        # Create test images
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_image)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64_str}"

        response = self.client.post(
            reverse('upload_passport_and_verify'),
            data=json.dumps({
                'passport_image': data_url,
                'live_image': data_url
            }),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('match', data)
        self.assertIn('distance', data)
        self.assertIn('confidence', data)

    def test_upload_passport_and_verify_missing_images(self):
        """Test passport verification with missing images"""
        response = self.client.post(
            reverse('upload_passport_and_verify'),
            data=json.dumps({}),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)


class ChallengeReliabilityTest(TestCase):
    """Test the reliability of each challenge type"""

    def test_all_challenges_have_valid_config(self):
        """Test that all challenges have valid configuration"""
        for challenge in CHALLENGES:
            self.assertIn('instruction', challenge)
            self.assertIn('type', challenge)
            self.assertIn('threshold', challenge)
            self.assertGreater(challenge['threshold'], 0)
            self.assertGreater(len(challenge['instruction']), 0)

    def test_blink_challenge_config(self):
        """Test blink challenge configuration"""
        blink_challenge = next(c for c in CHALLENGES if c['type'] == 'blink')

        self.assertEqual(blink_challenge['instruction'], "چند بار پلک بزنید")
        self.assertGreaterEqual(blink_challenge['threshold'], 2)

    def test_head_turn_challenge_config(self):
        """Test head turn challenge configuration"""
        head_turn = next(c for c in CHALLENGES if c['type'] == 'head_turn')

        self.assertEqual(head_turn['instruction'], "سر خود را به چپ و راست بچرخانید")
        self.assertGreaterEqual(head_turn['threshold'], 15.0)


class ErrorRateTest(TestCase):
    """Test error rates for critical functions"""

    def test_error_rate_calculation(self):
        """Calculate error rates for different functions"""
        test_functions = [
            self._test_ear_reliability,
            self._test_smile_detection_reliability,
            self._test_tongue_detection_reliability,
        ]

        print("\n" + "=" * 50)
        print("ERROR RATE ANALYSIS")
        print("=" * 50)

        for test_func in test_functions:
            success_count = 0
            num_iterations = 50

            for _ in range(num_iterations):
                try:
                    test_func()
                    success_count += 1
                except:
                    pass

            error_rate = (1 - success_count / num_iterations) * 100
            print(f"{test_func.__name__}: {error_rate:.2f}% error rate")

        print("=" * 50)

    def _test_ear_reliability(self):
        """Test EAR calculation reliability"""
        landmarks = []
        for i in range(478):
            landmark = Mock()
            landmark.x = np.random.uniform(0.1, 0.9)
            landmark.y = np.random.uniform(0.1, 0.9)
            landmarks.append(landmark)

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        ear = eye_aspect_ratio(landmarks, left_eye_indices)

        self.assertIsInstance(ear, float)
        self.assertGreaterEqual(ear, 0.0)

    def _test_smile_detection_reliability(self):
        """Test smile detection reliability"""
        landmarks = []
        for i in range(478):
            landmark = Mock()
            landmark.x = np.random.uniform(0.1, 0.9)
            landmark.y = np.random.uniform(0.1, 0.9)
            landmarks.append(landmark)

        result = detect_smile(landmarks)
        self.assertIsInstance(result, bool)

    def _test_tongue_detection_reliability(self):
        """Test tongue detection reliability"""
        landmarks = []
        for i in range(478):
            landmark = Mock()
            landmark.x = np.random.uniform(0.1, 0.9)
            landmark.y = np.random.uniform(0.1, 0.9)
            landmarks.append(landmark)

        result = detect_tongue(landmarks)
        self.assertIsInstance(result, bool)