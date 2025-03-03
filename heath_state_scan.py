import random  # For mock data - replace with real API calls in production
import time
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# User information (would come from profile/input in real app)
user_info = {
    "weight_kg": 70,
    "height_cm": 175,
    "age": 30,
    "gender": "male",
    "resting_heart_rate": 65,
    "known_conditions": ["seasonal allergies"],
    "medications": []
}

# Health metrics thresholds
HEALTH_THRESHOLDS = {
    "heart_rate": {"normal": (60, 100), "warning": (50, 120), "danger": (40, 140)},
    "respiratory_rate": {"normal": (12, 20), "warning": (10, 24), "danger": (8, 30)},
    "temperature": {"normal": (36.1, 37.2), "warning": (35.5, 38.0), "danger": (35.0, 39.0)},
    "oxygen_saturation": {"normal": (95, 100), "warning": (90, 94), "danger": (0, 89)},
    "posture_score": {"normal": (80, 100), "warning": (60, 79), "danger": (0, 59)},
    "symmetry_score": {"normal": (80, 100), "warning": (60, 79), "danger": (0, 59)},
    "fatigue_score": {"normal": (80, 100), "warning": (60, 79), "danger": (0, 59)},
    "sleep_hours": {"normal": (7, 9), "warning": (5, 10), "danger": (0, 4)}
}

class HealthMonitor:
    def __init__(self, user_info):
        self.user_info = user_info
        self.current_metrics = {}
        self.historical_metrics = {}
        self.start_time = time.time()
        self.face_landmarks_history = []
        self.pose_landmarks_history = []
        self.last_blink_check_time = time.time()
        self.blink_count = 0
        self.blink_rate = 0
        self.face_color_history = []
        self.last_vitals_update = time.time()
        self.initializing = True
        self.initialization_frames = 0
        
        # Initialize metrics
        self.initialize_metrics()
        
        # Health API connection
        self.health_api = HealthAPI()
        
    def initialize_metrics(self):
        """Initialize health metrics with default values or from historical data"""
        # Core vitals
        self.current_metrics["heart_rate"] = 0  # Will be detected
        self.current_metrics["respiratory_rate"] = 0  # Will be detected
        self.current_metrics["temperature"] = 0  # Will be estimated
        self.current_metrics["oxygen_saturation"] = 0  # Will be estimated
        
        # Computer vision metrics
        self.current_metrics["posture_score"] = 0  # Will be calculated
        self.current_metrics["symmetry_score"] = 0  # Will be calculated
        self.current_metrics["fatigue_score"] = 0  # Will be estimated
        self.current_metrics["blink_rate"] = 0  # Will be counted
        self.current_metrics["skin_tone_variation"] = 0  # Will be monitored
        
        # Health API metrics (mock data for now)
        health_data = self.health_api.get_recent_health_data()
        self.current_metrics["sleep_hours"] = health_data.get("sleep_hours", 0)
        self.current_metrics["activity_level"] = health_data.get("activity_level", 0)
        self.current_metrics["hydration"] = health_data.get("hydration", 0)
        self.current_metrics["stress_level"] = health_data.get("stress_level", 0)
        self.current_metrics["recent_workouts"] = health_data.get("recent_workouts", [])
        
        # Calculated scores
        self.current_metrics["overall_health_score"] = 0  # Will be calculated
        self.current_metrics["illness_risk"] = {"score": 0, "factors": []}  # Will be calculated
    
    def update_vitals(self, face_landmarks, pose_landmarks):
        """Update vital signs using computer vision analysis"""
        # Only update vitals every 0.5 seconds to avoid excessive processing
        current_time = time.time()
        if current_time - self.last_vitals_update < 0.5:
            return
        
        self.last_vitals_update = current_time
        
        # Store landmark history for temporal analysis
        if face_landmarks:
            landmark_points = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            self.face_landmarks_history.append(landmark_points)
            if len(self.face_landmarks_history) > 90:  # Keep ~3 seconds at 30 fps
                self.face_landmarks_history.pop(0)
                
            # Analyze face for skin color
            self._analyze_face_color(face_landmarks)
            
            # Detect blinks
            self._detect_blinks(face_landmarks)
        
        if pose_landmarks:
            landmark_points = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])
            self.pose_landmarks_history.append(landmark_points)
            if len(self.pose_landmarks_history) > 90:
                self.pose_landmarks_history.pop(0)
                
            # Analyze posture
            self._analyze_posture(pose_landmarks)
            
            # Analyze body symmetry
            self._analyze_symmetry(pose_landmarks)
        
        # We need enough history for reliable measurements
        if len(self.face_landmarks_history) > 30 and len(self.pose_landmarks_history) > 30:
            self.initializing = False
            
            # Estimate heart rate from facial landmarks
            self._estimate_heart_rate()
            
            # Estimate respiratory rate from chest movement
            self._estimate_respiratory_rate()
            
            # Estimate temperature from face analysis
            self._estimate_temperature()
            
            # Estimate oxygen saturation
            self._estimate_oxygen_saturation()
            
            # Calculate fatigue score
            self._calculate_fatigue_score()
            
            # Calculate overall health score
            self._calculate_health_score()
            
            # Assess illness risk
            self._assess_illness_risk()
        else:
            self.initialization_frames += 1
            if self.initialization_frames % 10 == 0:
                print(f"Initializing health monitoring: {len(self.face_landmarks_history)}/30 frames")

    def _analyze_face_color(self, face_landmarks):
        """Analyze face color for health indicators"""
        # This would use OpenCV to extract face color information
        # For simplicity, we'll use mock data
        self.face_color_history.append({"r": 150, "g": 120, "b": 110})
        if len(self.face_color_history) > 90:
            self.face_color_history.pop(0)
    
    def _detect_blinks(self, face_landmarks):
        """Detect eye blinks from face landmarks"""
        # MediaPipe face mesh provides eye landmarks
        # For demo purposes using simplified detection
        
        # Get eye landmarks
        if not face_landmarks or not face_landmarks.landmark:
            return
            
        # Check landmarks corresponding to eye openness
        # (These indices are examples, actual indices depend on MediaPipe's face mesh topology)
        left_eye_top = face_landmarks.landmark[159]
        left_eye_bottom = face_landmarks.landmark[145]
        right_eye_top = face_landmarks.landmark[386]
        right_eye_bottom = face_landmarks.landmark[374]
        
        # Calculate eye aspect ratio (EAR)
        left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
        average_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Detect blink if eye height is below threshold
        # This is simplified - real implementation would be more robust
        if average_eye_height < 0.02:  # Threshold for closed eyes
            if not hasattr(self, "eye_closed"):
                self.eye_closed = True
                self.blink_count += 1
        else:
            self.eye_closed = False
            
        # Update blink rate (blinks per minute)
        current_time = time.time()
        elapsed_time = current_time - self.last_blink_check_time
        
        if elapsed_time >= 60:  # Update every minute
            self.blink_rate = self.blink_count
            self.blink_count = 0
            self.last_blink_check_time = current_time
            self.current_metrics["blink_rate"] = self.blink_rate
    
    def _analyze_posture(self, pose_landmarks):
        """Analyze posture from pose landmarks"""
        if not pose_landmarks or not pose_landmarks.landmark:
            return
            
        # Calculate alignment of shoulders, hips, and spine
        left_shoulder = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ])
        right_shoulder = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ])
        left_hip = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
        ])
        right_hip = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ])
        
        # Check shoulder alignment (should be horizontal)
        shoulder_angle = np.degrees(np.arctan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        ))
        shoulder_alignment = 100 - min(abs(shoulder_angle), 15) * 6
        
        # Check hip alignment
        hip_angle = np.degrees(np.arctan2(
            right_hip[1] - left_hip[1],
            right_hip[0] - left_hip[0]
        ))
        hip_alignment = 100 - min(abs(hip_angle), 15) * 6
        
        # Check spine alignment
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        spine_angle = np.degrees(np.arctan2(
            mid_hip[1] - mid_shoulder[1],
            mid_hip[0] - mid_shoulder[0]
        ))
        spine_alignment = 100 - min(abs(spine_angle - 90), 30) * 3
        
        # Calculate overall posture score
        posture_score = (shoulder_alignment + hip_alignment + spine_alignment) / 3
        self.current_metrics["posture_score"] = posture_score
    
    def _analyze_symmetry(self, pose_landmarks):
        """Analyze body symmetry from pose landmarks"""
        if not pose_landmarks or not pose_landmarks.landmark:
            return
            
        # Check symmetry between left and right sides
        # Compare key joint positions and angles
        
        # Example: Compare arm lengths
        left_shoulder = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ])
        left_elbow = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ])
        left_wrist = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        ])
        
        right_shoulder = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ])
        right_elbow = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        ])
        right_wrist = np.array([
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        ])
        
        # Calculate arm segment lengths
        left_upper_arm = np.linalg.norm(left_elbow - left_shoulder)
        left_lower_arm = np.linalg.norm(left_wrist - left_elbow)
        right_upper_arm = np.linalg.norm(right_elbow - right_shoulder)
        right_lower_arm = np.linalg.norm(right_wrist - right_elbow)
        
        # Compare arm symmetry
        upper_arm_sym = 100 - min(abs(left_upper_arm - right_upper_arm) / max(left_upper_arm, right_upper_arm) * 100, 30)
        lower_arm_sym = 100 - min(abs(left_lower_arm - right_lower_arm) / max(left_lower_arm, right_lower_arm) * 100, 30)
        
        # Calculate overall symmetry score
        symmetry_score = (upper_arm_sym + lower_arm_sym) / 2
        self.current_metrics["symmetry_score"] = symmetry_score
    
    def _estimate_heart_rate(self):
        """Estimate heart rate from facial color changes"""
        # In a real implementation, this would analyze subtle color changes
        # in the face that correspond to blood flow (photoplethysmography)
        
        # For demo purposes, we'll use mock data or estimate from other metrics
        if hasattr(self, "last_mock_hr_update") and time.time() - self.last_mock_hr_update < 5:
            return
            
        # Base on user's resting heart rate with some natural variation
        base_hr = self.user_info["resting_heart_rate"]
        variation = np.sin(time.time() / 10) * 5  # Simulate subtle changes over time
        
        # Add variation based on detected activity level
        activity_boost = self.current_metrics.get("activity_level", 0) * 0.5
        
        # Calculate heart rate
        heart_rate = base_hr + variation + activity_boost
        self.current_metrics["heart_rate"] = round(heart_rate)
        self.last_mock_hr_update = time.time()
    
    def _estimate_respiratory_rate(self):
        """Estimate respiratory rate from chest movement"""
        # In a real implementation, this would track the rise and fall of the chest
        # For demo purposes, we'll estimate based on other metrics
        
        if hasattr(self, "last_mock_rr_update") and time.time() - self.last_mock_rr_update < 5:
            return
            
        # Base respiratory rate with natural variation
        base_rr = 14  # Average adult at rest
        variation = np.sin(time.time() / 15) * 1  # Subtle variation
        
        # Link to heart rate (they're correlated)
        hr_factor = (self.current_metrics["heart_rate"] - 60) / 40  # Normalize
        hr_contribution = hr_factor * 3  # Scale factor
        
        # Calculate respiratory rate
        respiratory_rate = base_rr + variation + hr_contribution
        self.current_metrics["respiratory_rate"] = round(respiratory_rate, 1)
        self.last_mock_rr_update = time.time()
    
    def _estimate_temperature(self):
        """Estimate temperature from face analysis"""
        # In a real implementation, this would use thermal imaging or color analysis
        # For demo purposes, we'll use mock data
        
        if hasattr(self, "last_mock_temp_update") and time.time() - self.last_mock_temp_update < 10:
            return
            
        # Normal body temperature with slight variations
        base_temp = 36.6
        variation = np.sin(time.time() / 20) * 0.2
        
        # Factor in detected health metrics
        if self.current_metrics["heart_rate"] > 100:
            temp_increase = (self.current_metrics["heart_rate"] - 100) * 0.01
        else:
            temp_increase = 0
            
        # Calculate temperature
        temperature = base_temp + variation + temp_increase
        self.current_metrics["temperature"] = round(temperature, 1)
        self.last_mock_temp_update = time.time()
    
    def _estimate_oxygen_saturation(self):
        """Estimate blood oxygen saturation"""
        # In a real implementation, this would require specialized hardware or advanced CV
        # For demo purposes, we'll use mock data
        
        if hasattr(self, "last_mock_o2_update") and time.time() - self.last_mock_o2_update < 10:
            return
            
        # Typical healthy range is 95-100%
        base_o2 = 98
        variation = np.sin(time.time() / 25) * 1
        
        # Factor in respiratory rate (lower RR might indicate better O2)
        rr_factor = max(0, (20 - self.current_metrics["respiratory_rate"]) / 20)
        rr_contribution = rr_factor * 1
        
        # Calculate O2 saturation
        o2_sat = base_o2 + variation + rr_contribution
        o2_sat = min(100, max(90, o2_sat))  # Clamp to realistic range
        
        self.current_metrics["oxygen_saturation"] = round(o2_sat)
        self.last_mock_o2_update = time.time()
    
    def _calculate_fatigue_score(self):
        """Calculate fatigue score based on multiple indicators"""
        if not self.current_metrics["blink_rate"]:
            return
            
        # Factors to consider:
        # 1. Blink rate (increased blinking can indicate fatigue)
        blink_score = 100 - min(max(0, self.current_metrics["blink_rate"] - 15), 30) * 2
        
        # 2. Posture (slumping can indicate fatigue)
        posture_score = self.current_metrics["posture_score"]
        
        # 3. Sleep data (if available)
        sleep_factor = 0
        if self.current_metrics["sleep_hours"]:
            optimal_sleep = 8
            sleep_deficit = max(0, optimal_sleep - self.current_metrics["sleep_hours"])
            sleep_factor = 100 - min(sleep_deficit * 10, 40)
        else:
            sleep_factor = 70  # Neutral if no data
        
        # Calculate overall fatigue score
        if sleep_factor > 0:
            fatigue_score = (blink_score * 0.3) + (posture_score * 0.3) + (sleep_factor * 0.4)
        else:
            fatigue_score = (blink_score * 0.5) + (posture_score * 0.5)
            
        self.current_metrics["fatigue_score"] = round(fatigue_score)
    
    def _calculate_health_score(self):
        """Calculate overall health score from all metrics"""
        # Define weights for different health components
        weights = {
            "vitals": 0.3,  # Heart rate, respiratory rate, temperature, O2
            "posture": 0.1,  # Posture and symmetry
            "activity": 0.2,  # Recent physical activity
            "fatigue": 0.15,  # Fatigue indicators
            "sleep": 0.15,  # Sleep quality and quantity
            "hydration": 0.1   # Hydration level
        }
        
        # Calculate vital signs score
        vitals_score = 0
        if self.current_metrics["heart_rate"]:
            hr_min, hr_max = HEALTH_THRESHOLDS["heart_rate"]["normal"]
            hr_score = 100 - min(
                abs(self.current_metrics["heart_rate"] - (hr_min + hr_max) / 2) / ((hr_max - hr_min) / 2) * 100, 
                50
            )
            vitals_score += hr_score * 0.25
            
        if self.current_metrics["respiratory_rate"]:
            rr_min, rr_max = HEALTH_THRESHOLDS["respiratory_rate"]["normal"]
            rr_score = 100 - min(
                abs(self.current_metrics["respiratory_rate"] - (rr_min + rr_max) / 2) / ((rr_max - rr_min) / 2) * 100, 
                50
            )
            vitals_score += rr_score * 0.25
            
        if self.current_metrics["temperature"]:
            temp_min, temp_max = HEALTH_THRESHOLDS["temperature"]["normal"]
            temp_score = 100 - min(
                abs(self.current_metrics["temperature"] - (temp_min + temp_max) / 2) / ((temp_max - temp_min) / 2) * 100, 
                50
            )
            vitals_score += temp_score * 0.25
            
        if self.current_metrics["oxygen_saturation"]:
            o2_min, o2_max = HEALTH_THRESHOLDS["oxygen_saturation"]["normal"]
            o2_score = 100 - min(
                abs(self.current_metrics["oxygen_saturation"] - (o2_min + o2_max) / 2) / ((o2_max - o2_min) / 2) * 100, 
                50
            )
            vitals_score += o2_score * 0.25
            
        # Posture score (already calculated)
        posture_score = (self.current_metrics["posture_score"] + self.current_metrics["symmetry_score"]) / 2
        
        # Activity score from Health API
        activity_score = self.current_metrics["activity_level"]
        
        # Fatigue score (already calculated)
        fatigue_score = self.current_metrics["fatigue_score"]
        
        # Sleep score
        sleep_score = 0
        if self.current_metrics["sleep_hours"]:
            sleep_min, sleep_max = HEALTH_THRESHOLDS["sleep_hours"]["normal"]
            sleep_hours = self.current_metrics["sleep_hours"]
            
            if sleep_min <= sleep_hours <= sleep_max:
                sleep_score = 100
            else:
                deficit = min(abs(sleep_hours - sleep_min), abs(sleep_hours - sleep_max))
                sleep_score = max(0, 100 - deficit * 20)
        
        # Hydration score
        hydration_score = self.current_metrics["hydration"]
        
        # Calculate weighted average
        health_score = (
            vitals_score * weights["vitals"] +
            posture_score * weights["posture"] +
            activity_score * weights["activity"] +
            fatigue_score * weights["fatigue"] +
            sleep_score * weights["sleep"] +
            hydration_score * weights["hydration"]
        )
        
        self.current_metrics["overall_health_score"] = round(health_score)
    
    def _assess_illness_risk(self):
        """Assess risk of illness based on vital signs and other metrics"""
        risk_factors = []
        risk_score = 0
        
        # Check vital signs against warning thresholds
        if self.current_metrics["heart_rate"]:
            hr_min, hr_max = HEALTH_THRESHOLDS["heart_rate"]["warning"]
            if self.current_metrics["heart_rate"] < hr_min:
                risk_factors.append("Low heart rate")
                risk_score += 10
            elif self.current_metrics["heart_rate"] > hr_max:
                risk_factors.append("Elevated heart rate")
                risk_score += 15
        
        if self.current_metrics["respiratory_rate"]:
            rr_min, rr_max = HEALTH_THRESHOLDS["respiratory_rate"]["warning"]
            if self.current_metrics["respiratory_rate"] < rr_min:
                risk_factors.append("Low respiratory rate")
                risk_score += 10
            elif self.current_metrics["respiratory_rate"] > rr_max:
                risk_factors.append("Elevated respiratory rate")
                risk_score += 15
        
        if self.current_metrics["temperature"]:
            temp_min, temp_max = HEALTH_THRESHOLDS["temperature"]["warning"]
            if self.current_metrics["temperature"] < temp_min:
                risk_factors.append("Low body temperature")
                risk_score += 15
            elif self.current_metrics["temperature"] > temp_max:
                risk_factors.append("Elevated body temperature")
                risk_score += 25
        
        if self.current_metrics["oxygen_saturation"]:
            o2_min, _ = HEALTH_THRESHOLDS["oxygen_saturation"]["warning"]
            if self.current_metrics["oxygen_saturation"] < o2_min:
                risk_factors.append("Low oxygen saturation")
                risk_score += 20
        
        # Check fatigue level
        if self.current_metrics["fatigue_score"] < 60:
            risk_factors.append("High fatigue level")
            risk_score += 10
        
        # Check recent sleep
        if self.current_metrics["sleep_hours"] < 5:
            risk_factors.append("Sleep deficit")
            risk_score += 15
        
        # Check hydration
        if self.current_metrics["hydration"] < 50:
            risk_factors.append("Dehydration risk")
            risk_score += 10
        
        # Normalize risk score
        risk_score = min(100, risk_score)
        
        # Store results
        self.current_metrics["illness_risk"] = {
            "score": risk_score,
            "factors": risk_factors
        }
    
    def get_health_state(self):
        """Return comprehensive health state information"""
        return {
            "overall_health_score": self.current_metrics["overall_health_score"],
            "vital_signs": {
                "heart_rate": self.current_metrics["heart_rate"],
                "respiratory_rate": self.current_metrics["respiratory_rate"],
                "temperature": self.current_metrics["temperature"],
                "oxygen_saturation": self.current_metrics["oxygen_saturation"]
            },
            "posture": {
                "posture_score": self.current_metrics["posture_score"],
                "symmetry_score": self.current_metrics["symmetry_score"]
            },
            "fatigue": {
                "fatigue_score": self.current_metrics["fatigue_score"],
                "blink_rate": self.current_metrics["blink_rate"]
            },
            "lifestyle": {
                "sleep_hours": self.current_metrics["sleep_hours"],
                "activity_level": self.current_metrics["activity_level"],
                "hydration": self.current_metrics["hydration"]
            },
            "illness_risk": self.current_metrics["illness_risk"],
            "status": "initializing" if self.initializing else "active"
        }
    
    def save_health_snapshot(self):
        """Save current health metrics to Health API"""
        if self.initializing:
            return False
            
        snapshot = {
            "timestamp": datetime.now(),
            "overall_health_score": self.current_metrics["overall_health_score"],
            "vital_signs": {
                "heart_rate": self.current_metrics["heart_rate"],
                "respiratory_rate": self.current_metrics["respiratory_rate"],
                "temperature": self.current_metrics["temperature"],
                "oxygen_saturation": self.current_metrics["oxygen_saturation"]
            },
            "fatigue_score": self.current_metrics["fatigue_score"],
            "posture_score": self.current_metrics["posture_score"],
            "illness_risk": self.current_metrics["illness_risk"]["score"]
        }
        
        return self.health_api.save_health_data(snapshot)

# Mock Apple Health API
class HealthAPI:
    def __init__(self):
        self.data = []
        self.health_snapshots = []
        self.available_data_types = [
            "sleep", "activity", "workouts", "weight", "heart_rate", 
            "respiratory_rate", "oxygen_saturation", "hydration"
        ]
    
    def get_recent_health_data(self):
        """Get recent health data from Apple Health (mock)"""
        # In a real app, this would query the Apple Health API
        # For demo purposes, return mock data
        
        # Mock data: Randomize to simulate real changes
        sleep_hours = 7 + random.uniform(-1, 1)
        activity_level = 65 + random.uniform(-10, 10)
        hydration = 75 + random.uniform(-15, 10)
        stress_level = 40 + random.uniform(-20, 20)
        
        # Mock workout history
        recent_workouts = [
            {"date": datetime.now() - timedelta(days=1), "type": "Running", "duration": 30, "calories": 320},
            {"date": datetime.now() - timedelta(days=3), "type": "Strength", "duration": 45, "calories": 280}
        ]
        
        return {
            "sleep_hours": sleep_hours,
            "activity_level": activity_level,
            "hydration": hydration,
            "stress_level": stress_level,
            "recent_workouts": recent_workouts
        }
    
    def save_health_data(self, data):
        """Save health data to Apple Health (mock)"""
        # In a real app, this would use the Apple Health API to save data
        self.health_snapshots.append(data)
        print(f"Saved health snapshot to Health API: {datetime.now()}")
        return True
    
    def get_data_availability(self):
        """Check what health data types are available"""
        # In a real app, this would query the Apple Health API
        # to determine what data is available for this user
        return {data_type: True for data_type in self.available_data_types}
    
    def get_historical_data(self, data_type, days=30):
        """Get historical health data (mock)"""
        # In a real app, this would query historical data from the Health API
        if data_type not in self.available_data_types:
            return []
            
        # Generate mock historical data
        history = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            if data_type == "sleep":
                value = 7 + random.uniform(-1.5, 1)
                unit = "hours"
            elif data_type == "heart_rate":
                value = 65 + random.uniform(-10, 15)
                unit = "bpm"
            elif data_type == "activity":
                value = 70 + random.uniform(-30, 20)
                unit = "score"
            elif data_type == "hydration":
                value = 75 + random.uniform(-25, 15)
                unit = "percent"
            else:
                value = 50 + random.uniform(-20, 30)
                unit = "value"
                
            history.append({
                "date": date,
                "value": value,
                "unit": unit
            })
            
        return history

def get_color_for_metric(metric, value):
   """Get color based on metric value compared to thresholds"""
   if metric not in HEALTH_THRESHOLDS:
       return (0, 255, 0)  # Default green
       
   thresholds = HEALTH_THRESHOLDS[metric]
   
   # Check if value is in normal range
   normal_min, normal_max = thresholds["normal"]
   if normal_min <= value <= normal_max:
       return (0, 255, 0)  # Green
       
   # Check if value is in warning range
   warning_min, warning_max = thresholds["warning"]
   if warning_min <= value <= warning_max:
       return (0, 255, 255)  # Yellow
       
   # Value is in danger range
   return (0, 0, 255)  # Red (Note: OpenCV uses BGR)

def create_health_dashboard(health_state):
   """Create a dashboard image from health state data"""
   # Create a blank white image
   dashboard = np.ones((600, 800, 3), np.uint8) * 255
   
   # Set up fonts
   title_font = cv2.FONT_HERSHEY_SIMPLEX
   font = cv2.FONT_HERSHEY_PLAIN
   
   # Draw title
   cv2.putText(dashboard, "Health Dashboard", (20, 40), title_font, 1.2, (0, 0, 0), 2)
   
   # Draw overall health score
   health_score = health_state["overall_health_score"]
   score_color = (0, int(255 * health_score / 100), int(255 * (100 - health_score) / 100))
   cv2.putText(dashboard, f"Overall Health Score: {health_score}", (20, 80), title_font, 1, score_color, 2)
   
   # Draw vital signs section
   cv2.putText(dashboard, "Vital Signs", (20, 120), title_font, 0.8, (0, 0, 0), 2)
   vital_signs = health_state["vital_signs"]
   y_pos = 150
   for metric, value in vital_signs.items():
       if value == 0:  # Skip uninitialized values
           continue
           
       # Format display based on metric
       if metric == "heart_rate":
           display_text = f"Heart Rate: {value} bpm"
       elif metric == "respiratory_rate":
           display_text = f"Respiratory Rate: {value} bpm"
       elif metric == "temperature":
           display_text = f"Body Temperature: {value}°C"
       elif metric == "oxygen_saturation":
           display_text = f"Oxygen Saturation: {value}%"
       else:
           display_text = f"{metric}: {value}"
           
       color = get_color_for_metric(metric, value)
       cv2.putText(dashboard, display_text, (40, y_pos), font, 1.2, color, 1)
       y_pos += 30
   
   # Draw posture section
   cv2.putText(dashboard, "Body Metrics", (20, y_pos + 20), title_font, 0.8, (0, 0, 0), 2)
   y_pos += 50
   
   posture = health_state["posture"]
   for metric, value in posture.items():
       display_text = f"{metric.replace('_', ' ').title()}: {value}"
       color = get_color_for_metric(metric, value)
       cv2.putText(dashboard, display_text, (40, y_pos), font, 1.2, color, 1)
       y_pos += 30
   
   # Draw fatigue section
   fatigue = health_state["fatigue"]
   for metric, value in fatigue.items():
       if metric == "blink_rate":
           display_text = f"Blink Rate: {value} blinks/min"
       else:
           display_text = f"{metric.replace('_', ' ').title()}: {value}"
       
       # Set color based on metric
       if metric == "fatigue_score":
           color = get_color_for_metric(metric, value)
       else:
           color = (0, 0, 0)  # Black for non-scored metrics
           
       cv2.putText(dashboard, display_text, (40, y_pos), font, 1.2, color, 1)
       y_pos += 30
   
   # Draw lifestyle section
   cv2.putText(dashboard, "Lifestyle Metrics", (400, 120), title_font, 0.8, (0, 0, 0), 2)
   
   lifestyle = health_state["lifestyle"]
   y_pos = 150
   for metric, value in lifestyle.items():
       if value == 0:  # Skip uninitialized values
           continue
           
       if metric == "sleep_hours":
           display_text = f"Sleep: {value:.1f} hours"
           color = get_color_for_metric(metric, value)
       elif metric == "activity_level":
           display_text = f"Activity Level: {value:.1f}/100"
           color = (0, min(255, int(value * 2.55)), max(0, int(255 - value * 2.55)))
       elif metric == "hydration":
           display_text = f"Hydration: {value:.1f}%"
           color = (min(255, int((100 - value) * 2.55)), min(255, int(value * 2.55)), 0)
       else:
           display_text = f"{metric.replace('_', ' ').title()}: {value}"
           color = (0, 0, 0)
           
       cv2.putText(dashboard, display_text, (420, y_pos), font, 1.2, color, 1)
       y_pos += 30
   
   # Draw illness risk section
   illness_risk = health_state["illness_risk"]
   cv2.putText(dashboard, "Illness Risk Assessment", (400, y_pos + 20), title_font, 0.8, (0, 0, 0), 2)
   y_pos += 50
   
   risk_score = illness_risk["score"]
   risk_color = (0, max(0, 255 - risk_score * 2.55), min(255, risk_score * 2.55))
   cv2.putText(dashboard, f"Risk Score: {risk_score}/100", (420, y_pos), font, 1.2, risk_color, 1)
   y_pos += 30
   
   # List risk factors
   risk_factors = illness_risk["factors"]
   if risk_factors:
       cv2.putText(dashboard, "Risk Factors:", (420, y_pos), font, 1.2, (0, 0, 0), 1)
       y_pos += 30
       
       for factor in risk_factors[:3]:  # Show up to 3 factors
           cv2.putText(dashboard, f"- {factor}", (440, y_pos), font, 1.1, (0, 0, 150), 1)
           y_pos += 25
   else:
       cv2.putText(dashboard, "No significant risk factors detected", (420, y_pos), font, 1.1, (0, 150, 0), 1)
   
   # Add timestamp
   cv2.putText(dashboard, f"Last Updated: {datetime.now().strftime('%H:%M:%S')}", 
               (20, 580), font, 1, (100, 100, 100), 1)
               
   # Add status indicator
   status = health_state["status"]
   if status == "initializing":
       cv2.putText(dashboard, "Status: Initializing...", (600, 580), font, 1, (0, 0, 255), 1)
   else:
       cv2.putText(dashboard, "Status: Active", (600, 580), font, 1, (0, 255, 0), 1)
   
   return dashboard

def main():
   # Initialize components
   cap = cv2.VideoCapture(0)  # Use webcam
   
   # Start with a health scan mode
   print("===== HEALTH MONITORING SYSTEM =====")
   print("This system will scan your health state using computer vision and health data.")
   print("Starting health scan...")
   
   # Initialize health monitor
   health_monitor = HealthMonitor(user_info)
   
   # Dashboard setup
   dashboard_active = False
   health_dashboard = None
   last_dashboard_update = time.time()
   
   # Main processing loop
   scan_start_time = time.time()
   while cap.isOpened():
       success, image = cap.read()
       if not success:
           print("Failed to capture image")
           break
       
       # Convert the BGR image to RGB for MediaPipe
       image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
       # Process the image for pose and face detection
       pose_results = pose.process(image_rgb)
       face_results = face_mesh.process(image_rgb)
       
       # Draw landmarks on the image
       annotated_image = image.copy()
       
       if pose_results.pose_landmarks:
           mp_drawing.draw_landmarks(
               annotated_image,
               pose_results.pose_landmarks,
               mp_pose.POSE_CONNECTIONS,
               landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
           )
       
       if face_results.multi_face_landmarks:
           for face_landmarks in face_results.multi_face_landmarks:
               mp_drawing.draw_landmarks(
                   annotated_image,
                   face_landmarks,
                   mp_face_mesh.FACEMESH_CONTOURS,
                   landmark_drawing_spec=None,
                   connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
               )
       
       # Update health monitoring
       if pose_results.pose_landmarks and face_results.multi_face_landmarks:
           health_monitor.update_vitals(
               face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None,
               pose_results.pose_landmarks
           )
       
       # Create or update health dashboard
       current_time = time.time()
       if current_time - last_dashboard_update > 1:  # Update dashboard every second
           health_state = health_monitor.get_health_state()
           health_dashboard = create_health_dashboard(health_state)
           last_dashboard_update = current_time
           
           # After initial scanning period, save health snapshot
           if current_time - scan_start_time > 15 and not dashboard_active:
               health_monitor.save_health_snapshot()
               dashboard_active = True
       
       # Show the video feed and dashboard
       if health_dashboard is not None and dashboard_active:
           # Display camera feed in smaller window
           camera_height, camera_width = annotated_image.shape[:2]
           small_camera = cv2.resize(annotated_image, (int(camera_width/2), int(camera_height/2)))
           
           # Insert camera feed into dashboard
           h, w = small_camera.shape[:2]
           health_dashboard[400:400+h, 400:400+w] = small_camera
           
           # Show dashboard
           cv2.imshow('Health Monitor', health_dashboard)
       else:
           # During initialization, just show camera feed with countdown
           elapsed = int(current_time - scan_start_time)
           remaining = max(0, 15 - elapsed)
           cv2.putText(
               annotated_image,
               f"Health scan initializing: {remaining}s remaining",
               (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX,
               1,
               (0, 0, 255),
               2
           )
           cv2.imshow('Health Monitor', annotated_image)
       
       # Break the loop when 'q' is pressed
       if cv2.waitKey(5) & 0xFF == ord('q'):
           break
   
   # Finalize and save health data
   if not health_monitor.initializing:
       health_monitor.save_health_snapshot()
       print("\n===== HEALTH SCAN COMPLETE =====")
       health_state = health_monitor.get_health_state()
       
       print(f"Overall Health Score: {health_state['overall_health_score']}/100")
       print("\nVital Signs:")
       for metric, value in health_state["vital_signs"].items():
           if value > 0:
               print(f"  {metric.replace('_', ' ').title()}: {value}")
       
       print("\nBody Metrics:")
       print(f"  Posture Score: {health_state['posture']['posture_score']}")
       print(f"  Symmetry Score: {health_state['posture']['symmetry_score']}")
       print(f"  Fatigue Score: {health_state['fatigue']['fatigue_score']}")
       
       print("\nIllness Risk Assessment:")
       print(f"  Risk Score: {health_state['illness_risk']['score']}/100")
       if health_state['illness_risk']['factors']:
           print("  Risk Factors:")
           for factor in health_state['illness_risk']['factors']:
               print(f"    - {factor}")
       else:
           print("  No significant risk factors detected")
           
       print("\nRecommendations:")
       if health_state['overall_health_score'] < 60:
           print("  - Consider consulting a healthcare professional for a checkup")
       
       if 'Low respiratory rate' in health_state['illness_risk']['factors']:
           print("  - Practice deep breathing exercises")
           
       if health_state['posture']['posture_score'] < 70:
           print("  - Work on improving posture to reduce strain")
       
       if health_state['fatigue']['fatigue_score'] < 70:
           print("  - Prioritize adequate rest and sleep")
       
       if health_state['lifestyle']['sleep_hours'] < 6:
           print("  - Increase sleep duration to 7-8 hours")
   else:
       print("\nHealth scan was interrupted before completion.")
   
   # Release resources
   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()