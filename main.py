import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# User information (would come from profile/input in real app)
user_weight_kg = 70  # kg
user_height_cm = 175  # cm
user_age = 30
user_gender = 'male'  # 'male' or 'female'
user_bmr = 1800  # Base Metabolic Rate (calories/day)

# Workout types with their MET values
WORKOUT_TYPES = {
    1: {"name": "Push-ups", "met": 8.0, "target_areas": ["chest", "shoulders", "triceps"]},
    2: {"name": "Squats", "met": 5.0, "target_areas": ["quadriceps", "glutes", "hamstrings"]},
    3: {"name": "Jumping Jacks", "met": 8.0, "target_areas": ["full body", "cardio"]},
    4: {"name": "Lunges", "met": 6.0, "target_areas": ["quadriceps", "glutes", "hamstrings"]},
    5: {"name": "Planks", "met": 4.0, "target_areas": ["core", "shoulders"]},
    6: {"name": "Burpees", "met": 8.0, "target_areas": ["full body", "cardio"]},
}

class WorkoutTracker:
    def __init__(self, user_weight, workout_type):
        self.user_weight = user_weight
        self.workout_type = workout_type
        self.workout_info = WORKOUT_TYPES[workout_type]
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.calories_burned = 0
        self.rep_count = 0
        self.last_positions = []
        self.in_rep_position = False  # Tracks if user is in the "down" position of a rep
        
    def update_calories(self):
        """Update calories burned based on workout MET value"""
        current_time = time.time()
        duration = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Calculate calories: MET * weight(kg) * duration(hours)
        met = self.workout_info["met"]
        calories = met * self.user_weight * (duration / 3600)
        self.calories_burned += calories
        
        return self.calories_burned
    
    def update_reps(self, landmarks):
        """Count repetitions based on pose landmarks and workout type"""
        if landmarks is None:
            return
            
        # Convert landmarks to numpy array
        lm_pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark])
        
        # Store current position for tracking
        if len(self.last_positions) >= 10:
            self.last_positions.pop(0)
        self.last_positions.append(lm_pose.copy())
        
        # Rep counting logic depends on workout type
        if self.workout_type == 1:  # Push-ups
            self._count_pushup_reps(lm_pose)
        elif self.workout_type == 2:  # Squats
            self._count_squat_reps(lm_pose)
        elif self.workout_type == 3:  # Jumping Jacks
            self._count_jumping_jack_reps(lm_pose)
        elif self.workout_type == 4:  # Lunges
            self._count_lunge_reps(lm_pose)
        elif self.workout_type == 5:  # Planks
            # Planks don't have reps, but we could track proper form
            self._check_plank_form(lm_pose)
        elif self.workout_type == 6:  # Burpees
            self._count_burpee_reps(lm_pose)
        
    def _count_pushup_reps(self, lm_pose):
        """Count push-up repetitions"""
        # Get shoulder and elbow landmarks
        left_shoulder = lm_pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value, :2]
        right_shoulder = lm_pose[mp_pose.PoseLandmark.RIGHT_SHOULDER.value, :2]
        left_elbow = lm_pose[mp_pose.PoseLandmark.LEFT_ELBOW.value, :2]
        right_elbow = lm_pose[mp_pose.PoseLandmark.RIGHT_ELBOW.value, :2]
        left_wrist = lm_pose[mp_pose.PoseLandmark.LEFT_WRIST.value, :2]
        right_wrist = lm_pose[mp_pose.PoseLandmark.RIGHT_WRIST.value, :2]
        
        # Calculate elbow angles
        left_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_angle = (left_angle + right_angle) / 2
        
        # Push-up detection logic (lower position = elbows bent, higher = arms extended)
        if not self.in_rep_position and avg_angle < 110:  # User went down
            self.in_rep_position = True
        elif self.in_rep_position and avg_angle > 160:  # User went back up
            self.rep_count += 1
            self.in_rep_position = False
    
    def _count_squat_reps(self, lm_pose):
        """Count squat repetitions"""
        # Get hip, knee and ankle landmarks
        left_hip = lm_pose[mp_pose.PoseLandmark.LEFT_HIP.value, :2]
        left_knee = lm_pose[mp_pose.PoseLandmark.LEFT_KNEE.value, :2]
        left_ankle = lm_pose[mp_pose.PoseLandmark.LEFT_ANKLE.value, :2]
        
        # Calculate knee angle
        knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        
        # Squat detection logic
        if not self.in_rep_position and knee_angle < 120:  # Going down into squat
            self.in_rep_position = True
        elif self.in_rep_position and knee_angle > 160:  # Standing back up
            self.rep_count += 1
            self.in_rep_position = False
    
    def _count_jumping_jack_reps(self, lm_pose):
        """Count jumping jack repetitions"""
        # Get shoulder, hip and ankle landmarks for tracking arm and leg positions
        left_shoulder = lm_pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value, :2]
        right_shoulder = lm_pose[mp_pose.PoseLandmark.RIGHT_SHOULDER.value, :2]
        left_wrist = lm_pose[mp_pose.PoseLandmark.LEFT_WRIST.value, :2]
        right_wrist = lm_pose[mp_pose.PoseLandmark.RIGHT_WRIST.value, :2]
        
        # Calculate distance between wrists (arms spread wide = large distance)
        wrist_distance = np.linalg.norm(left_wrist - right_wrist)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Normalize by shoulder width to account for different body sizes and distances from camera
        normalized_wrist_distance = wrist_distance / shoulder_width
        
        # Jumping jack detection
        if not self.in_rep_position and normalized_wrist_distance > 2.5:  # Arms spread wide
            self.in_rep_position = True
        elif self.in_rep_position and normalized_wrist_distance < 1.2:  # Arms back down
            self.rep_count += 1
            self.in_rep_position = False
    
    def _count_lunge_reps(self, lm_pose):
        """Count lunge repetitions"""
        # Get hip, knee and ankle landmarks
        left_hip = lm_pose[mp_pose.PoseLandmark.LEFT_HIP.value, :2]
        left_knee = lm_pose[mp_pose.PoseLandmark.LEFT_KNEE.value, :2]
        left_ankle = lm_pose[mp_pose.PoseLandmark.LEFT_ANKLE.value, :2]
        right_hip = lm_pose[mp_pose.PoseLandmark.RIGHT_HIP.value, :2]
        right_knee = lm_pose[mp_pose.PoseLandmark.RIGHT_KNEE.value, :2]
        right_ankle = lm_pose[mp_pose.PoseLandmark.RIGHT_ANKLE.value, :2]
        
        # Calculate knee angles
        left_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        # Lunge detection logic (one knee bent significantly)
        if not self.in_rep_position and (left_angle < 110 or right_angle < 110):
            self.in_rep_position = True
        elif self.in_rep_position and (left_angle > 160 and right_angle > 160):
            self.rep_count += 1
            self.in_rep_position = False
    
    def _check_plank_form(self, lm_pose):
        """Check if plank form is correct - no rep counting"""
        # Get shoulder, hip and ankle landmarks
        left_shoulder = lm_pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value, :2]
        left_hip = lm_pose[mp_pose.PoseLandmark.LEFT_HIP.value, :2]
        left_ankle = lm_pose[mp_pose.PoseLandmark.LEFT_ANKLE.value, :2]
        
        # Calculate body alignment angle (should be close to 180 degrees for good form)
        body_angle = self._calculate_angle(left_shoulder, left_hip, left_ankle)
        
        # We're not counting reps for planks, just checking form
        self.good_form = 160 < body_angle < 200
    
    def _count_burpee_reps(self, lm_pose):
        """Count burpee repetitions"""
        # For burpees, we'll track vertical position of hips and whether the person is standing
        hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
        hip_y = lm_pose[hip_idx, 1]  # y coordinate of hip (up/down)
        
        if len(self.last_positions) < 5:
            return
            
        # Calculate average hip height from recent frames
        recent_hip_y = [pos[hip_idx, 1] for pos in self.last_positions[-5:]]
        avg_hip_y = np.mean(recent_hip_y)
        
        # Get shoulder landmarks for height reference
        shoulder_y = lm_pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value, 1]
        
        # Burpee detection logic
        # First detect if person is in low position (on ground)
        if not self.in_rep_position and hip_y > 0.7:  # Hip is low (on ground)
            self.in_rep_position = True
        # Then detect if person has jumped up (hip is high)
        elif self.in_rep_position and hip_y < 0.5 and shoulder_y < 0.3:  # Person has jumped up
            self.rep_count += 1
            self.in_rep_position = False
    
    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points (in degrees)"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Handle numerical errors
        cosine_angle = min(1.0, max(-1.0, cosine_angle))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def get_stats(self):
        """Return current workout statistics"""
        total_duration = time.time() - self.start_time
        return {
            'workout_type': self.workout_info["name"],
            'target_areas': self.workout_info["target_areas"],
            'calories_burned': round(self.calories_burned, 2),
            'duration': round(total_duration, 2),
            'rep_count': self.rep_count,
            'met_value': self.workout_info["met"]
        }

# Mock Apple Health API integration
class HealthAPI:
    def __init__(self):
        self.data = []
    
    def save_workout(self, start_time, end_time, calories, workout_type, reps=None):
        """Mock saving workout data to Health API"""
        workout_data = {
            'start_time': start_time,
            'end_time': end_time,
            'calories': calories,
            'workout_type': workout_type,
            'reps': reps,
            'source': 'WorkoutVision App'
        }
        self.data.append(workout_data)
        print(f"Saved workout to Health API: {workout_data}")
        return True

def display_workout_menu():
    """Display available workout types and get user selection"""
    print("\n===== WORKOUT TYPES =====")
    for key, workout in WORKOUT_TYPES.items():
        targets = ", ".join(workout["target_areas"])
        print(f"{key}. {workout['name']} - Targets: {targets}")
    
    while True:
        try:
            choice = int(input("\nSelect workout type (1-6): "))
            if 1 <= choice <= 6:
                return choice
            print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    # Get user to select workout type
    print("===== WORKOUT CALORIE TRACKER =====")
    print("This application will track your workout and estimate calories burned.")
    workout_choice = display_workout_menu()
    selected_workout = WORKOUT_TYPES[workout_choice]["name"]
    print(f"\nSelected workout: {selected_workout}")
    print(f"Target areas: {', '.join(WORKOUT_TYPES[workout_choice]['target_areas'])}")
    print("\nPress 'q' at any time to end the workout.")
    print("Press any key to begin...")
    input()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    # Initialize workout tracker and health API
    tracker = WorkoutTracker(user_weight=user_weight_kg, workout_type=workout_choice)
    health_api = HealthAPI()
    
    # Record start time for the workout
    workout_start_time = datetime.now()
    
    # Stats display setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            break
            
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect pose
        results = pose.process(image_rgb)
        
        # Draw pose landmarks on the image
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            # Update reps and calories
            tracker.update_reps(results.pose_landmarks)
        
        tracker.update_calories()
        stats = tracker.get_stats()
        
        # Display stats on frame
        cv2.putText(image, f"Workout: {stats['workout_type']}", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Reps: {stats['rep_count']}", (10, 60), font, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Calories: {stats['calories_burned']}", (10, 90), font, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Duration: {stats['duration']:.1f} sec", (10, 120), font, 0.7, (0, 255, 0), 2)
        
        # Show the image
        cv2.imshow('Workout Tracker', image)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # End workout
    workout_end_time = datetime.now()
    final_stats = tracker.get_stats()
    
    # Save workout to health API
    health_api.save_workout(
        workout_start_time,
        workout_end_time,
        final_stats['calories_burned'],
        final_stats['workout_type'],
        reps=final_stats['rep_count']
    )
    
    # Print final stats
    print("\n===== WORKOUT SUMMARY =====")
    print(f"Workout Type: {final_stats['workout_type']}")
    print(f"Target Areas: {', '.join(final_stats['target_areas'])}")
    print(f"Total Duration: {final_stats['duration']:.1f} seconds")
    print(f"Total Calories Burned: {final_stats['calories_burned']}")
    print(f"Repetitions Completed: {final_stats['rep_count']}")
    print(f"MET Value Used: {final_stats['met_value']}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()