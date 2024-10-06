# File: C:\Users\81809\Documents\学校\卒研\test-1\yolotest\components\pose_estimations.py

from datetime import datetime
import math

class PoseEstimator:
    def __init__(self):
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.right_hand_raised = False
        self.right_hand_raised_start_time = None
        self.RIGHT_HAND_RAISED_DURATION = 2
        self.flexion_history = []

    def is_right_hand_raised(self, keypoints):
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
        right_wrist = next((kp for kp in keypoints if kp['name'] == 'right_wrist'), None)
        
        if right_shoulder and right_wrist:
            return right_wrist['y'] < right_shoulder['y']
        return False

    def organize_skeleton_data(self, skeleton_data, timestamp):
        organized_skeleton = {
            'timestamp': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'),
            'keypoints': []
        }
        
        for i, keypoint in enumerate(skeleton_data):
            if len(keypoint) == 3:  # x, y, confidence がある場合
                organized_skeleton['keypoints'].append({
                    'name': self.keypoint_names[i],
                    'x': round(float(keypoint[0]), 2),
                    'y': round(float(keypoint[1]), 2),
                    'confidence': round(float(keypoint[2]), 2)
                })
        
        return organized_skeleton

    def check_pose(self, organized_data, current_time):
        if self.is_right_hand_raised(organized_data['keypoints']):
            if not self.right_hand_raised:
                self.right_hand_raised = True
                self.right_hand_raised_start_time = current_time
        else:
            self.right_hand_raised = False
            self.right_hand_raised_start_time = None

        if self.right_hand_raised and (current_time - self.right_hand_raised_start_time) >= self.RIGHT_HAND_RAISED_DURATION:
            return True  # complete条件
        return False
    
    def is_t_pose(self, keypoints):
        left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
        left_elbow = next((kp for kp in keypoints if kp['name'] == 'left_elbow'), None)
        right_elbow = next((kp for kp in keypoints if kp['name'] == 'right_elbow'), None)
        
        if all([left_shoulder, right_shoulder, left_elbow, right_elbow]):
            shoulders_y_diff = abs(left_shoulder['y'] - right_shoulder['y'])
            left_arm_straight = abs(left_shoulder['y'] - left_elbow['y']) < 0.1
            right_arm_straight = abs(right_shoulder['y'] - right_elbow['y']) < 0.1
            
            return shoulders_y_diff < 0.1 and left_arm_straight and right_arm_straight
        return False
    
    def neck_flexion(self, keypoints, tolerance=0.1, time_window=5):
        # 必要なキーポイントを取得
        nose = next((kp for kp in keypoints if kp['name'] == 'nose'), None)
        left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
        left_hip = next((kp for kp in keypoints if kp['name'] == 'left_hip'), None)
        right_hip = next((kp for kp in keypoints if kp['name'] == 'right_hip'), None)
        
        # キーポイントが検出されていない場合はFalseを返す
        if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            return False

        # 肩の中点を計算
        shoulder_midpoint = {
            'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
            'y': (left_shoulder['y'] + right_shoulder['y']) / 2
        }

        # 腰の中点を計算
        hip_midpoint = {
            'x': (left_hip['x'] + right_hip['x']) / 2,
            'y': (left_hip['y'] + right_hip['y']) / 2
        }

        # 肩幅を計算（スケール調整のため）
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])

        # 鼻と肩の中点の垂直距離を計算
        nose_to_shoulder_y = abs(nose['y'] - shoulder_midpoint['y'])

        # 背中の傾きを計算
        back_angle = math.atan2(shoulder_midpoint['y'] - hip_midpoint['y'], 
                                shoulder_midpoint['x'] - hip_midpoint['x'])
        back_angle_degrees = math.degrees(back_angle)

        # 首の屈曲を判定
        neck_flexed = nose_to_shoulder_y <= tolerance * shoulder_width

        # 背中がまっすぐかどうかを判定（許容範囲は±15度）
        back_straight = abs(back_angle_degrees - 90) <= 15

        # 時系列データを使用した判定
        if len(self.flexion_history) >= time_window:
            self.flexion_history.pop(0)
        self.flexion_history.append(neck_flexed and back_straight)
        
        # time_window内のフレームの半分以上でTrue判定された場合にTrueを返す
        return sum(self.flexion_history) > len(self.flexion_history) / 2

    def validate_pose(self, keypoints, tolerance):
        # 両手が腰に当たっているかを確認
        left_wrist = next((kp for kp in keypoints if kp['name'] == 'left_wrist'), None)
        right_wrist = next((kp for kp in keypoints if kp['name'] == 'right_wrist'), None)
        left_hip = next((kp for kp in keypoints if kp['name'] == 'left_hip'), None)
        right_hip = next((kp for kp in keypoints if kp['name'] == 'right_hip'), None)
        left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)

        if all([left_wrist, right_wrist, left_hip, right_hip, left_shoulder, right_shoulder]):
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            left_hand_on_hip = abs(left_wrist['y'] - left_hip['y']) < tolerance * shoulder_width
            right_hand_on_hip = abs(right_wrist['y'] - right_hip['y']) < tolerance * shoulder_width
            return left_hand_on_hip and right_hand_on_hip
        return False

    def assess_neck_flexion_pose(self, keypoints, tolerance=0.1):
        return self.neck_flexion(keypoints) and self.validate_pose(keypoints, tolerance)

# 姿勢推定辞書
available_poses = {
    'right_hand_raised': 'Right Hand Raised',
    't_pose': 'T-Pose',
    'neck_flexion': 'Neck Flexion'
}