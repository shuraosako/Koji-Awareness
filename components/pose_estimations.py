# File: C:\Users\81809\Documents\学校\卒研\test-1\yolotest\components\pose_estimations.py

from datetime import datetime

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
            return True  # 撮影を停止する条件を満たした
        return False

    # 他の姿勢推定メソッドをここに追加できます
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

# 利用可能な姿勢推定の辞書
available_poses = {
    'right_hand_raised': 'Right Hand Raised',
    't_pose': 'T-Pose'
}