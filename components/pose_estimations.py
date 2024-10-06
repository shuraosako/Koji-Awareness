# File: C:\Users\81809\Documents\学校\卒研\test-1\yolotest\components\pose_estimations.py

from datetime import datetime
import math
from collections import deque

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
        self.flexion_history = []  # 首の屈曲
        self.lateral_flexion_history = deque(maxlen=90)  # 首の側屈3秒間
        self.LATERAL_FLEXION_DURATION = 3  # 首の側屈
        self.rotation_history = deque(maxlen=90)  # 首の回旋3秒間
        self.ROTATION_DURATION = 3 #首の回旋
        self.extension_history = deque(maxlen=90)  # 首の伸展3秒間
        self.EXTENSION_DURATION = 3  #首の伸展

    # ここから右手挙げ
    def is_right_hand_raised(self, keypoints):
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
        right_wrist = next((kp for kp in keypoints if kp['name'] == 'right_wrist'), None)
        
        if right_shoulder and right_wrist:
            return right_wrist['y'] < right_shoulder['y']
        return False

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
    # ここまで右手挙げ
    
    # ここから首の屈曲
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
    # ここまで首の屈曲

    # ここから首の側屈
    def lateral_flexion_neck(self, keypoints, threshold_angle=15):
        # 必要なキーポイントを取得
        left_eye = next((kp for kp in keypoints if kp['name'] == 'left_eye'), None)
        right_eye = next((kp for kp in keypoints if kp['name'] == 'right_eye'), None)
        nose = next((kp for kp in keypoints if kp['name'] == 'nose'), None)
        left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
        left_elbow = next((kp for kp in keypoints if kp['name'] == 'left_elbow'), None)
        right_elbow = next((kp for kp in keypoints if kp['name'] == 'right_elbow'), None)
        left_hip = next((kp for kp in keypoints if kp['name'] == 'left_hip'), None)
        right_hip = next((kp for kp in keypoints if kp['name'] == 'right_hip'), None)

        # キーポイントが検出されていない場合はFalseを返す
        if not all([left_eye, right_eye, nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_hip, right_hip]):
            return False

        # 両目の中点を計算
        eyes_midpoint = {
            'x': (left_eye['x'] + right_eye['x']) / 2,
            'y': (left_eye['y'] + right_eye['y']) / 2
        }

        # V_center（両目の中点と鼻のベクトル）を計算
        v_center = [nose['x'] - eyes_midpoint['x'], nose['y'] - eyes_midpoint['y']]

        # 左右の V_arm（肩と肘のベクトル）を計算
        v_arm_left = [left_elbow['x'] - left_shoulder['x'], left_elbow['y'] - left_shoulder['y']]
        v_arm_right = [right_elbow['x'] - right_shoulder['x'], right_elbow['y'] - right_shoulder['y']]

        # ベクトル間の角度を計算する関数
        def angle_between(v1, v2):
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
            cos_angle = dot_product / (magnitude1 * magnitude2)
            angle = math.acos(max(-1, min(cos_angle, 1)))  # アークコサインの定義域を確保
            return math.degrees(angle)

        # 左右の角度を計算
        angle_left = angle_between(v_center, v_arm_left)
        angle_right = angle_between(v_center, v_arm_right)

        # 背中がまっすぐかを確認
        shoulder_midpoint = {
            'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
            'y': (left_shoulder['y'] + right_shoulder['y']) / 2
        }
        hip_midpoint = {
            'x': (left_hip['x'] + right_hip['x']) / 2,
            'y': (left_hip['y'] + right_hip['y']) / 2
        }
        back_angle = math.degrees(math.atan2(shoulder_midpoint['y'] - hip_midpoint['y'], 
                                             shoulder_midpoint['x'] - hip_midpoint['x']))
        back_straight = abs(back_angle - 90) <= 15

        # 両手が腰に当たっているかを確認
        hands_on_hips = self.validate_pose(keypoints, 0.1)

        # 側屈の判定
        lateral_flexion = (angle_left <= threshold_angle or angle_right <= threshold_angle) and back_straight and hands_on_hips

        # 履歴に追加
        self.lateral_flexion_history.append(lateral_flexion)

        # 3秒間維持されているかを確認
        if len(self.lateral_flexion_history) == 90 and all(self.lateral_flexion_history):
            return True

        return False
    # ここまで首の側屈

     # ここから首の回旋
    def neck_rotation(self, keypoints, threshold_angle=15):
        # 必要なキーポイントを取得
        nose = next((kp for kp in keypoints if kp['name'] == 'nose'), None)
        left_eye = next((kp for kp in keypoints if kp['name'] == 'left_eye'), None)
        right_eye = next((kp for kp in keypoints if kp['name'] == 'right_eye'), None)
        left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
        left_hip = next((kp for kp in keypoints if kp['name'] == 'left_hip'), None)
        right_hip = next((kp for kp in keypoints if kp['name'] == 'right_hip'), None)

        # キーポイントが検出されていない場合はFalseを返す
        if not all([nose, left_eye, right_eye, left_shoulder, right_shoulder, left_hip, right_hip]):
            return False

        # 顔の正中線のベクトルを計算
        face_midpoint = {
            'x': (left_eye['x'] + right_eye['x']) / 2,
            'y': (left_eye['y'] + right_eye['y']) / 2
        }
        face_vector = [nose['x'] - face_midpoint['x'], nose['y'] - face_midpoint['y']]

        # 肩のベクトルを計算
        shoulder_vector = [right_shoulder['x'] - left_shoulder['x'], right_shoulder['y'] - left_shoulder['y']]

        # ベクトル間の角度を計算する関数
        def angle_between(v1, v2):
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
            cos_angle = dot_product / (magnitude1 * magnitude2)
            angle = math.acos(max(-1, min(cos_angle, 1)))  # アークコサインの定義域を確保
            return math.degrees(angle)

        # 顔の正中線と肩の向きの角度を計算
        rotation_angle = angle_between(face_vector, shoulder_vector)

        # 背中がまっすぐかを確認
        hip_midpoint = {
            'x': (left_hip['x'] + right_hip['x']) / 2,
            'y': (left_hip['y'] + right_hip['y']) / 2
        }
        shoulder_midpoint = {
            'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
            'y': (left_shoulder['y'] + right_shoulder['y']) / 2
        }
        back_angle = math.degrees(math.atan2(shoulder_midpoint['y'] - hip_midpoint['y'], 
                                             shoulder_midpoint['x'] - hip_midpoint['x']))
        back_straight = abs(back_angle - 90) <= 15

        # 両手が腰に当たっているかを確認
        hands_on_hips = self.validate_pose(keypoints, 0.1)

        # 首の回旋を判定
        neck_rotated = abs(rotation_angle - 90) <= threshold_angle and back_straight and hands_on_hips

        # 履歴に追加
        self.rotation_history.append(neck_rotated)

        # 3秒間維持されているかを確認
        if len(self.rotation_history) == 90 and all(self.rotation_history):
            return True

        return False
    # ここまで首の回旋

    # ここから首の伸展（Neck Extension）
    def neck_extension(self, keypoints, threshold_angle=15):
        # 必要なキーポイントを取得
        nose = next((kp for kp in keypoints if kp['name'] == 'nose'), None)
        left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
        left_elbow = next((kp for kp in keypoints if kp['name'] == 'left_elbow'), None)
        right_elbow = next((kp for kp in keypoints if kp['name'] == 'right_elbow'), None)
        left_wrist = next((kp for kp in keypoints if kp['name'] == 'left_wrist'), None)
        right_wrist = next((kp for kp in keypoints if kp['name'] == 'right_wrist'), None)
        left_knee = next((kp for kp in keypoints if kp['name'] == 'left_knee'), None)
        right_knee = next((kp for kp in keypoints if kp['name'] == 'right_knee'), None)
        left_hip = next((kp for kp in keypoints if kp['name'] == 'left_hip'), None)
        right_hip = next((kp for kp in keypoints if kp['name'] == 'right_hip'), None)
        left_ankle = next((kp for kp in keypoints if kp['name'] == 'left_ankle'), None)
        right_ankle = next((kp for kp in keypoints if kp['name'] == 'right_ankle'), None)

        # キーポイントが検出されていない場合はFalseを返す
        if not all([nose, left_shoulder, right_shoulder, left_elbow, right_elbow, 
                    left_wrist, right_wrist, left_knee, right_knee, left_hip, right_hip,
                    left_ankle, right_ankle]):
            return False

        # スフィンクスのポーズを確認
        elbow_knee_angle = self.calculate_angle(left_elbow, left_knee, right_knee)
        back_horizontal = abs(self.calculate_angle(left_shoulder, left_hip, right_hip) - 180) <= threshold_angle

        # 首の伸展を確認
        neck_extended = nose['y'] < min(left_shoulder['y'], right_shoulder['y'])

        # 体の長さを推定（肩から足首までの距離）
        body_length = max(
            self.distance(left_shoulder, left_ankle),
            self.distance(right_shoulder, right_ankle)
        )

        # 「4足分」の距離を推定（体の長さの約2/3）
        required_distance = body_length * 2/3

        # 手首の位置が適切な距離にあるか確認
        wrist_to_shoulder_distance = min(
            self.distance(left_wrist, left_shoulder),
            self.distance(right_wrist, right_shoulder)
        )
        wrist_position_correct = abs(wrist_to_shoulder_distance - required_distance) <= required_distance * 0.1  # 10%の誤差を許容

        # 「印」の高さを推定（体の長さの約1/3）
        mark_height = body_length / 3

        # 頭が「印」の高さまで上がっているか確認
        head_reaches_mark = nose['y'] <= left_shoulder['y'] - mark_height

        # すべての条件を満たしているか確認
        extension_correct = (elbow_knee_angle >= 85 and elbow_knee_angle <= 95 and
                             back_horizontal and neck_extended and
                             wrist_position_correct and head_reaches_mark)

        # 履歴に追加
        self.extension_history.append(extension_correct)

        # 3秒間維持されているかを確認
        if len(self.extension_history) == 90 and all(self.extension_history):
            return True

        return False

    def calculate_angle(self, point1, point2, point3):
        # 3点間の角度を計算する補助関数
        angle = math.degrees(math.atan2(point3['y'] - point2['y'], point3['x'] - point2['x']) -
                             math.atan2(point1['y'] - point2['y'], point1['x'] - point2['x']))
        return abs(angle)

    def distance(self, point1, point2):
        # 2点間の距離を計算する補助関数
        return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)
    #ここまで首の伸展

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

# 姿勢推定辞書
available_poses = {
    'right_hand_raised': 'Right Hand Raised',
    't_pose': 'T-Pose',
    'neck_flexion': 'Neck Flexion',
    'lateral_flexion_neck': 'Lateral Flexion Neck',
    'neck_rotation': 'Neck Rotation',
    'neck_extension': '首の伸展'
}