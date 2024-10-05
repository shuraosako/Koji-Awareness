from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
import traceback
import pandas as pd
import time
from components.pose_estimations import PoseEstimator, available_poses

app = Flask(__name__)

# YOLOv8モデルをロード
model = YOLO('yolov8n-pose.pt')

# カメラキャプチャの設定
camera = cv2.VideoCapture(0)

# FPSを30に設定
FPS = 30
FRAME_INTERVAL = 1.0 / FPS

# セッションデータを保存するグローバル変数
session_data = []

# PoseEstimatorのインスタンスを作成
pose_estimator = PoseEstimator()

# 選択された姿勢推定メソッド
selected_pose = 'right_hand_raised'

def generate_frames():
    global session_data, selected_pose
    last_frame_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        if elapsed_time >= FRAME_INTERVAL:
            success, frame = camera.read()
            if not success:
                break
            else:
                # YOLOv8による推論を実行
                results = model(frame)
                
                # 結果を描画
                annotated_frame = results[0].plot()
                
                # 骨格データを抽出し整理
                skeleton_data = []
                timestamp = datetime.now().timestamp()
                stop_recording = False
                for result in results:
                    for pose in result.keypoints.data:
                        organized_data = pose_estimator.organize_skeleton_data(pose.tolist(), timestamp)
                        skeleton_data.append(organized_data)
                        
                        # 選択された姿勢推定メソッドを実行
                        if selected_pose == 'right_hand_raised':
                            stop_recording = pose_estimator.check_pose(organized_data, current_time)
                        elif selected_pose == 't_pose':
                            stop_recording = pose_estimator.is_t_pose(organized_data['keypoints'])
                
                if stop_recording:
                    print(f"Detected {available_poses[selected_pose]}. Stopping video.")
                    yield (b'--frame\r\n'
                           b'Content-Type: text/plain\r\n\r\n'
                           b'COMPLETE\r\n')
                    break
                
                # セッションデータに追加
                session_data.extend(skeleton_data)
                
                # JPEGにエンコード
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
                
                last_frame_time = current_time
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(max(0, FRAME_INTERVAL - elapsed_time))

@app.route('/')
def index():
    return render_template('index.html', poses=available_poses)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_pose', methods=['POST'])
def set_pose():
    global selected_pose
    selected_pose = request.form.get('pose')
    return jsonify({'message': f'Pose set to {available_poses[selected_pose]}'})

@app.route('/stop_and_save', methods=['POST'])
def stop_and_save():
    global session_data
    try:
        if not session_data:
            return jsonify({'message': 'No data to save. Please start the video feed first.'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # ファイルを保存するディレクトリを指定
        save_dir = r'C:\Users\81809\Documents\学校\適当\yolotest'
        
        print(f"Session data length: {len(session_data)}")
        print(f"First item in session data: {session_data[0] if session_data else 'No data'}")
        
        # データを整理
        organized_data = []
        for item in session_data:
            keypoints = item['keypoints']
            keypoint_dict = {kp['name']: (kp['x'], kp['y']) for kp in keypoints if kp['confidence'] > 0.5}
            organized_data.append({'timestamp': item['timestamp'], **keypoint_dict})
        
        # DataFrameに変換
        df = pd.DataFrame(organized_data)
        
        if df.empty:
            return jsonify({'message': 'No valid data to save.'}), 400
        
        # タイムスタンプでソート
        df = df.sort_values('timestamp')
        
        # X座標とY座標を別々の列に分割
        for column in df.columns:
            if column != 'timestamp':
                df[f'{column}_x'] = df[column].apply(lambda x: x[0] if isinstance(x, tuple) else None)
                df[f'{column}_y'] = df[column].apply(lambda x: x[1] if isinstance(x, tuple) else None)
                df = df.drop(column, axis=1)
        
        # 列を並び替え
        columns = ['timestamp'] + sorted([col for col in df.columns if col != 'timestamp'])
        df = df[columns]
        
        # Excelファイルとして保存
        excel_filename = f'keypoints_data_{timestamp}.xlsx'
        excel_path = os.path.join(save_dir, excel_filename)
        df.to_excel(excel_path, index=False)
        
        # セッションデータをリセット
        session_data = []
        
        print(f"Session data saved successfully as {excel_path}")
        return jsonify({'message': f'Session data saved as {excel_filename}'})
    except Exception as e:
        error_msg = f"Error saving session data: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)