import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pandas as pd
import os

def process_video(participant_name):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # Construct paths
    video_path=f'C:/Users/user/OneDrive/שולחן העבודה/שנה ג/סמסטר ב/brain research work/Results/{participant_name}.mp4'

    excel_folder ="C:/Users/user/OneDrive/שולחן העבודה/שנה ג/סמסטר ב/brain research work/smile detection results"
    
    # Create a FaceLandmarker object
    base_options = python.BaseOptions(model_asset_path="C:/Users/user/Downloads/face_landmarker_v2_with_blendshapes (1).task")
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Function to draw landmarks on the image
    def draw_landmarks_on_image(image, detection_result):
        for face_landmarks in detection_result.face_landmarks:
            for idx, landmark in enumerate(face_landmarks):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        return image

    # Function to extract face blendshapes data
    def extract_face_blendshapes(face_blendshapes):
        blendshape_data = [(blendshape.category_name, blendshape.score) for blendshape in face_blendshapes]
        return blendshape_data

    # Smile detection thresholds
    smile_thresholds = {
        'mouthSmileLeft': 0.2137247813732795,  # Lowered from 0.3
        'mouthSmileRight': 0.20704765056839897, # Lowered from 0.3
        'mouthPressLeft': 0.0017642588948572435, # Lowered from 0.02
        'mouthPressRight': 0.01247548061576947,# Lowered from 0.02
        'eyeSquintLeft': 0.36315783269948043,   # Lowered from 0.2
        'eyeSquintRight': 0.4324113565116444   # Lowered from 0.2
    }


    # Process the video
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    smile_data_frames = []
    smile_data_seconds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks from the input image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        # Process the detection result and extract blendshape data
        if detection_result.face_landmarks:
            blendshape_data = extract_face_blendshapes(detection_result.face_blendshapes[0])
            blendshape_dict = {blendshape[0]: blendshape[1] for blendshape in blendshape_data}
            is_smiling = all(blendshape_dict.get(name, 0) > threshold for name, threshold in smile_thresholds.items())
            if is_smiling:
                smile_data_frames.append([frame_count, is_smiling])
            
            second = frame_count // 25
            if is_smiling and (len(smile_data_seconds) == 0 or smile_data_seconds[-1][0] != second):
                smile_data_seconds.append([second, is_smiling])

        frame_count += 1

    cap.release()

    # Convert accumulated smile data into DataFrames
    df_smile_data_frames = pd.DataFrame(smile_data_frames, columns=['frame', 'smile_detected'])
    df_smile_data_seconds = pd.DataFrame(smile_data_seconds, columns=['second', 'smile_detected'])

    # Save smile data to Excel files
    df_smile_data_frames.to_excel(os.path.join(excel_folder, f'smile_frames_{participant_name}.xlsx'), index=False)
    df_smile_data_seconds.to_excel(os.path.join(excel_folder, f'smile_seconds_{participant_name}.xlsx'), index=False)
    print(f"Smile detection data for {participant_name} saved to Excel.")

# Example usage
process_video('81671316_67_right_standup_4')