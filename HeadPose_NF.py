import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation, MaskedAffineAutoregressiveTransform

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Function to extract head pose angles from video
def extract_head_pose_data(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    head_pose_angles = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        img_h, img_w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]  # Assuming only one face is detected

            face_2d = []
            face_3d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 10, 234, 454, 199]:  # Select at least four landmarks
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    z = lm.z * 3000  # Scaling factor for z

                    face_2d.append([x, y])
                    face_3d.append([x, y, z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            if len(face_2d) >= 4 and len(face_3d) >= 4:
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
                if success:
                    rmat, _ = cv2.Rodrigues(rotation_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                    pitch, yaw, roll = angles[0] * 360, angles[1] * 360, angles[2] * 360
                    head_pose_angles.append([pitch, yaw, roll])

    cap.release()
    return np.array(head_pose_angles)

# Define the DIS as the head pose angles
def define_dis(head_pose_angles):
    scaler = StandardScaler()
    head_pose_scaled = scaler.fit_transform(head_pose_angles)
    return head_pose_scaled

# Set up the Normalizing Flow model
def setup_nf_model(input_dim):
    base_distribution = StandardNormal(shape=[input_dim])
    transforms = []

    for _ in range(5):
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=10))

    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_distribution)
    return flow

# Train the Normalizing Flow model
def train_nf_model(flow, data, epochs=100, lr=1e-3):
    optimizer = Adam(flow.parameters(), lr=lr)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    for epoch in range(epochs):
        flow.train()
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=data_tensor).mean()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Extract the DIS for the individual
def extract_dis(flow, data):
    flow.eval()
    with torch.no_grad():
        dis = flow.log_prob(torch.tensor(data, dtype=torch.float32)).numpy()
    return dis

# Example usage
video_path = r"videoPath.mp4"  # Replace with your video path
head_pose_angles = extract_head_pose_data(video_path)
head_pose_scaled = define_dis(head_pose_angles)
input_dim = head_pose_scaled.shape[1]
flow = setup_nf_model(input_dim)
train_nf_model(flow, head_pose_scaled)
dis = extract_dis(flow, head_pose_scaled)

print("Dynamic Identity Signature (DIS) extracted:")
print(dis)
