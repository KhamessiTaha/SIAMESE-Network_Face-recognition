import pandas as pd
import os
import cv2
import dlib

# Paths
data_path = "C:/Users/tahas/Desktop/Face Recognition/SIAMESE-Network_Face-recognition/data/CelebA"
images_path = os.path.join(data_path, "img_align_celeba/img_align_celeba")
landmarks_path = os.path.join(data_path, "list_landmarks_align_celeba.csv")

# Load landmarks
landmarks_df = pd.read_csv(landmarks_path)

# Print the first few rows to verify the column names
print(landmarks_df.head())

# Load the dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Create a directory for processed images
processed_images_path = os.path.join(data_path, "processed")
if not os.path.exists(processed_images_path):
    os.makedirs(processed_images_path)

def preprocess_image(img_path, landmarks):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Align and resize the face image
    # (This is a simplified version; consider using a more sophisticated alignment technique)
    center = ((landmarks[0] + landmarks[2]) // 2, (landmarks[1] + landmarks[3]) // 2)
    angle = cv2.fastAtan2(landmarks[3] - landmarks[1], landmarks[2] - landmarks[0])
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_face = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Crop and resize to 100x100
    cropped_face = aligned_face[center[1] - 50:center[1] + 50, center[0] - 50:center[0] + 50]
    resized_face = cv2.resize(cropped_face, (100, 100))

    return resized_face

def preprocess_celeba():
    # Preprocess and save all images
    for index, row in landmarks_df.iterrows():
        img_file = row['image_id']
        landmarks = [row['lefteye_x'], row['lefteye_y'],
                     row['righteye_x'], row['righteye_y'],
                     row['nose_x'], row['nose_y'],
                     row['leftmouth_x'], row['leftmouth_y'],
                     row['rightmouth_x'], row['rightmouth_y']]
        img_path = os.path.join(images_path, img_file)
        processed_img = preprocess_image(img_path, landmarks)
        if processed_img is not None:
            output_path = os.path.join(processed_images_path, img_file)
            cv2.imwrite(output_path, processed_img)

if __name__ == "__main__":
    preprocess_celeba()
