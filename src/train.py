import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from model import build_siamese_model

# Paths
data_path = "C:/Users/tahas/Desktop/Face Recognition/SIAMESE-Network_Face-recognition/data/CelebA"
processed_images_path = os.path.join(data_path, "processed")

# Load preprocessed images and labels
def load_data(processed_images_path):
    images = []
    labels = []
    for img_file in os.listdir(processed_images_path):
        img_path = os.path.join(processed_images_path, img_file)
        img = load_img(img_path, target_size=(100, 100))
        img = img_to_array(img)
        images.append(img)
        label = int(img_file.split('.')[0])  # Use image_id as label
        labels.append(label)
    return np.array(images), np.array(labels)

# Create pairs of images and labels
def create_pairs(images, labels):
    pairs = []
    pair_labels = []
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    for idx1 in range(len(images)):
        current_image = images[idx1]
        label = labels[idx1]
        idx2 = np.random.choice(class_indices[label])
        positive_image = images[idx2]
        pairs += [[current_image, positive_image]]
        pair_labels += [1]
        negative_label = (label + np.random.randint(1, num_classes)) % num_classes
        idx2 = np.random.choice(class_indices[negative_label])
        negative_image = images[idx2]
        pairs += [[current_image, negative_image]]
        pair_labels += [0]
    return np.array(pairs), np.array(pair_labels)

if __name__ == "__main__":
    # Load and preprocess the data
    images, labels = load_data(processed_images_path)
    pairs, pair_labels = create_pairs(images, labels)
    pairs_train, pairs_val, labels_train, labels_val = train_test_split(pairs, pair_labels, test_size=0.2, random_state=42)
    
    input_shape = (100, 100, 3)
    model = build_siamese_model(input_shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit([pairs_train[:, 0], pairs_train[:, 1]], labels_train,
              validation_data=([pairs_val[:, 0], pairs_val[:, 1]], labels_val),
              batch_size=32,
              epochs=10)
    
    # Save the trained model
    model.save("siamese_model.h5")
