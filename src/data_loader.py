import numpy as np
import os
import cv2

def load_data(dataset_path):
    data = []
    labels = []
    
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))
            data.append(img)
            labels.append(1 if "similar" in filename else 0)
    
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    
    return data, labels

def create_pairs(data, labels):
    pairs = []
    pair_labels = []
    
    num_classes = len(np.unique(labels))
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for idx1 in range(len(data)):
        current_image = data[idx1]
        label = labels[idx1]
        
        idx2 = np.random.choice(digit_indices[label])
        positive_image = data[idx2]
        
        pairs.append([current_image, positive_image])
        pair_labels.append(1)
        
        negative_label = np.random.choice(list(set(range(num_classes)) - {label}))
        idx2 = np.random.choice(digit_indices[negative_label])
        negative_image = data[idx2]
        
        pairs.append([current_image, negative_image])
        pair_labels.append(0)
    
    return np.array(pairs), np.array(pair_labels)

if __name__ == "__main__":
    data, labels = load_data("data/processed")
    pairs, pair_labels = create_pairs(data, labels)
    print(f"Pairs: {pairs.shape}, Labels: {pair_labels.shape}")
