import cv2
import numpy as np
import tensorflow as tf
from model import create_siamese_network

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = img.astype("float") / 255.0
    return np.expand_dims(img, axis=0)

def test(image_path1, image_path2):
    input_shape = (100, 100, 3)
    model = create_siamese_network(input_shape)
    model.load_weights("models/siamese_network.h5")

    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)

    prediction = model.predict([img1, img2])
    similarity_score = prediction[0][0]

    print(f"Similarity Score: {similarity_score}")
    if similarity_score > 0.5:
        print("The images are similar.")
    else:
        print("The images are not similar.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python test.py <path_to_image1> <path_to_image2>")
    else:
        image_path1 = sys.argv[1]
        image_path2 = sys.argv[2]
        test(image_path1, image_path2)
