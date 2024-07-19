import numpy as np
import tensorflow as tf
from data_loader import load_data, create_pairs
from model import create_siamese_network

def evaluate():
    input_shape = (100, 100, 3)
    model = create_siamese_network(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights("models/siamese_network.h5")

    data, labels = load_data("data/processed")
    pairs, pair_labels = create_pairs(data, labels)

    results = model.evaluate([pairs[:, 0], pairs[:, 1]], pair_labels)
    print(f"Evaluation results: Loss = {results[0]}, Accuracy = {results[1]}")

if __name__ == "__main__":
    evaluate()
