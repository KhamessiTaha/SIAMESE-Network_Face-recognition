import tensorflow as tf
from tensorflow.keras import layers, models

def build_siamese_model(input_shape):
    """
    Builds a Siamese Network model.
    
    Parameters:
    - input_shape: The shape of the input images.
    
    Returns:
    - model: The Siamese Network model.
    """

    # Define the base network
    def base_network(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        return models.Model(inputs, x)

    input_shape = input_shape
    base_model = base_network(input_shape)

    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    # Compute the Euclidean distance between the two processed images
    distance = layers.Lambda(lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    outputs = layers.Dense(1, activation='sigmoid')(distance)

    model = models.Model([input_a, input_b], outputs)
    return model

if __name__ == "__main__":
    input_shape = (100, 100, 3)
    model = build_siamese_model(input_shape)
    model.summary()
