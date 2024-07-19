import tensorflow as tf

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = tf.keras.backend.square(y_pred)
    margin_square = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))
    return tf.keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)
