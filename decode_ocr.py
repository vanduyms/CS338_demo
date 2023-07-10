import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers

characters = {'E', 'C', 'N', 'p', 'J', 'Y', 'M', 'e', 'g', 'b', 'V', 'G', 'P', 't', 'U', '3', 'n', 'L', '0', 'B', 'A', '8', 'F', 'O', '2', 'a', '/', '1', 'c', 'o', 'y', 'v', 'r', '6', 'R', 'D', '5', 'T', '9', 'S', 'l', 'u', '7', '4'}
characters = sorted(list(characters))

char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

img_width, img_height= 224, 64
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
    

max_length = 11
def decode_pred(pred_label):
  # Input length
  input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]

  # CTC decode
  decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0][:,:max_length]

  # Converting numerics back to their character values
  chars = num_to_char(decode)

  # Join all the characters
  texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]

  # Remove the unknown token
  filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]

  return filtered_texts

with keras.utils.custom_object_scope({'CTCLayer': CTCLayer}):
  model = keras.models.load_model('./model/OCR/best_model_new.h5')

ocr_pred_model = keras.Model(
    inputs=model.get_layer("image").input,
    outputs=model.get_layer('dense2').output
)

def OCR(img_path):
  image = Image.open(img_path)
  image_arr = np.array(image)
  rgb_image = tf.image.convert_image_dtype(image_arr, tf.float32)[..., :3]
  grayscale_image = tf.image.rgb_to_grayscale(rgb_image)

  grayscale_image = tf.transpose(grayscale_image, perm=[1, 0, 2])
  resized_image = tf.image.resize(grayscale_image, (224,  64))

  res = ocr_pred_model.predict(tf.expand_dims(resized_image, axis=0))
  label_predict = decode_pred(res)[0]

  print(f"Predicted Label : {label_predict}")

  return label_predict
