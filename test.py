


import tensorflow as tf
import matplotlib.pyplot as plt

def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = img / 255.
    return img

img_arr = load_image_from_path('testimages/96420612_feb18fc6c6.jpg')

noise = tf.random.normal(img_arr.shape)*0.1
noisy_image = (img_arr + noise)
noisy_image = (noisy_image - tf.reduce_min(noisy_image))/(tf.reduce_max(noisy_image) - tf.reduce_min(noisy_image))
noisy_image.shape

# TensorShape([299, 299, 3])

plt.imshow(img_arr)
plt.show()

plt.imshow(noisy_image)
plt.show()


def get_model():
    return get_caption_model()

caption_model = get_model()

def predict():
    captions = []
    pred_caption = generate_caption('testimages/96420612_feb18fc6c6.jpg', caption_model)

    # https://huggingface.co/spaces/Vinayak-14/Image-captioner/tree/main