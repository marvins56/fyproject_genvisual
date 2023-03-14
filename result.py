import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array

model = load_model('model_weights.h5')

word_to_index = {}
index_to_word = {}

with open('flickr_8k_dataset.txt', 'r') as f:
    dataset = f.read().split('\n')
    for line in dataset:
        if len(line) > 1:
            tokens = line.split()
            image_id, image_desc = tokens[0], tokens[1:]
            for i in range(len(image_desc)):
                word = image_desc[i]
                if word not in word_to_index:
                    word_to_index[word] = len(word_to_index) + 1
                    index_to_word[len(index_to_word) + 1] = word

vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

img = load_img('Flickr_Data/Flickr_Data/Images/1453366750_6e8cf601bf.jpg', target_size=(224, 224))
img = img_to_array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

features = vgg.predict(img).reshape(1, 7*7, 512)

start_token = 'startseq'
end_token = 'endseq'
max_length = 20

input_text = [word_to_index[start_token]]

for i in range(max_length):
    sequence = np.array(input_text)
    predictions = model.predict([features, sequence])
    predicted_word_index = np.argmax(predictions[0])
    predicted_word = index_to_word[predicted_word_index]
    if predicted_word == end_token:
        break
    input_text.append(predicted_word_index)

generated_caption = ' '.join([index_to_word[i] for i in input_text])
print(generated_caption)
