import os
import pickle
import numpy as np

from tqdm.notebook import tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.model import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

BASE_DIR = 'F:\kauta marvin\marv\COMPUTER SCIENCE\computer science all yrs\year 3\sem 2\PROJECTS\Flickr8k_Dataset\Flicker8k_Dataset'
WORKING_DIR = 'F:\kauta marvin\marv\COMPUTER SCIENCE\computer science all yrs\year 3\sem 2\PROJECTS\GENVISUAL'

# EXTRACT IMAGE features
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layer[-2].output)
# summarise
print(model.summary)

# extract images
features = {}
directory = os.path.join(BASE_DIR, 'Images')
for img_name in tqdm(os.listdir(directory)):
    # load the image pixels to numpy array
    image = img_to_array(image)
    # reshape image for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #      # reshape image for vgg
    image = preprocess_input(image)
    #     extract features
    feature = model.predict(image, verbose=0)
    #   get image id
    image_id = img_name.split('.')[0]
    #   store feature
    features[image_id] = feature

#   storing features in pickle

pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))

# load featured from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

# load captions data

with open(os.path.join(BASE_DIR, 'caption.txt'), 'r') as f:
    next(f)
    caption_doc = f.read()

# run
caption_doc
# create mapping of image to caption

mapping = {}

# process lines
for line in tqdm(caption_doc.split('\n')):
    # split the line by comma
    token = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    #     remove extensions from image
    image_id = image_id.split('.')[0]
    # convert caption list to sting
    caption = " ".join(caption)
    #     create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    #     store caption
    mapping[image_id].append(caption)

# runto see how may images
len(mapping)


# preprocessing data

def clean(mapping):
    for key, captions in mapping.items():
        # make one cption at a time
        for i in range(len(captions)):
            caption = captions[i]
            # preprocess steps
            # convert to lower case
            caption = caption.lower()
            # delete digits specialchars
            caption = caption.replace('[A-Za-z]', '')

            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = '<start' + "".join([word for word in caption.split() if len(word) > 1]) + ' <end>'
            captions[i] = caption



# before preprosessor

mapping['palce id image here']

# after preprocessing
clean(mapping)

mapping['palce id image here']

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

len(all_captions)

# tokenising the text

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

vocab_size

# get max length of caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length

# train test split
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
train = image_ids[split:]


# create data generator to get data in batch to avaoid syste crash

def data_generator(dataaa_keys, mapping,features,tokenizer,max_length,vocab_size,batch_size):
    # loop over images
    x1,x2, y = list(), list(), list()
    n =0
    while 1:
        n += 1
        captions = mapping[key]
        # process sequence
        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
#             split the sequence in x y pairs
            for i in range(1, len(seq)):
                # split into input and out put pair
                in_seq, out_seq = seq[:1], seq[i]
                # put input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                x1.append(features[key][0])
                x2.append(in_seq)
                y.append(out_seq)
        if n == batch_size:
            x1,x2,y = np.array(x1), np.array(x2), np.array(y)
            yield [x1, x2], y
            x1, x2, y = list(), list(), list()
            n = 0


#Padding sequence normalizes the size of all captions to the max size filling them with zeros for better results

# encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
plot_model(model, show_shapes=True)


# shape=(4096,) - output length of the features from the VGG model
#
# Dense - single dimension linear layer array
#
# Dropout() - used to add regularization to the data, avoiding over fitting & dropping out a fraction of the data from the layers
#
# model.compile() - compilation of the model
#
# loss=’sparse_categorical_crossentropy’ - loss function for category outputs
#
# optimizer=’adam’ - automatically adjust the learning rate for the model over the no. of epochs
#
# Model plot shows the concatenation of the inputs and outputs into a single layer
#
# Feature extraction of image was already done using VGG, no CNN model was needed in this step.


# Now let us train the model

epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# save the model
model.save(WORKING_DIR+'/best_model.h5')


# Generate Captions for the Image

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text

# Caption generator appending all the words for an image
#
# The caption starts with 'startseq' and the model continues
# to predict the caption until the 'endseq' appeared

# Now we validate the data using BLEU Score

from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
    # calcuate BLEU score
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))