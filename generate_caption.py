# import os
#
# from PIL import Image
# import matplotlib.pyplot as plt
# from keras import Model
# from keras.applications import VGG16
# from keras.applications.convnext import preprocess_input
# from keras.utils import load_img, img_to_array
#
# from caption import BASE_DIR, mapping, predict_caption, model, features, tokenizer, max_length
#
# # 
# def generate_caption(image_name):
#     # load the image
#     # image_name = "1001773457_577c3a7d70.jpg"
#     image_id = image_name.split('.')[0]
#     img_path = os.path.join(BASE_DIR, "Images", image_name)
#     image = Image.open(img_path)
#     captions = mapping[image_id]
#     print('---------------------Actual---------------------')
#     for caption in captions:
#         print(caption)
#     # predict the caption
#     y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
#     print('--------------------Predicted--------------------')
#     print(y_pred)
#     plt.imshow(image)
#
# generate_caption("1001773457_577c3a7d70.jpg")
#
#
# # testing with real image
#
# vgg_model = VGG16()
# # restructure the model
# vgg_model = Model(inputs=vgg_model.inputs,
#                   outputs=vgg_model.layers[-2].output)
#
# image_path = '/kaggle/input/flickr8k/Images/1000268201_693b08cb0e.jpg'
# # load image
# image = load_img(image_path, target_size=(224, 224))
# # convert image pixels to numpy array
# image = img_to_array(image)
# # reshape data for model
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# # preprocess image from vgg
# image = preprocess_input(image)
# # extract features
# feature = vgg_model.predict(image, verbose=0)
# # predict from the trained model
# predict_caption(model, feature, tokenizer, max_length)