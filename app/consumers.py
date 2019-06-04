from __future__ import print_function, unicode_literals
from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
import json
import sys
import uuid
import base64
from datetime import datetime
import cv2
from contextlib import closing
import math

import numpy as np
import keras
from keras.models import model_from_json
import pickle
import numpy as np
import scipy.misc
from hand3d.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from hand3d.utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d
import tensorflow as tf
global graph, model

from tqdm import tqdm
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from .data import load_data

TEST_MODE = False; # for use without mobile client

# MNIST Dataset
# (train_images, train_labels),(test_images, test_labels) = mnist.load_data(path='sign_mnist_train.csv')
(train_images, train_labels),(test_images, test_labels), class_names = load_data()

print(str(class_names))

# d = {}
#
# for i in range(len(train_labels)):
#     d[train_labels[i]] = train_images[i]
#
# for key in d.keys():
#     plt.imshow(d[key])
#
# plt.show()
# exit()

train_images  = np.expand_dims(train_images.astype(np.float32) / 255.0, axis=3)
test_images = np.expand_dims(test_images.astype(np.float32) / 255.0, axis=3)
train_labels = to_categorical(train_labels)

# Training parameters
batch_size = 128
n_epochs = 25
n_classes = 26
learning_rate = 1e-4

# 2D Convolutional Function
def conv2d(x, W, b, strides=1):
    x = tf.cast(x, tf.float32)
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Define Weights and Biases
weights = {
    # Convolution Layers
    'c1': tf.get_variable('W1', shape=(3,3,1,16), \
            initializer=tf.contrib.layers.xavier_initializer()),
    'c2': tf.get_variable('W2', shape=(3,3,16,16), \
            initializer=tf.contrib.layers.xavier_initializer()),
    'c3': tf.get_variable('W3', shape=(3,3,16,32), \
            initializer=tf.contrib.layers.xavier_initializer()),
    'c4': tf.get_variable('W4', shape=(3,3,32,32), \
            initializer=tf.contrib.layers.xavier_initializer()),

    # Dense Layers
    'd1': tf.get_variable('W5', shape=(7*7*32,128),
            initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128,n_classes),
            initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    # Convolution Layers
    'c1': tf.get_variable('B1', shape=(16), initializer=tf.zeros_initializer()),
    'c2': tf.get_variable('B2', shape=(16), initializer=tf.zeros_initializer()),
    'c3': tf.get_variable('B3', shape=(32), initializer=tf.zeros_initializer()),
    'c4': tf.get_variable('B4', shape=(32), initializer=tf.zeros_initializer()),

    # Dense Layers
    'd1': tf.get_variable('B5', shape=(128), initializer=tf.zeros_initializer()),
    'out': tf.get_variable('B6', shape=(n_classes), initializer=tf.zeros_initializer()),
}

# Model Function
def conv_net(data, weights, biases, training=False):
    # Convolution layers
    conv1 = conv2d(data, weights['c1'], biases['c1']) # [28,28,16]
    conv2 = conv2d(conv1, weights['c2'], biases['c2']) # [28,28,16]
    pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # [14,14,16]

    conv3 = conv2d(pool1, weights['c3'], biases['c3']) # [14,14,32]
    conv4 = conv2d(conv3, weights['c4'], biases['c4']) # [14,14,32]
    pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # [7,7,32]

    # Flatten
    flat = tf.reshape(pool2, [-1, weights['d1'].get_shape().as_list()[0]])
    # [7*7*32] = [1568]

    # Fully connected layer
    fc1 = tf.add(tf.matmul(flat, weights['d1']), biases['d1']) # [128]
    fc1 = tf.nn.relu(fc1) # [128]

    # Dropout
    if training:
        fc1 = tf.nn.dropout(fc1, rate=0.2)

    # Output
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out']) # [10]
    return out

# Dataflow Graph
dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).repeat().batch(batch_size)
iterator = dataset.make_initializable_iterator()
batch_images, batch_labels = iterator.get_next()
logits = conv_net(batch_images, weights, biases, training=True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=batch_labels))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
test_predictions = tf.nn.softmax(conv_net(test_images, weights, biases))
acc,acc_op = tf.metrics.accuracy(predictions=tf.argmax(test_predictions,1), labels=test_labels)

# Run Session
#with tf.Session() as sess:
#    # Initialize Variables
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
#    sess.run(iterator.initializer)
#
#    # Train the Model
#    for epoch in range(n_epochs):
#        prog_bar = tqdm(range(int(len(train_images)/batch_size)))
#        for step in prog_bar:
#            _,cost = sess.run([train_op,loss])
#            prog_bar.set_description("cost: {:.3f}".format(cost))
#        accuracy = sess.run(acc_op)
#
#        print('\nEpoch {} Accuracy: {:.3f}'.format(epoch+1, accuracy))
#
#    # Show Sample Predictions
#    # predictions = sess.run(tf.argmax(conv_net(test_images[:25], weights, biases), axis=1))
#    # f, axarr = plt.subplots(5, 5, figsize=(25,25))
#    # for idx in range(25):
#    #     axarr[int(idx/5), idx%5].imshow(np.squeeze(test_images[idx]), cmap='gray')
#    #     axarr[int(idx/5), idx%5].set_title(str(predictions[idx]),fontsize=50)
#    #
#    # Save Model
#    saver = tf.train.Saver()
#    saver.save(sess, './model.ckpt')

graph = tf.get_default_graph()
# JSON_PATH = "./bin/model.json"
# WEIGHTS_PATH = "./bin/model.h5"
# CLASSES_PATH = "./bin/classes.pickle"

# def read_model():
#     # load json into new model
#     json_file = open(JSON_PATH, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights(WEIGHTS_PATH)
#     # compile model
#     loaded_model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'],)
#     return loaded_model
#
# model = read_model()
# classes = pickle.load( open(CLASSES_PATH,"rb") )

class Translator():

    image_tf = None
    hand_side_tf = None
    evaluation = None
    net = None
    hand_scoremap_tf = None
    image_crop_tf = None
    scale_tf = None
    center_tf = None
    keypoints_scoremap_tf = None
    keypoint_coord3d_tf = None
    sess = None

    def __init__(self):

        print('Initializing hand3d')

        # network input
        self.image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
        self.hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
        self.evaluation = tf.placeholder_with_default(True, shape=())

        # build network
        self.net = ColorHandPose3DNetwork()
        self.hand_scoremap_tf, self.image_crop_tf, self.scale_tf, self.center_tf,\
        self.keypoints_scoremap_tf, self.keypoint_coord3d_tf = self.net.inference(self.image_tf, self.hand_side_tf, self.evaluation)

        # start tensorflow and initialize network
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.net.init(self.sess)

    # TODO: bulk process frames?
    def image_to_hand(self, image):
        """convert an image frame into hand vector with hand3d"""
        # pre processing

        cv2.imwrite("imdir/raw_frame.jpg", image)
        # y, x = image.shape
        # image = image[0:int(y*0.6),0:x]

        # cv2.imwrite("imdir/raw_crop.jpg", image)
        image_raw = scipy.misc.imresize(image, (240, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

        # prediction
        input = [self.hand_scoremap_tf, self.image_crop_tf,
                self.scale_tf, self.center_tf,
                self.keypoints_scoremap_tf, self.keypoint_coord3d_tf]
        hand_scoremap_v, image_crop_v, scale_v, center_v,\
        keypoints_scoremap_v, keypoint_coord3d_v = self.sess.run(input,feed_dict={self.image_tf: image_v})

        def rgb2gray(rgb):
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            gray = 0.3333 * r + 0.3333 * g + 0.3333 * b
            return gray

        # post processing
        # keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)
        # hand_vector = np.expand_dims(keypoint_coord3d_v,0)
        image_crop_v = np.squeeze(image_crop_v)
        image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        cv2.imwrite("imdir/cropped.jpg", image_crop_v)
        image = scipy.misc.imresize(image_crop_v, (28, 28))
        image = rgb2gray(image)
        cv2.imwrite("imdir/grayscale_cropped.jpg", image)
        image = np.expand_dims((image.astype('float') / 255.0) - 0.5, 3)

        return image

    def letter_predict(self, hand):
        """convert hand vector data to asl letter"""
        ret = None
        with graph.as_default():
            ret = model.predict(hand)

        return ret

    def process(self,frame):
        """
        for a given video frame:
            - extract the hand vector data
            - predict the asl letter
        """
#        hand = self.image_to_hand(frame)
        # prediction = self.letter_predict(hand)[0]
        with graph.as_default():
            predictions = sess.run(tf.argmax(conv_net(np.array([frame]), weights, biases), axis=1))
#        self.client.send(text_data=json.dumps(
#            {'translation': str(predictions)}
#        ))
        print('class: ' + str(predictions[0]))
        print('predicted letter: ' + str(class_names[predictions[0]]))
        print('\n')
        return class_names[predictions[0]]
        # predicted_label = np.argmax(prediction)
        # predicted_letter = classes[predicted_label]
        # confidence_threshold = 0.75;
        # confidence = prediction[predicted_label]
        # # TODO: stop sending if client disconnects
        # print(str(confidence)+" : "+predicted_letter)
        # if confidence > confidence_threshold:
            # self.client.send(text_data=json.dumps(
            #     {'translation': predicted_letter}
            # ))


with graph.as_default():
    sess = tf.Session(graph=graph)
    # Run Session
    # with tf.Session() as sess:
        # Initialize Variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer)

    # Train the Model
    for epoch in range(n_epochs):
        prog_bar = tqdm(range(int(len(train_images)/batch_size)))
        for step in prog_bar:
            _,cost = sess.run([train_op,loss])
            prog_bar.set_description("cost: {:.3f}".format(cost))
        accuracy = sess.run(acc_op)

        print('\nEpoch {} Accuracy: {:.3f}'.format(epoch+1, accuracy))


trans = Translator()
# image = scipy.misc.imread('./carlsen.jpg')
# hand = trans.image_to_hand(image)
# trans.process(np.array(hand))
#    a = np.array(test_images[0])
#    print("good:")
#    print(a.shape)
#    print("bad:")
#    print(b.shape)

    # Show Sample Predictions
    # predictions = sess.run(tf.argmax(conv_net(test_images[:25], weights, biases), axis=1))
    # f, axarr = plt.subplots(5, 5, figsize=(25,25))
    # for idx in range(25):
    #     axarr[int(idx/5), idx%5].imshow(np.squeeze(test_images[idx]), cmap='gray')
    #     axarr[int(idx/5), idx%5].set_title(str(predictions[idx]),fontsize=50)
    #
    # Save Model
#    saver = tf.train.Saver()
#    saver.save(sess, './model.ckpt')


# TODO: get frames without saving video file for speed
def get_frames(video_file):
    frames = []
    vidcap = cv2.VideoCapture(video_file)
    success = True
    count = 0
    while success:
      success, image = vidcap.read()
      if success:
          h, w = image.shape[:2]
          img_c = (w / 2, h / 2)

          rot = cv2.getRotationMatrix2D(img_c, 270, 1)

          rad = math.radians(270)
          sin = math.sin(rad)
          cos = math.cos(rad)
          b_w = int((h * abs(sin)) + (w * abs(cos)))
          b_h = int((h * abs(cos)) + (w * abs(sin)))

          rot[0, 2] += ((b_w / 2) - img_c[0])
          rot[1, 2] += ((b_h / 2) - img_c[1])

          image = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
          cv2.imwrite("frames/frame_%d.jpg" % count, image)     # save frame as JPEG file
          frames.append(image)
      count += 1

    return frames

class AslConsumer(WebsocketConsumer):

    def connect(self):

        self.client_name = 'client_%s' % uuid.uuid4()
        # self.translator = translate.Translator(self)
        async_to_sync(self.channel_layer.group_add)(self.client_name,self.channel_name)
        self.accept()

    def disconnect(self, close_code):
        self.translator = None
        async_to_sync(self.channel_layer.group_discard)(self.client_name,self.channel_name)

    def receive(self, text_data):
        # TODO: use logging
        print('receiving video @ ' + str(datetime.now().time()))

        # decode video data from base64
        # remove leading padding header (data:video/mp4;base64,)
        # TODO: find a safer way to strip header
        if not TEST_MODE:
            decoded_string = base64.b64decode(text_data[22:])

        # write video data to mp4 file
        video_file_path = 'test.mp4'
        if not TEST_MODE:
            f = open(video_file_path,'wb')
            f.write(decoded_string)
            f.close()

        # get video frames
        video_frames = get_frames(video_file_path)

        # process frame with ML
        frame_step = 15
        counter = 0
        for frame in video_frames[0::frame_step]:
            print('\nprocessing frame number ' + str(counter))
            # self.translator.process(frame)
            hand = trans.image_to_hand(frame)
            ret = trans.process(np.array(hand))
            self.send(text_data=json.dumps(
                {'translation': ret}
            ))
            counter += 15

        print('processed request @ ' + str(datetime.now().time()))
