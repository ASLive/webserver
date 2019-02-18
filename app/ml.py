from __future__ import print_function, unicode_literals

import tensorflow as tf
import keras
from keras.models import model_from_json
import pickle

#TODO: clean up imports

import tensorflow as tf
import numpy as np
import scipy.misc
from mpl_toolkits.mplot3d import Axes3D

from hand3d.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from hand3d.utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

JSON_PATH = "./bin/model.json"
WEIGHTS_PATH = "./bin/model.h5"
CLASSES_PATH = "./bin/classes.pickle"

# network input
image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
evaluation = tf.placeholder_with_default(True, shape=())

# build network
net = ColorHandPose3DNetwork()
hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

# start tensorflow and initialize network
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
net.init(sess)
def compile(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],)

def read_model():
    # load json and create model
    json_file = open(JSON_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(WEIGHTS_PATH)
    compile(loaded_model)
    return loaded_model

# TODO: bulk process frames?
def image_to_hand(image):
    image_raw = scipy.misc.imresize(image, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                         keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                        feed_dict={image_tf: image_v})

    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)
    return keypoint_coord3d_v

def read_classes():
    return pickle.load( open(CLASSES_PATH,"rb") )

model = read_model()

def predict(hand):
    return model.predict(hand)
