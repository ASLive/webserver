from __future__ import print_function, unicode_literals
import json
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

graph = tf.get_default_graph()
JSON_PATH = "./bin/model.json"
WEIGHTS_PATH = "./bin/model.h5"
CLASSES_PATH = "./bin/classes.pickle"

def read_model():
    # load json into new model
    json_file = open(JSON_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(WEIGHTS_PATH)
    # compile model
    loaded_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],)
    return loaded_model

model = read_model()
classes = pickle.load( open(CLASSES_PATH,"rb") )

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

    def __init__(self,consumer):

        self.client = consumer

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
        image_raw = scipy.misc.imresize(image, (240, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

        # prediction
        input = [self.hand_scoremap_tf, self.image_crop_tf,
                self.scale_tf, self.center_tf,
                self.keypoints_scoremap_tf, self.keypoint_coord3d_tf]
        keypoint_coord3d_v = self.sess.run(input,feed_dict={self.image_tf: image_v})[-1]

        # post processing
        keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)
        hand_vector = np.expand_dims(keypoint_coord3d_v,0)

        return hand_vector

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
        hand = self.image_to_hand(frame)
        prediction = self.letter_predict(hand)[0]
        predicted_label = np.argmax(prediction)
        predicted_letter = classes[predicted_label]
        confidence_threshold = 0.60;
        prediction_confidence = prediction[predicted_label]
        # TODO: stop sending if client disconnects
        print(prediction)
        print(predicted_letter)
        if prediction_confidence > confidence_threshold:
            self.client.send(text_data=json.dumps(
                {'translation': predicted_letter}
            ))
