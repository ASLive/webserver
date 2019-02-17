import tensorflow as tf
import keras
from keras.models import model_from_json
import pickle

JSON_PATH = "./bin/model.json"
WEIGHTS_PATH = "./bin/model.h5"
CLASSES_PATH = "./bin/classes.pickle"

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

def read_classes():
    return pickle.load( open(CLASSES_PATH,"rb") )
