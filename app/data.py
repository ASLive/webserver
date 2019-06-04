import os
import numpy as np
import math
import scipy
import pickle

TRAIN_DATA_PATH = "/Users/evanradcliffe/Senior Design/webserver/app/asl-alphabet/asl_alphabet_train"

split_arr = lambda arr: arr[int(len(arr)/7):]

def load_data():
    images, labels, class_names = read_data()
    (train_images, train_labels), (test_images, test_labels) = split_data(images, labels)
    train_images = split_arr(train_images)
    train_labels = split_arr(train_labels)
    test_images = split_arr(test_images)
    test_labels = split_arr(test_labels)
    return (train_images, train_labels), (test_images, test_labels), class_names
    # return [train_images, train_labels, test_images, test_labels, class_names]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.3333 * r + 0.3333 * g + 0.3333 * b
    return gray

def read_hand3d():
    """read data from files (run setup_asl.py to generate)"""
    images = pickle.load( open("./pickle/images.pickle","rb") )
    labels = pickle.load( open("./pickle/labels.pickle","rb") )
    classes = pickle.load( open("./pickle/classes.pickle","rb") )
    return np.array(images), np.array(labels), classes

def read_data():
    """read data from files"""
    print("loading data...",end="")
    ret_images = []
    ret_labels = []
    ret_class_names = []
    count = 0
    for label in list(os.walk(TRAIN_DATA_PATH)): # walk directory
        full_path, image_list = label[0], label[2]
        letter = full_path[len(TRAIN_DATA_PATH)+1:] # get letter class
        if len(letter) > 0:
            # get list of file paths to each image
            image_path_list = [TRAIN_DATA_PATH+"/"+letter+"/"+file for file in image_list]
            ret_class_names.append(letter)
            # print(letter, count)
            print(".",end="")
            if len(image_path_list) > 0:
                # iterate each image
                for i in range(len(image_path_list)):
                    # add image, letter to ret array
                    image = scipy.misc.imread(image_path_list[i])
                    image = scipy.misc.imresize(image, (28, 28))
                    image = rgb2gray(image)
                    # image = np.expand_dims((image.astype('float') / 255.0) - 0.5, 0)
                    ret_images.append(image)
                    ret_labels.append(count)

                count += 1
    print()

    return np.array(ret_images), np.array(ret_labels), ret_class_names

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def split_data(images, labels):
    """split training and testing data"""
    train_percent = 0.7
    count = math.floor(len(images)*train_percent)
    images, labels = unison_shuffled_copies(images, labels)
    train_images, test_images = images[:count], images[count:]
    train_labels, test_labels = labels[:count], labels[count:]
    return (train_images, train_labels), (test_images, test_labels)
