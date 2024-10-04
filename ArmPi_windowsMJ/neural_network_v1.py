#!/usr/bin/env python3
# encoding:utf-8

"""
File name:      neural_network_v1.py
Author:         Michael Johnson
Date:           9/20/2024
Description:    AI robot system neural network object and methods for loading and saving data and models.
"""

# Import library packages.
import os
import tensorflow as tf
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

import common_code_windows_MJ as cc


def plot_image(i, predictions_array, true_label, class_dict, img):
    true_label, img = true_label[i][0], img[i]
    class_dict = list(class_dict.keys())
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # res = cv2.resize(img, dsize=(48, 64), interpolation=cv2.INTER_NEAREST)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto')

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_dict[predicted_label], 100*np.max(predictions_array),
                                         class_dict[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
   true_label = true_label[i][0]
   plt.grid(False)
   plt.xticks(range(3))
   plt.yticks([])
   thisplot = plt.bar(range(3), predictions_array, color="#777777")
   plt.ylim([0, 1])
   predicted_label = np.argmax(predictions_array)

   thisplot[predicted_label].set_color('red')
   thisplot[true_label].set_color('blue')

class AIRobotNeuralNetwork:
    def __init__(self, path):
        """
        Neural network class used for loading and saving data and creating, testing, and saving models.
        :param path: a Path object path of train directory or test directory
        """
        self.img_path = path

        self.dataset = None
        self.labels = None
        self.block_dict = {}

        self.train_set = None
        self.train_labels = None
        self.test_set = None
        self.test_labels = None

        self.model = None

    def load_new_imgs(self):
        """
        Loads images into tensors.
        """
        imgs = []
        block_labels = []
        labels = []
        color_idx = 0

        # Walk through all files in image path, reading and appending them.
        walker = os.walk(self.img_path)
        for root, subdirs, files in walker:
            for f in files:
                if f.endswith('.jpg'):
                    imgs.append(cv2.imread(os.path.join(str(root), f)))

                    # Pull out color from file name and append to block_labels list. Add new colors to block dictionary.
                    color = f.split('_')[1]
                    color = color.lower()
                    block_labels.append(color)
                    if color not in self.block_dict.keys():
                        self.block_dict[color] = color_idx
                        color_idx += 1

        # Create labels_list of block indexes corresponding to block dictionary.
        for bl in block_labels:
            labels.append(self.block_dict[bl])

        self.labels = np.vstack(labels)
        self.dataset = np.stack(imgs)

    def create_train_test_sets(self, train_size=0.8, seed=None):
        """
        Creates training and testing sets at random.
        :param train_size: a float decimal number representing the percentage of data to be reserved for training
        :param seed: an integer used to seed the random number generator for creating train/test splits; use the same integer for repeated results
        """
        self.train_set, self.test_set, self.train_labels, self.test_labels = train_test_split(
            self.dataset, self.labels, train_size=train_size, random_state=seed)

    def save_data(self):
        """
        Saves training and testing sets to disk.
        """
        with open(os.path.join(self.img_path, "train.pickle"), "wb") as save_file:
            pickle.dump((self.train_set, self.train_labels, self.block_dict), save_file)

        with open(os.path.join(self.img_path, "test.pickle"), "wb") as save_file:
            pickle.dump((self.test_set, self.test_labels, self.block_dict), save_file)

    def load_data(self):
        """
        Loads training and testing sets from disk.
        """
        with open(os.path.join(self.img_path, "train.pickle"), "rb") as load_file:
            (self.train_set, self.train_labels, self.block_dict) = pickle.load(load_file)

        with open(os.path.join(self.img_path, "test.pickle"), "rb") as load_file:
            (self.test_set, self.test_labels, self.block_dict) = pickle.load(load_file)

        # TODO: Compare loaded dictionaries to make sure they are identical. Or, make train and test dict separate.

    def make_model(self):
        """
        Makes convolutional neural network model.
        Model layers example pulled from https://medium.com/@chenycy/a-simple-convolutional-neural-network-cnn-classifier-based-on-real-images-084110d52c18.
        """
        print("\nMaking model...")
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Input((480, 640, 3)),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(3)
        # ])

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((480, 640, 3)),
            tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['accuracy'])

    def train_model(self):
        """
        Fits training data to model.
        """
        print("\nTraining model...")
        self.model.fit(self.train_set, self.train_labels, epochs=10)

    def test_model(self):
        """
        Tests trained model with testing set.
        """
        print("\nTesting model...")
        test_loss, test_acc = self.model.evaluate(self.test_set, self.test_labels, verbose=2)

        print('\nTest accuracy:', test_acc)

    def save_model(self):
        """
        Saves trained model to disk.
        """
        print("\nSaving model...")
        with open(os.path.join(self.img_path, "model.pickle"), "wb") as save_file:
            pickle.dump(self.model, save_file)

    def load_model(self):
        """
        Loads trained model from disk.
        """
        print("\nLoading model...")
        with open(os.path.join(self.img_path, "model.pickle"), "rb") as load_file:
            self.model = pickle.load(load_file)

    def make_predictions(self):
        """
        Creates probability distribution of predictions from trained model.
        """
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        return probability_model.predict(self.test_set)


if __name__ == '__main__':
    dev_nn = AIRobotNeuralNetwork(cc.dev_path)

    c = input("Load new data? Y/n: ")
    if c.lower() == 'y':
        dev_nn.load_new_imgs()
        dev_nn.create_train_test_sets()
        dev_nn.save_data()
    else:
        dev_nn.load_data()

    print("Loaded Images:")
    print(dev_nn.train_set.shape)
    print("Loaded Labels:")
    print(dev_nn.train_labels.shape)
    print("Loaded Dictionary:")
    print(dev_nn.block_dict.items())

    c = input("Create new model? Y/n: ")
    if c.lower() == 'y':
        dev_nn.make_model()
        dev_nn.train_model()
        dev_nn.save_model()
    else:
        dev_nn.load_model()

    dev_nn.test_model()

    predictions = dev_nn.make_predictions()

    print("Prediction 0: ", predictions[0])
    print("Arg: ", np.argmax(predictions[0]))
    print("Class: ", dev_nn.test_labels[np.argmax(predictions[0])])

    # i = 0
    # plt.figure(figsize=(16, 6))
    # plt.subplot(1, 2, 1)
    # plot_image(i, predictions[i], dev_nn.test_labels, dev_nn.block_dict, dev_nn.test_set)
    # plt.subplot(1, 2, 2)
    # plot_value_array(i, predictions[i], dev_nn.test_labels)
    # plt.show()

    # Prints the first 15 images alongside their predictions, indicating if they are correct (blue) or incorrect (red).
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(16 * num_cols, 6 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], dev_nn.test_labels, dev_nn.block_dict, dev_nn.test_set)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], dev_nn.test_labels)
    plt.tight_layout()
    plt.show()