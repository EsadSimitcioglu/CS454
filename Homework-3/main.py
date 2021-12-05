import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SLP:
    def __init__(self):
        self.labels = []
        self.X = []
        self.N = []
        self.y_matrix = []
        self.w_matrix = []
        self.w_matrix_best = []
        self.temp_w = []


    def arrange_props(self,matrix,K_length):
        self.labels = matrix.iloc[:, 0]  # Only the first column (just class)
        self.X = matrix.iloc[:, 1:] / 255  # All rows and all column except first column
        self.N = matrix.shape[0]  # Number of rows in the dataset
        self.y_matrix = np.zeros((self.N, K_length))  # Y matrix for training set
        self.w_matrix = np.random.uniform(low=-0.01, high=0.01, size=(D_length + 1, K_length))  # Random matrix for wegihts
        self.w_matrix_best = np.zeros((D_length + 1, K_length))  # Best possible values for matrix for weights

    def forward(self):
        for dataset_row_count in range(self.N):
            row = self.X.iloc[dataset_row_count, :]
            row = np.append(row,1)
            row = np.transpose(row)
            o = np.zeros(K_length)

            for i in range(K_length):
                o[i] = np.dot(self.w_matrix[:,i],row)

            self.y_matrix[dataset_row_count] = softmax(o)

    def forward_and_backward(self):
        temp_w = np.zeros(self.w_matrix.shape)  # Copy w_matrix to the temp matrix
        for dataset_row_count in range(self.N):
            row = self.X.iloc[dataset_row_count, :]
            row = np.append(row,1)
            row = np.transpose(row)
            o = np.zeros(K_length)

            for i in range(K_length):
                o[i] = np.dot(self.w_matrix[:,i],row)

            self.y_matrix[dataset_row_count] = softmax(o)
            one_hot_matrix = one_hot_encode(self.labels.iloc[dataset_row_count])

            for i in range(K_length):
                for j in range(D_length + 1):
                    temp_w[j, i] += (one_hot_matrix[i] - self.y_matrix[dataset_row_count][i]) * row[j]

            for i in range(K_length):
                for j in range(D_length + 1):
                    self.w_matrix[j, i] += temp_w[j, i] * learning_rate

class MLP:
    def __init__(self):
        self.X = []
        self.N = []
        self.y_matrix = []
        self.w_matrix = []
        self.w_matrix_best = []
        self.v_matrix = []
        self.labels = []

    def arrange_props(self, H, matrix):
        self.X = matrix.iloc[:, 1:] / 255
        self.N = matrix.shape[0]
        self.v_matrix = np.random.uniform(-0.01, 0.01, (H + 1, K_length))
        self.w_matrix = np.random.uniform(-0.01, 0.01, (D_length + 1, H))
        self.w_matrix_best = np.zeros((D_length + 1, H))
        self.y_matrix = np.zeros((self.N, K_length))
        self.labels = matrix.iloc[:, 0]

    def forward(self, H, randomList):
        for index in randomList:
            x_t = self.X.iloc[index, :]
            x_t = np.append(x_t, 1)
            x_t = np.transpose(x_t)

            z_h = np.zeros(H)

            for h in range(H):
                z_h[h] = np.dot(np.transpose(self.w_matrix[:, h]), x_t)

            z_h = sigmoid(z_h)
            z_h = np.append(z_h, 1)
            z_h = np.transpose(z_h)

            o = np.zeros(K_length)
            for i in range(K_length):
                o[i] = np.dot(np.transpose(self.v_matrix[:, i]), z_h)

            self.y_matrix[index] = softmax(o)

    def forward_and_backward(self, H, randomList):
        temp_v = np.zeros((H + 1, K_length))
        temp_w = np.zeros((D_length + 1, H))

        for index in randomList:

            x_t = self.X.iloc[index, :]
            x_t = np.append(x_t, 1)
            x_t = np.transpose(x_t)

            z_h = np.zeros(H)

            for h in range(H):
                z_h[h] = np.dot(np.transpose(self.w_matrix[:, h]), x_t)

            z_h = sigmoid(z_h)
            z_h = np.append(z_h, 1)
            z_h = np.transpose(z_h)

            o = np.zeros(K_length)

            for i in range(K_length):
                o[i] = np.dot(np.transpose(self.v_matrix[:, i]), z_h)

            self.y_matrix[index] = softmax(o)

            r_t = self.labels.iloc[index]
            r_t = transform_vector(r_t, K_length)

            for i in range(K_length):
                temp_v[:, i] = learning_rate * ((r_t[i] - self.y_matrix[index][i])) * z_h

            for h in range(H):
                error = 0.0
                for i in range(K_length):
                    error += (r_t[i] - self.y_matrix[index][i]) * self.v_matrix[h][i]
                temp_w[:, h] = learning_rate * error * z_h[h] * (1 - z_h[h]) * x_t

            for i in range(K_length):
                self.v_matrix[:, i] = self.v_matrix[:, i] + temp_v[:, i]

            for h in range(H):
                self.w_matrix[:, h] = self.w_matrix[:, h] + temp_w[:, h]



def plot_mean_images(weights):
    # Creating 43 subplots
    fig, axes = plt.subplots(4, 3)
    # Set the height of plat 8px8px
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.suptitle('Mean Images')
    # For each subplot run the code inside loop
    for label in range(12):
        # If the subplot index is a label (0,1,2...9)
        if label<10:
            axes[label//3][label%3].imshow(weights[:,label].reshape(10,10),)
        # Do not show the axes of subplots
        axes[label//3][label%3].axis('off')
    # Showing the plot
    plt.show()

def find_max(list):
    max = 0
    maxIndex = -1
    for index, i in enumerate(list):
        if i > max:
            max = i
            maxIndex = index
    return maxIndex

def transform_vector(value,K):
    vector = np.zeros(K)
    vector[value-1] = 1
    return vector

def softmax(x_value):
    temp_value = np.exp(x_value - np.max(x_value-1))
    return temp_value / temp_value.sum()

def sigmoid(o):
    return 1. / (1. + np.exp(-o))

def one_hot_encode(y_value):
    return np.transpose(np.eye(10)[y_value])

def accuracy_confusion_matrix(x):
    correct = 0
    total   = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            total += x[i][j]
            if i == j:
                correct += x[i][j]
    return correct/total

learning_rate = 0.1
K_length = 10  # number of classes
D_length = 100 # number of features
N_length = 1000 # number of rows

def Question1():

    singleLayerTrain = SLP()
    singleLayerTest = SLP()

    matrix_train = pd.read_csv('training.csv', header=None,  skiprows=1) # training dataset
    matrix_test = pd.read_csv('testing.csv', header=None,  skiprows=1 ) #testing dataset

    singleLayerTrain.arrange_props(matrix_train,K_length)
    singleLayerTest.arrange_props(matrix_test,K_length)

    confusion_matrix_train = np.zeros((K_length, K_length))  # Confusion matrix for training set with zeroes
    confusion_matrix_test = np.zeros((K_length, K_length))  # Confusion matrix for testing set with zeroes

    accuracy_best = 0

    for epoch in range(50):

        confusion_matrix_train = np.zeros((K_length, K_length))  # Confusion matrix for training set with zeroes
        confusion_matrix_test = np.zeros((K_length, K_length))  # Confusion matrix for testing set with zeroes

        #Training
        singleLayerTrain.forward_and_backward()
        singleLayerTrain.forward()

        #Testing
        for dataset_row_count in range(singleLayerTest.N):
            x_t_test = singleLayerTest.X.iloc[dataset_row_count,:]
            x_t_test = np.append(x_t_test, 1)
            x_t_test = np.transpose(x_t_test)

            o = np.zeros(K_length)

            for i in range(K_length):
                o[i] = np.dot((singleLayerTrain.w_matrix[:, i]), x_t_test)

            singleLayerTest.y_matrix[dataset_row_count] = softmax(o)

        #Confusion Matrix
        for dataset_row_count in range(singleLayerTrain.N):
            trainMax = find_max(singleLayerTrain.y_matrix[dataset_row_count])
            testMax = find_max(singleLayerTest.y_matrix[dataset_row_count])

            confusion_matrix_train[singleLayerTrain.labels.iloc[dataset_row_count] - 1, trainMax] += 1
            confusion_matrix_test[singleLayerTest.labels.iloc[dataset_row_count] - 1, testMax] += 1

        #Accuracy Calculation
        accuracy_training = accuracy_confusion_matrix(confusion_matrix_train)
        accuracy_test = accuracy_confusion_matrix(confusion_matrix_test)

        #Result
        print("Epoch: ", epoch + 1, " | Training Accuracy: ", accuracy_training, " | Testing Accuracy: ", accuracy_test)

        #Accuracy Comparision
        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            singleLayerTrain.w_matrix_best = singleLayerTrain.w_matrix

    print(confusion_matrix_train)
    print(confusion_matrix_test)

    #Plot
    plot_mean_images(singleLayerTrain.w_matrix_best[:-1, :])


def Question2():
    matrix_train = pd.read_csv('training.csv', header=None, skiprows=1)  # training dataset
    matrix_test = pd.read_csv('testing.csv', header=None, skiprows=1)  # testing dataset

    for H in [5, 10, 25, 50, 75]:

        multiLayerTrain = MLP()
        multiLayerTest = MLP()

        multiLayerTrain.arrange_props(H, matrix_train)
        multiLayerTest.arrange_props(H, matrix_test)

        confusion_matrix_train = np.zeros((K_length, K_length))  # Confusion matrix for training set with zeroes
        confusion_matrix_test = np.zeros((K_length, K_length))  # Confusion matrix for testing set with zeroes

        accuracy_best = 0.0
        for epoch in range(50):

            randomList = np.arange(N_length)
            np.random.shuffle(randomList)

            # Training -> forward and backward
            multiLayerTrain.forward_and_backward(H, randomList)

            # Training -> forward
            multiLayerTrain.forward(H, randomList)

            # Testing -> forward
            for instance in randomList:
                xt = multiLayerTest.X.iloc[instance, :]
                xt = np.append(xt, 1)
                xt = np.transpose(xt)

                z = np.zeros(H)
                for h in range(H):
                    z[h] = np.dot(np.transpose(multiLayerTrain.w_matrix[:, h]), xt)

                z = sigmoid(z)
                z = np.append(z, 1)
                z = np.transpose(z)

                o = np.zeros(K_length)
                for i in range(K_length):
                    o[i] = np.dot(np.transpose(multiLayerTrain.v_matrix[:, i]), z)

                multiLayerTest.y_matrix[instance] = softmax(o)

            # Confusion Matrix
            for dataset_row_count in range(multiLayerTrain.N):
                trainMax = find_max(multiLayerTrain.y_matrix[dataset_row_count])
                testMax = find_max(multiLayerTest.y_matrix[dataset_row_count])

                confusion_matrix_train[multiLayerTrain.labels.iloc[dataset_row_count], trainMax] += 1
                confusion_matrix_test[multiLayerTest.labels.iloc[dataset_row_count], testMax] += 1

            # Accuracy Calculation
            accuracy_training = accuracy_confusion_matrix(confusion_matrix_train)
            accuracy_test = accuracy_confusion_matrix(confusion_matrix_test)

            # Result
            print("Epoch: ", epoch + 1, " | Training Accuracy: ", accuracy_training, " | Testing Accuracy: ",
                  accuracy_test)

            # Accuracy Comparision
            if accuracy_test > accuracy_best:
                accuracy_best = accuracy_test
                multiLayerTrain.w_matrix_best = multiLayerTrain.w_matrix

        print(confusion_matrix_train)
        print(confusion_matrix_test)


#Question1()
Question2()


