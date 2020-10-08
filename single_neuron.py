from __future__ import print_function
import numpy as np


#########################
####### FUNCTIONS #######

def sigmoid(x):
    return 1/(1+np.exp(-x))

def compute_layer_before_activation(x, W, b):
    """
    INPUTS
        x : input layer
        W : weights
        b : biases
    OUTPUT : output layer
    """
    return np.matmul(W, x)+b


def get_all_outputs(x, dict_weights, dict_bias):
    """
    computes and stores in a dict the outputs of each layer 
    given initial input, and weights and biases for all the layers
    """
    nb_layers=len(dict_weights)
    dict_Z={-1:x}
    dict_A={-1:x}
    for i in range(nb_layers):
        # predict for each layer
        dict_Z[i]=compute_layer_before_activation(dict_Z[i-1], dict_weights[i], dict_bias[i])
        dict_A[i]=sigmoid(dict_Z[i])     
    return dict_Z, dict_A

def predict(x, dict_weights, dict_bias):
    """
    Given initial input, and weights and biases for all the layers,
    returns the prediction, which is the round output of the last layer
    """
    dict_Z, dict_A=get_all_outputs(x, dict_weights, dict_bias)
    return np.round(dict_A[len(dict_A)-2]) # because index starts at -1
    #argmax(dict_A[len(dict_A)-2], axis=0)


def computeMSE(pred, target):
    # Mean square error
    mse=np.sum(np.square(pred-target))/(pred.size)
    return mse

def L(predictions, targets):
    """
    Loss function : MSE (not binary cross-entropy)
    """
    #return -np.sum(targets*np.log(predictions))/targets.shape[1]
    return computeMSE(predictions,targets)


def training(x, y, dict_weights, dict_bias, lr, iterations):
    """
    Computes prediction, propagation, backpropagation and Update
    returns weights and biases for all the layers
    """
    
    nb_layers=len(dict_weights)
    dict_d_A={} # len: nb_layers
    dict_d_Z={}
    dict_d_W={}
    dict_d_b={}
    acc_train=[]
    acc_val=[]
    loss_train=[]
    loss_val=[]

    x_train, y_train, x_val, y_val=split_training_val(x, y)
    np_input=x_train.shape[1]

    for epoch in range (iterations):
        # Prediction: last is y_hat, others are A
        dict_Z, dict_A=get_all_outputs(x_train, dict_weights, dict_bias)
        
        # Propagation
        y_hat_train=dict_A[len(dict_A)-2] # because index starts at -1
        if epoch>0:
            loss_train.append(np.round(L(y_hat_train, y_train), 4))
            acc_train.append(np.round(compute_accuracy(np.round(y_hat_train), y_train), 4))

        # Back propagation
        for i in range(nb_layers-1, -1, -1):
            if i==nb_layers-1:
                dict_d_A[i]=y_hat_train-y_train #d_y_hat=y_hat-y
                dict_d_W[i]=np.matmul(dict_d_A[i],dict_A[i-1].T)/np_input                
                dict_d_b[i]=np.sum(dict_d_A[i])/np_input
            else:
                dict_d_A[i]=np.matmul(dict_weights[i+1].T, dict_d_A[i+1])
                dict_d_Z[i]=dict_d_A[i]*sigmoid(dict_Z[i])*(1-sigmoid(dict_Z[i]))
                #dict_d_W[i]=np.matmul(dict_d_Z[i],dict_A[i-1].T)/np_input
                dict_d_W[i]=np.matmul(dict_d_Z[i],dict_A[i-1].T)/np_input
                dict_d_b[i]=np.sum(dict_d_Z[i])/np_input
             
        # Update
        for i in range(len(dict_weights)):
            dict_weights[i]=dict_weights[i]-lr*dict_d_W[i]
            dict_bias[i]=dict_bias[i]-lr*dict_d_b[i]
        
        y_hat_val=predict(x_val, dict_weights, dict_bias)
        loss_val.append(np.round(L(y_hat_val, y_val), 4))
        acc_val.append(np.round(compute_accuracy(y_hat_val, y_val), 4))

        if epoch>0:
            print ("Epoch : "+str(epoch)+" --> loss : "+str(loss_train[epoch-1])+", acc : "+str(acc_val[epoch-1])+", val_loss : "+str(loss_val[epoch-1])+", val_acc : "+str(acc_val[epoch-1]))
    
    y_hat_train=predict(x_train, dict_weights, dict_bias)
    loss_train.append(np.round(L(y_hat_train, y_train), 4))
    acc_train.append(np.round(compute_accuracy(y_hat_train, y_train), 4))

    print ("Epoch : "+str(iterations)+" --> loss : "+str(loss_train[iterations-1])+", acc : "+str(acc_val[iterations-1])+", val_loss : "+str(loss_val[iterations-1])+", val_acc : "+str(acc_val[iterations-1]))

    return dict_weights, dict_bias, loss_train, acc_train, loss_val, acc_val



def initialization(nb_neurons_per_layer, input_layer):
    """
    Initializes the weights and biases for each layer:
    weights : Gaussian distrib
    biases : 0
    Returns dict_weights, dict_bias
    """
    layer_input_size=input_layer.shape[0]
    dict_weights={}
    dict_bias={}
    for i in range(len(nb_neurons_per_layer)):
        dict_weights[i]=np.random.randn(nb_neurons_per_layer[i], layer_input_size)/100        
        dict_bias[i]=np.zeros((nb_neurons_per_layer[i], 1)) # [0] * nb_neurons_per_layer[i]
        layer_input_size=nb_neurons_per_layer[i]
    return dict_weights, dict_bias

def split_training_val(x, y):
    slicing_index=int(np.round(x.shape[1]*0.7))
    x_train=x[:, :slicing_index]
    y_train=y[:, :slicing_index]
    x_val=x[:, slicing_index:]
    y_val=y[:, slicing_index:]
    return x_train, y_train, x_val, y_val

def compute_accuracy(y_hat, y):
    return np.count_nonzero(y_hat==y)/y_hat.shape[1]


####### FUNCTIONS #######
#########################



#In this first part, we just prepare our data (mnist) 
#for training and testing

#import keras
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

"""
#Display one image and corresponding label 
import matplotlib
import matplotlib.pyplot as plt
i = 3
print('y[{}]={}'.format(i, y_train[:,i]))
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
"""

#Let start our work: creating a neural network
#First, we just use a single neuron. 


#####TO COMPLETE
part=0

while part<1 or part>2 :
    part = int(input("Choose method :\n1 for single neuron\n2 for one hidden layer\nChosen method : "))

if part==1:
    print("Method : SINGLE NEURON")
    nb_neurons_per_layer=[1]
else:
    print("Method : ONE HIDDEN LAYER")
    nb_neurons_per_layer=[64, 1]

nb_iterations=500


# Initialization
dict_weights, dict_bias=initialization(nb_neurons_per_layer, X_train)
#lr=1 #learning_rate
lr=2


dict_weights, dict_bias, loss_train, acc_train, loss_val, acc_val= training(X_train, y_train, dict_weights, dict_bias, lr, nb_iterations)


import matplotlib.pyplot as plt

plt.figure(1)
plt.title('loss_train vs loss_val')
loss_train_plt, = plt.plot(loss_train, "r")
loss_val_plt, = plt.plot(loss_val, "b")
plt.legend([loss_train_plt, loss_val_plt], ["Train loss", "Validation loss"])
plt.savefig("loss_single_neuron_5.png")


plt.figure(2)
plt.title('acc_train vs acc_val')
acc_train_plt, = plt.plot(acc_train, "r")
acc_val_plt, = plt.plot(acc_val, "b")
plt.legend([acc_train_plt, acc_val_plt], ["Train accuracy", "Validation accuracy"])
plt.savefig("lacc_single_neuron_5.png")

plt.show()


y_test_hat=predict(X_test, dict_weights, dict_bias)
print("-- ACCURACY ON TEST --")
print(np.count_nonzero(y_test_hat==y_test)/y_test.shape[1])
