from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#In this first part, we just prepare our data (mnist) 
#for training and testing
import sys
#import keras
from tensorflow.keras.datasets import mnist

#load the MNIST dataset from tensorflow
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
X_test  = X_test / 255    # avoid the weights getting too big


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


# #Display one image and corresponding label 
'''
i = 3
print('y[{}]={}'.format(i, y_train[:,i]))
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
'''
#user input the number of iterations,if no input, set defualt number 500, learning rate 0.7
if len(sys.argv) < 2:
    iterations = 500
    lr = 0.7

#if there is only one input, check if it's iteration number or learning rate 
elif len(sys.argv) < 3:
    if float(sys.argv[1]) <10:
        lr = sys.argv[1]
        iterations = 500
    else:
        iterations = int(sys.argv[1])
        lr = 0.7

else: 
    if float(sys.argv[1])>10: #check if the first input is iteration number
        iterations = int(sys.argv[1])
        lr = float(sys.argv[2])
    else:
        lr = float(sys.argv[1])
        iterations = int(sys.argv[2])
        

class NeuralNetwork:
    def __init__(self, x_train, y_train, iterations,lr):
        #read the whole dataset into class
        self.input = x_train
        self.y     = y_train

        # initialize four parameters
        self.w1   = np.random.normal(0.01, 1, (64,self.input.shape[0])) *0.01
        self.w2   = np.random.normal(0.01, 1, (1,64)) *0.01
        self.b1   = np.random.normal(0.01, 1, (64,1)) *0.01
        self.b2   = 0.01

        self.epoch = iterations
        self.lr = lr     

    def feedforward(self):
        self.Z1 = np.dot(self.w1, self.X_train_to_go) + self.b1 #Z1 = W1*X + b1
        self.A1 = self.get_sigmoid(self.Z1,'false')             #A1 = sigmoid(Z1)
        self.Z2 = np.dot(self.w2,self.A1) + self.b2             #Z2 = W2*A1 +b2
        self.A2 = self.get_sigmoid(self.Z2,'false')             #A2 = sigmoid(Z2)
        loss = self.get_loss(self.A2, self.y_train_to_go,'false')
        return loss

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        self.dj_dA2 = self.get_loss(self.A2,self.y_train_to_go,'true')
        self.dA2_dZ2 = self.get_sigmoid(self.Z2,'true')
        self.dZ2_dw2 = self.A1
        self.delta2 = self.dj_dA2*self.dA2_dZ2
        self.dj_dw2 = (1.0/self.m)*np.matmul(self.delta2,np.transpose(self.dZ2_dw2)) #the derivative of w2 

        self.dZ2_dA1 = self.w2
        self.dA1_dZ1 = self.get_sigmoid(self.Z1,'true')
        self.dZ1_dw1 = self.X_train_to_go
        self.delta1 = np.matmul(np.transpose(self.dZ2_dA1),self.delta2)*self.dA1_dZ1
        self.dj_dw1 = (1.0/self.m)*np.matmul(self.delta1, self.dZ1_dw1.transpose()) #THE derivative of w1


        self.dj_db2 = (1.0/self.m)*np.sum(self.delta2) # the derivative of b2
        self.dj_db1 = (1.0/self.m)*np.sum(self.delta1,axis=1).reshape((64,1)) #the derivative of b1

        '''
        self.dZ2 = self.A2 - self.y_train_to_go
        self.dw2 = (1.0/self.m)*np.matmul(self.dZ2,self.A1.transpost())
        self.b2 = (1.0/self.m)*np.sum(self.dZ2)

        self.dZ1 = self.A1 - 
        self.dw1 = 
        '''

        
        # update the weights with the derivative (slope) of the loss function
        self.w1 -= lr*self.dj_dw1
        self.w2 -= lr*self.dj_dw2
        self.b1 -= lr*self.dj_db1
        self.b2 -= lr*self.dj_db2
        return self.w1, self.w2, self.b1, self.b2

    #Let start our work: creating a neural network
    #First, we just use a single neuron. 

    def get_sigmoid(self, Z, backward):
        sig = 1/(1 + np.exp(-Z))

        if backward == 'false': # if not backward
            return sig
        else:
            return sig*(1-sig) 

    def get_loss(self, A, y_gt, backward):
        if backward == 'false':
            loss = -(1.0/y_gt.shape[1]) * np.sum(y_gt*np.log(A+1e-8) + (1-y_gt)*np.log(1-A+1e-8))
        else:
            #loss = -(1.0/self.m) * np.sum(y_gt/A - (1-y_gt)/(1-A))
            loss = - np.divide(y_gt,A+1e-8) + np.divide((1-y_gt),(1-A)+1e-8)
        return loss

    #####TO COMPLETE
    def training(self):
        #Now, we shuffle the training set
        samples_num = self.input.shape[1] #number of examples

        np.random.seed(138)
        shuffle_index = np.random.permutation(samples_num)
        X_train, y_train = self.input[:,shuffle_index], self.y[:,shuffle_index]

        X_train_to_go, y_train_to_go = self.input[:,:int(2*samples_num/3)], self.y[:,:int(2*samples_num/3)]
        np.random.seed(138)
        shuffle_index_1 = np.random.permutation(int(2*samples_num/3))
        self.X_train_to_go, self.y_train_to_go = X_train_to_go[:,shuffle_index_1], y_train_to_go[:,shuffle_index_1]

        X_val, y_val = self.input[:,int(2*samples_num/3):], self.y[:,int(2*samples_num/3):]
        np.random.seed(138)
        shuffle_index_2 = np.random.permutation(int(samples_num/3))
        X_val, y_val = X_val[:,shuffle_index_2], y_val[:,shuffle_index_2]


        self.m = self.X_train_to_go.shape[1]
        training_loss_history=[]
        validation_loss_history=[]
        train_accuracy_history=[]
        validation_accuracy_history = []

        for i in range(self.epoch):
            
            # forward propagation
            loss_train = self.feedforward()
            training_loss_history.append(loss_train)
            acc_train = self.get_accuracy(self.A2,self.y_train_to_go)
            train_accuracy_history.append(acc_train)
            
            # back propagation
            w1,w2,b1,b2 = self.backprop()

            #test on the validation set
            y_val_predicted = self.classify(X_val)
            loss_val = self.get_loss(y_val_predicted,y_val,'false')
            validation_loss_history.append(loss_val)

            acc_val = self.get_accuracy(y_val_predicted,y_val)
            validation_accuracy_history.append(acc_val)

            print("Epoch" + str(i) + ": validation accuracy " + str(acc_val) + '\n' +" training loss: "+ str(loss_train)+"validation loss:" + str(loss_val))

        return np.array(training_loss_history),np.array(validation_loss_history),np.array(train_accuracy_history),np.array(validation_accuracy_history)


    def classify(self,x_test):
        A = self.get_sigmoid(np.dot(self.w1,x_test) + self.b1,'false')
        y_test_predicted = self.get_sigmoid(np.dot(self.w2,A) + self.b2,'false')
        return np.round(y_test_predicted)

    def get_accuracy(self,y_test_predicted,y_test):
        y_test_predicted = np.round(y_test_predicted)
        acc = (np.sum((y_test_predicted == y_test).astype(int)))/y_test_predicted.shape[1]
        return acc



nn = NeuralNetwork(X_train, y_train, iterations,lr)
loss_train,loss_val,acc_train,acc_val = nn.training()

plt.figure(1)
plt.title('loss_train vs loss_val')
loss_train_plt, = plt.plot(loss_train, "r")
loss_val_plt, = plt.plot(loss_val, "b")
plt.legend([loss_train_plt, loss_val_plt], ["Train loss", "Validation loss"])
plt.savefig('loss.png')

plt.figure(2)
plt.title('acc_train vs acc_val')
acc_train_plt, = plt.plot(acc_train, "r")
acc_val_plt, = plt.plot(acc_val, "b")
plt.legend([acc_train_plt, acc_val_plt], ["Train accuracy", "Validation accuracy"])
plt.savefig('accuracy.png')

plt.show()



y_test_hat = nn.classify(X_test)
test_acc = nn.get_accuracy(y_test_hat,y_test)
print("the accuracy on test set is:"+str(test_acc))

