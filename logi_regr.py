import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    test_dataset = h5py.File('datasets/test_happy.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])
    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0],1))
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0],1))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def init_wb (feature_size):
    weight = np.zeros((feature_size,1))
    bias = 0
    return weight, bias

def sigmoid (w1, b1, x_data):
    theta = np.dot(x_data,w1) + b1
    z = 1/(1+np.exp(-theta))
    return z

def loss_func (y_data, y_hat, sample_size):
    loss = np.sum(-(y_data * np.log10(y_hat)) - ((1-y_data) * np.log10(1-y_hat)))/sample_size
    return loss

def back_prop (x_sample,y_sample,act_func,sample):
    dZ = act_func - y_sample
    dW = (np.dot(x_sample.T,dZ))/sample
    dB = np.dot(dZ.T,np.ones((sample,1)))/sample
    return dZ,dW,dB

x_train, y_train, x_test, y_test, classif = load_dataset()

x, x_val, y, y_val = train_test_split(x_train, y_train, test_size=0.5, shuffle=False)

x = ((x.reshape(len(x),12288))/256-0.5)*2
x_val = ((x_val.reshape(len(x_val),12288))/256-0.5)*2
x_test = ((x_test.reshape(len(x_test),12288))/256-0.5)*2
# x = (x.reshape(len(x),12288))/256
# x_val = (x_val.reshape(len(x_val),12288))/256

m = len(x)
m1 = len(x_val)
m_test = len(x_test)
n = len(x[0])
alpha = 0.001
i=0
k=0
l=0
w, b = init_wb(n)
losses = []
losses_val = []
losses_test = []

while i<500:
    a = sigmoid(w,b,x)
    J = loss_func(y,a,m)
    losses.append(J)
    dz, dw, db = back_prop(x, y, a, m)
    
    w = w - alpha * dw
    b = b - alpha * db
    i+=1

# plt.plot(np.linspace(0,i,i),losses)

#validation
    
while k<500:
    a = sigmoid(w,b,x_val)
    J = loss_func(y_val,a,m1)
    losses_val.append(J)
    dz, dw, db = back_prop(x_val, y_val, a, m1)
    
    w = w - alpha * dw
    b = b - alpha * db
    k+=1
# plt.plot(np.linspace(0,k,k),losses_val)

#testing

while l<500:
    a = sigmoid(w,b,x_test)
    np.round(a)
    J = loss_func(y_test,a,m_test)
    losses_test.append(J)
    dz, dw, db = back_prop(x_test, y_test, a, m_test)
    w = w - alpha * dw
    b = b - alpha * db
    l+=1

accuracy = (1-J)*100
print("Accuracy:",accuracy)    
plt.plot(np.linspace(0,l,l),losses_test)