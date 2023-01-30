import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F 


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

x_train, y_train, x_test, y_test, classes= load_dataset()
x_train=(x_train - np.mean(x_train))/np.std(x_train)
x_test=(x_test - np.mean(x_test))/np.std(x_test)
x_train= x_train.reshape(600,12288)
x_test= x_test.reshape(150,12288)

USE_GPU = False
dtype = torch.float32
device = torch.device('cpu')
print_every = 100

hidden_layer_size = 1000
learning_rate = 0.001

def rand_init(shape):

    if len(shape) == 2:
        f = shape[0]
    else:
        f = np.prod(shape[1:])
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / f)
    w.requires_grad = True
    return w

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1) 


# splitting the training and test data into minibatches

train_data = []
for i in range(len(x_train)):
   train_data.append([x_train[i], y_train[i]])

loader_train = DataLoader(train_data, shuffle=False, batch_size=2)


test_data = []
for i in range(len(x_test)):
   test_data.append([x_test[i], y_test[i]])

loader_test = DataLoader(test_data, shuffle=False, batch_size=2)


def two_layers(x, params):

    x = flatten(x)
    
    w1, w2 = params
    
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x
    

def accu(loader, model_fn, params, split):

    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            preds = torch.sigmoid(scores)
            preds=preds>=0.5
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples

    return acc

def train(model_fn, params, learning_rate):

    loss_mat=[]
    for t, (x, y) in enumerate(loader_train):
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.float32)

        scores = model_fn(x, params)
        scores= torch.sigmoid(scores)
        loss = F.binary_cross_entropy(scores, y)
        loss_item=loss
        if t%print_every==0:
            loss_mat.append(loss_item.detach().numpy())


        loss.backward()


        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

             
                w.grad.zero_()


    accuracy= accu(loader_test, model_fn, params, 'test')
    print(accuracy)
    return loss_mat, accuracy



w1 = rand_init((12288, hidden_layer_size))
w2 = rand_init((hidden_layer_size, 1))

print_every = 1

loss_data, test_accuracy = train(two_layers, [w1, w2], learning_rate)
plt.plot(loss_data)
plt.show()

lr_set= [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
test_acc=[]
for learning_rate in lr_set:
    hidden_layer_size = 1000

    w1 = rand_init((12288, hidden_layer_size))
    w2 = rand_init((hidden_layer_size, 1))
    loss_data, test_accuracy= train(two_layers, [w1, w2], learning_rate)
    test_acc.append(test_accuracy)
plt.plot(lr_set, test_acc)
plt.show()

split_ratio= [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1, 0.005, 0.001]
train_set_ratio = [1 - sr for sr in split_ratio]
test_acc=[]
for sr in split_ratio:
    x_train_select, x_other, y_train_select, y_other = train_test_split(x_train, y_train, test_size=sr, shuffle=False)
    train_data = []
    for i in range(len(x_train_select)):
        train_data.append([x_train_select[i], y_train_select[i]])
        loader_train = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=2)      
    hidden_layer_size = 1000
    w1 = rand_init((12288, hidden_layer_size))
    w2 = rand_init((hidden_layer_size, 1))
    loss_data, test_accuracy= train(two_layers, [w1, w2], 1e-3)
    test_acc.append(test_accuracy)
plt.plot(train_set_ratio, test_acc)
plt.show()