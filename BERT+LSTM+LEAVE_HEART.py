import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def Eval_LSTM(X, Y, lstm):
    output='prediction10.txt'
    f=open(output,'w',encoding='utf-8')
    num_correct = 0
    for i in range(len(X)):
        if torch.cuda.is_available():
            X[i]=X[i].cuda()
        logProbs = lstm.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        f.write(str(pred)+'\n')
        if pred == int(Y[i]):
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))
    f.close()

class LSTM(nn.Module):
    def __init__(self, X, Y,NUM_CLASSES=20):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(1024,20,1)
        self.hidden = (torch.randn(1, 1, 20),
                  torch.randn(1, 1, 20))
        

    def forward(self, X, train=False):
        #TODO: Implement forward computation
        X = X.view(1,1,-1).float()
        #X = torch.DoubleTensor(X)
        #print(X)
        if torch.cuda.is_available():
            a=torch.randn(1,1,20).cuda()
            b=torch.randn(1,1,20).cuda()
            X=X.cuda()
            hidden=(a,b)
        out, (a,b) = self.lstm(X, hidden)
        return out

def Train_LSTM(X, Y,n_iter):
    print("Start Training!")
 
    lstm = LSTM(X, Y)
    if torch.cuda.is_available():
        lstm=lstm.cuda()
    #TODO: initialize optimizer.
    #optimizer = optim.SGD(mlp.parameters(), lr=.01)
    optimizer = optim.Adam(lstm.parameters(), lr=1e-5)
    Loss = torch.nn.CrossEntropyLoss()

    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            #TODO: compute gradients, do parameter update, compute loss.
            lstm.zero_grad()
            y = torch.tensor(Y[i]).long()
            if torch.cuda.is_available():
                X[i]=X[i].cuda()
                y=y.cuda()
            probs = lstm.forward(X[i])
            probs = probs.squeeze(0)
            loss = Loss(probs,y.unsqueeze(0))
            total_loss += loss
            loss.backward(retain_graph=True)
            lstm.hidden[0].detach_()
            lstm.hidden[1].detach_()
            optimizer.step()
            
        print(f"loss on epoch {epoch} = {total_loss}")
    return lstm

# Training tweets as arrays
with open("train_0_100000.obj", "rb") as tr1:
    train_0 = pickle.load(tr1)
with open("train_100000_200000.obj", "rb") as tr2:
    train_1 = pickle.load(tr2)
with open("train_200000_300000.obj", "rb") as tr3:
    train_2 = pickle.load(tr3)
with open("train_300000_400000.obj", "rb") as tr4:
    train_3 = pickle.load(tr4)
with open("train_400000_end.obj", "rb") as tr5:
    train_4 = pickle.load(tr5)

# training labels
with open("tweet.txt.labels", "r") as tr_labels:
    train_labels_raw = tr_labels.readlines()
#Test tweets as arrays
with open("test.obj", "rb") as test_tweets:
    test_twts = pickle.load(test_tweets)
# test labels
with open("test/us_test.txt.labels", "r") as tl:
    test_labels_raw = tl.readlines()

train_whole = np.array([])
train_whole = np.append(train_whole, train_0)
train_whole = np.append(train_whole, train_1)
train_whole = np.append(train_whole, train_2)
train_whole = np.append(train_whole, train_3)
train_whole = np.append(train_whole, train_4)
train_whole = np.reshape(train_whole, (437161,1024))
#train_whole_raw = np.reshape(train_whole, (100000,1024))
#print(train_whole_raw.shape)
test_whole = np.array([])
test_whole=np.append(test_whole,test_twts)
test_whole=np.reshape(test_whole,(50000,1024))


indexs_train=[]
train_labels=[]
for i in range(0,len(train_labels_raw)):
    label=int(train_labels_raw[i].rstrip('\n'))
    if label==0:
        indexs_train.append(i)
    else:
        train_labels.append(label)
train_whole=np.delete(train_whole,indexs_train,axis=0)

indexs_test=[]
test_labels=[]
for i in range(0,len(test_labels_raw)):
    label=int(test_labels_raw[i].rstrip('\n'))
    if label==0:
        indexs_test.append(i)
    else:
        test_labels.append(label)
test_whole=np.delete(test_whole,indexs_test,axis=0)

train_tensor = torch.from_numpy(train_whole)
lstm = Train_LSTM(train_tensor, train_labels,10)
X_test = torch.from_numpy(test_whole)
Eval_LSTM(X_test, test_labels, lstm)