# FFNN.py
#
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_fall19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys
import pickle
import numpy as np
import pandas as pd
#from Eval import Eval

import torch
import torch.nn as nn
import torch.optim as optim

from imdb import IMDBdata

class FFNN(nn.Module):
    def __init__(self, X, Y, VOCAB_SIZE, DIM_EMB=1000, NUM_CLASSES=20):
        super(FFNN, self).__init__()
        (self.VOCAB_SIZE, self.DIM_EMB, self.NUM_CLASSES) = (VOCAB_SIZE, DIM_EMB, NUM_CLASSES)
        #TODO: Initialize parameters.
        self.HID=32
        self.embedding=nn.Embedding(self.VOCAB_SIZE,self.DIM_EMB)
        self.tanh=nn.Tanh()
        self.hidden=nn.Linear(self.DIM_EMB,self.HID)
        self.output=nn.Linear(self.HID,self.NUM_CLASSES)
        self.softmax=nn.Softmax(dim=0)

        self.output2=nn.Linear(10,2)
        self.sigmoid=nn.Sigmoid()

        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, X, train=False):
        #TODO: Implement forward computation.
        embed=self.embedding(X)
        embed=sum(embed,0)/X.size()[0]
        hidden=self.tanh(self.hidden(embed))
        output1=self.output(hidden)
        return self.softmax(output1)


def Eval_FFNN(X, Y, mlp):
    output='prediction_ffnn.txt'
    f=open(output,'w',encoding='utf-8')
    num_correct = 0
    for i in range(len(X)):
        if torch.cuda.is_available():
            X[i]=X[i].cuda()     
        logProbs = mlp.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        f.write(str(pred)+'\n')
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))
    f.close()

def Train_FFNN(X, Y, vocab_size, n_iter):
    print("Start Training!")
    mlp = FFNN(X, Y, vocab_size)
    #TODO: initialize optimizer.
    if torch.cuda.is_available():
        mlp=mlp.cuda()
    optimizer=optim.Adam(mlp.parameters(),lr=0.1)

    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            x_input=X[i]
            y_input=torch.zeros(mlp.NUM_CLASSES)
            y_input[int(Y[i])]=1
            if torch.cuda.is_available():
                x_input=x_input.cuda()
                y_input=y_input.cuda()
            mlp.zero_grad()
            probs=mlp.forward(x_input)
            loss=torch.neg(torch.log(probs)).dot(y_input)
            total_loss+=loss
            loss.backward()
            optimizer.step()
            #TODO: compute gradients, do parameter update, compute loss.
        print(f"loss on epoch {epoch} = {total_loss}")
    return mlp

if __name__ == "__main__":
    train = IMDBdata(sys.argv[1])
    train.vocab.Lock()
    test  = IMDBdata(sys.argv[2], vocab=train.vocab)

    mlp = Train_FFNN(train.XwordList,train.Y,train.vocab.GetVocabSize(), 1)
    Eval_FFNN(test.XwordList, test.Y, mlp)
