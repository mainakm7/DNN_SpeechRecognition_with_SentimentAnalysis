import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset,DataLoader

import matplotlib.pyplot as plt
from glob import glob as glob_module
import os
import sys

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import random
from collections import Counter


# Define paths
notebook_folder = "notebooks"
data_folder = r"data\processed\imdb_data"

# Get current working directory
current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# Combine paths
notebook_path = os.path.join(parent_directory, notebook_folder)
data_path = os.path.join(parent_directory, data_folder)


train_pos_path = os.path.join(data_path, r"train\pos")
train_neg_path = os.path.join(data_path, r"train\neg")
test_pos_path = os.path.join(data_path, r"test\pos")
test_neg_path = os.path.join(data_path, r"test\neg")


train_pos_files = glob_module(os.path.join(train_pos_path, '*.txt'), recursive=True)
train_neg_files = glob_module(os.path.join(train_neg_path, '*.txt'), recursive=True)
test_pos_files = glob_module(os.path.join(test_pos_path, '*.txt'), recursive=True)
test_neg_files = glob_module(os.path.join(test_neg_path, '*.txt'), recursive=True)


def preprocess_files(files):
    processed_files = []
    stemmer = SnowballStemmer(language="english")
    stop_words = set(stopwords.words("english"))
    for file in files:
        with open(file, "r") as f:
            text = f.read().lower()  
            
            words = re.findall(r'\b\w+\b', text)
            
            words = [stemmer.stem(word) for word in words if word not in stop_words]
            processed_files.append(words)
    return processed_files


pos_train = preprocess_files(train_pos_files)
neg_train = preprocess_files(train_neg_files)
pos_test = preprocess_files(test_pos_files)
neg_test = preprocess_files(test_neg_files)



pos_train_label = list(np.ones(len(pos_train)))
neg_train_label = list(np.zeros(len(neg_train)))
pos_test_label = list(np.ones(len(pos_test)))
neg_test_label = list(np.zeros(len(neg_test)))



all_text = pos_train + neg_train + pos_test + neg_test
all_words = [word for doc in all_text for word in doc]
word_counts = Counter(all_words)
top_words = word_counts.most_common(10000)
word2id = {word: idx for idx, (word, _) in enumerate(top_words)}
id2word = {idx: word for word, idx in word2id.items()}


combined_data_train = list(zip(pos_train + neg_train, pos_train_label + neg_train_label))
combined_data_test = list(zip(pos_test + neg_test, pos_test_label + neg_test_label))


random.shuffle(combined_data_train)
random.shuffle(combined_data_test)


train_data, train_labels = zip(*combined_data_train)
test_data, test_labels = zip(*combined_data_test)


UNK_TOKEN = "<UNK>"
word2id[UNK_TOKEN] = len(word2id)
for doc in train_data:
    for i, word in enumerate(doc):
        doc[i] = word2id.get(word, word2id[UNK_TOKEN])
for doc in test_data:
    for i, word in enumerate(doc):
        doc[i] = word2id.get(word, word2id[UNK_TOKEN])

unknown = word2id[UNK_TOKEN]
        

vocab_size = len(word2id)

max_len_combined = 0
for doc in train_data + test_data:
    max_len_combined = max(max_len_combined, len(doc))
        
max_len = max_len_combined

train_data = [seq for seq in train_data if seq is not None]
test_data = [seq for seq in test_data if seq is not None]


train_data_tensor = [torch.tensor(seq) for seq in train_data]
test_data_tensor = [torch.tensor(seq) for seq in test_data]


padded_train_data_tensor = nn.utils.rnn.pad_sequence(train_data_tensor, batch_first=True, padding_value=0)
padded_test_data_tensor = nn.utils.rnn.pad_sequence(test_data_tensor, batch_first=True, padding_value=0)


padded_train_data_tensor = padded_train_data_tensor[:, :max_len]
padded_test_data_tensor = padded_test_data_tensor[:, :max_len]


train_labels_tensor = torch.tensor(train_labels).float()
test_labels_tensor = torch.tensor(test_labels).float()


train_set = TensorDataset(padded_train_data_tensor,train_labels_tensor)
test_set = TensorDataset(padded_test_data_tensor,test_labels_tensor)


batch_size = 64
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, drop_last = True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle=True, drop_last = True)



class sent_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 50)
        self.lstm = nn.LSTM(50, 100, bidirectional=True, batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(100 * 2) 
        self.fc1 = nn.Linear(100 * 2, 64)  
        self.bn_fc1 = nn.BatchNorm1d(64)  
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out.transpose(1, 2) 
        lstm_out = self.bn_lstm(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)  
        lstm_out = torch.max(lstm_out, dim=1)[0]  
        out = F.relu(self.fc1(lstm_out))
        out = self.bn_fc1(out)
        out = self.fc2(out)
        return out.squeeze(1)


model = sent_model()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model,epochs=10):
    
    
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
    
    
    losses = []
    trainbatchloss, trainbatchAcc = [], []
    trainAcc = []
    
    testbatchAcc = []
    testAcc  = []
    
    for i in range(epochs):
        sys.stdout.write(f"\n epoch {i+1}/{epochs}")
        model.train()
        for X,y in train_loader:
            X = X.to(device)
            y = y.to(device)
            yhat = model(X)
            loss = lossfun(y,yhat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            trainbatchloss.append(loss.item())
            trainbatchAcc.append( 100*torch.mean(((yhat>0) == y).float()).item() )
            
        
        trainAcc.append( np.mean(trainbatchAcc) )

        
        losses.append(np.mean(trainbatchloss) )
        
        model.eval()
        for Xtest,ytest in test_loader:
    
            Xtest = Xtest.to(device)
            ytest = ytest.to(device)
    
            with torch.no_grad():
                ypred = model(Xtest)
    
            testbatchAcc.append(100*torch.mean(((ypred>0) == ytest).float()).item())
        testAcc.append(np.mean(testbatchAcc))
        
    return trainAcc,testAcc,losses
        
    
        
    

sentimentmodel = sent_model()
sentimentmodel.to(device)
trainAcc,testAcc,losses = train_model(sentimentmodel,100)


