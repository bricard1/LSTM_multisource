
#Packages
from __future__ import print_function, division
import pandas as pd
import ast
import random
import torch
from torchtext import data
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torch.nn as nn
#import warnings
#warnings.filterwarnings("ignore")

yelp_amazon=[]
yelp_imdb=[]
amazon_imdb=[]
yelp=[]
amazon=[]
imdb=[]
all3=[]
SEED = 421

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='moses')
LABEL = data.LabelField(dtype=torch.float)
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Architecture
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
	
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))
    

#Everything below is repeated 100 times using different train/test splits.  
for i in range(0,100):
	
	#YelpData = 0
    pos = data.TabularDataset(path='yelp_labelled.txt', format='csv',csv_reader_params={'delimiter':"\t"},fields=[('text', TEXT),('label', LABEL)])
	#Of 1000 posts, 90/10 training/test
    trainandval, test_data=pos.split(split_ratio=0.90)
	#Of the remaining training data, 80/20 train/validation
    train_data, valid_data = trainandval.split(split_ratio=0.80)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE,sort_key=lambda x: len(x.text),device=device)


	#AmazonData = 1
    pos1 = data.TabularDataset(path='amazon_cells_labelled.txt', format='csv',csv_reader_params={'delimiter':"\t"},fields=[('text', TEXT),('label', LABEL)])
    trainandval1, test_data1=pos1.split(split_ratio=0.90)
    train_data1, valid_data1 = trainandval1.split(split_ratio=0.80)
    train_iterator1, valid_iterator1, test_iterator1 = data.BucketIterator.splits((train_data1, valid_data1, test_data1), batch_size=BATCH_SIZE,sort_key=lambda x: len(x.text),device=device)

	#imdbdata=3
    pos2 = data.TabularDataset(path='imdb_labelled.txt', format='csv',csv_reader_params={'delimiter':"\t"},fields=[('text', TEXT),
	('label', LABEL)])
    trainandval2, test_data2=pos2.split(split_ratio=0.90)
    train_data2, valid_data2 = trainandval2.split(split_ratio=0.80)
    train_iterator2, valid_iterator2, test_iterator2 = data.BucketIterator.splits((train_data2, valid_data2, test_data2),batch_size=BATCH_SIZE,sort_key=lambda x: len(x.text),device=device)


	# In[4]:


	#Maximum vocabulary, choose word vectors
    TEXT.build_vocab(train_data, train_data1,train_data2, vectors="glove.twitter.27B.200d")
    LABEL.build_vocab(train_data)
	#Network Hyperparameters
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    optimizer = optim.Adam(model.parameters(),lr=0.0025)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)


	# In[9]:

#Define training/evaluation
    def binary_accuracy(preds, y):#round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum()/len(correct)
        return acc


	# In[10]:


    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for batch in iterator:
            optimizer.zero_grad()
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)


	# In[11]:


    def evaluate(model, iterator, criterion):
	    
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        
        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
		
        return epoch_loss / len(iterator), epoch_acc / len(iterator)


	# In[14]:


    N_EPOCHS = 35
    bestmodelvalue=0
#start running combination models- here first train on Yelp/Amazon. Train and step using yelp data, evaluate on both amazon and yelp data, then repeat using amazon data. Save model and idenity best model using 
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
        train_loss1, train_acc1 = train(model, train_iterator1, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
	    #valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
	    #train_loss2, train_acc2 = train(model, train_iterator2, optimizer, criterion)
        if (valid_acc+valid_acc1)/2 >= bestmodelvalue:
            bestmodelvalue=(valid_acc+valid_acc1)/2
            torch.save(model.state_dict(), "sent_model_amazon_yelp.pt")
    model.load_state_dict(torch.load("sent_model_amazon_yelp.pt"))
    model.eval()
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    test_loss1, test_acc1 = evaluate(model, test_iterator1, criterion)
    test_loss2, test_acc2 = evaluate(model, test_iterator2, criterion)
    temp=[]	
    temp.extend([test_acc,test_acc1,test_acc2])
    yelp_amazon.append(temp)

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)


    pretrained_embeddings = TEXT.vocab.vectors


	# In[6]:


    model.embedding.weight.data.copy_(pretrained_embeddings)


	# In[7]:


    optimizer = optim.Adam(model.parameters(),lr=0.0025)


	# In[8]:


    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 35
    bestmodelvalue=0
#yelp +imdb
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        train_loss2, train_acc2 = train(model, train_iterator2, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        if (valid_acc+valid_acc2)/2 >= bestmodelvalue:
            bestmodelvalue=(valid_acc+valid_acc2)/2
            torch.save(model.state_dict(), "sent_model_yelp_imdb.pt")
    temp=[]
    model.load_state_dict(torch.load("sent_model_yelp_imdb.pt"))
    model.eval()
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    test_loss1, test_acc1 = evaluate(model, test_iterator1, criterion)
    test_loss2, test_acc2 = evaluate(model, test_iterator2, criterion)
    temp=[]
    temp.extend([test_acc,test_acc1,test_acc2])
    yelp_imdb.append(temp)
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)


	# In[5]:


    pretrained_embeddings = TEXT.vocab.vectors


	# In[6]:


    model.embedding.weight.data.copy_(pretrained_embeddings)


	# In[7]:


    optimizer = optim.Adam(model.parameters(),lr=0.0025)


	# In[8]:


    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 35
    bestmodelvalue=0
#amazon + imdb
    for epoch in range(N_EPOCHS):
        train_loss1, train_acc1 = train(model, train_iterator1, optimizer, criterion)
        valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
        valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        train_loss2, train_acc2 = train(model, train_iterator2, optimizer, criterion)
        valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
        valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        if (valid_acc1+valid_acc2)/2 >= bestmodelvalue:
            bestmodelvalue=(valid_acc+valid_acc2)/2
            torch.save(model.state_dict(), "sent_model_amazon_imdb.pt")
    temp=[]
    model.load_state_dict(torch.load("sent_model_amazon_imdb.pt"))
    model.eval()
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    test_loss1, test_acc1 = evaluate(model, test_iterator1, criterion)
    test_loss2, test_acc2 = evaluate(model, test_iterator2, criterion)
    temp=[]
    temp.extend([test_acc,test_acc1,test_acc2])
    amazon_imdb.append(temp)

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)


	# In[5]:


    pretrained_embeddings = TEXT.vocab.vectors


	# In[6]:


    model.embedding.weight.data.copy_(pretrained_embeddings)


	# In[7]:


    optimizer = optim.Adam(model.parameters(),lr=0.0025)


	# In[8]:


    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 35
    bestmodelvalue=0

#Next 3 are yelp only, amazon only, imdb only
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
	   # valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
	   # train_loss2, train_acc2 = train(model, train_iterator2, optimizer, criterion)
	   # valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
	   # valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        if (valid_acc+valid_acc)/2 >= bestmodelvalue:
            bestmodelvalue=(valid_acc+valid_acc)/2
            torch.save(model.state_dict(), "sent_model_yelp.pt")
    model.load_state_dict(torch.load("sent_model_yelp.pt"))
    model.eval()
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    test_loss1, test_acc1 = evaluate(model, test_iterator1, criterion)
    test_loss2, test_acc2 = evaluate(model, test_iterator2, criterion)
    temp=[]
    temp.extend([test_acc,test_acc1,test_acc2])
    yelp.append(temp)

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    optimizer = optim.Adam(model.parameters(),lr=0.0025)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 35
    bestmodelvalue=0

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator1, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator1, criterion)
	   # valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
	   # train_loss2, train_acc2 = train(model, train_iterator2, optimizer, criterion)
	   # valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
	   # valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        if (valid_acc+valid_acc)/2 >= bestmodelvalue:
            bestmodelvalue=(valid_acc+valid_acc)/2
            torch.save(model.state_dict(), "sent_model_amazon.pt")
    
    model.load_state_dict(torch.load("sent_model_amazon.pt"))
    model.eval()
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    test_loss1, test_acc1 = evaluate(model, test_iterator1, criterion)
    test_loss2, test_acc2 = evaluate(model, test_iterator2, criterion)
    temp=[]
    temp.extend([test_acc,test_acc1,test_acc2])
    amazon.append(temp)

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    optimizer = optim.Adam(model.parameters(),lr=0.0025)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    print('first 25 epochs for demonstration...')
    N_EPOCHS = 35
    bestmodelvalue=0


    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator2, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator2, criterion)
	   # valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
	   # train_loss2, train_acc2 = train(model, train_iterator2, optimizer, criterion)
	   # valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
	   # valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        if (valid_acc+valid_acc)/2 >= bestmodelvalue:
            bestmodelvalue=(valid_acc+valid_acc)/2
            torch.save(model.state_dict(), "sent_model_imdbonly.pt")

    model.load_state_dict(torch.load("sent_model_imdbonly.pt"))
    model.eval()
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    test_loss1, test_acc1 = evaluate(model, test_iterator1, criterion)
    test_loss2, test_acc2 = evaluate(model, test_iterator2, criterion)
    temp=[]
    temp.extend([test_acc,test_acc1,test_acc2])
    imdb.append(temp)




    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    optimizer = optim.Adam(model.parameters(),lr=0.0025)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

	# In[14]:

    N_EPOCHS = 25
    bestmodelvalue=0
#This is the combination of all 3 data sets. Still iteratively training through each, then evaluating error using the average. 
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
        valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        train_loss1, train_acc1 = train(model, train_iterator1, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
        valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        train_loss2, train_acc2 = train(model, train_iterator2, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        valid_loss1, valid_acc1 = evaluate(model, valid_iterator1, criterion)
        valid_loss2, valid_acc2 = evaluate(model, valid_iterator2, criterion)
        if (valid_acc+valid_acc1+valid_acc2)/3 >= bestmodelvalue:
            torch.save(model.state_dict(), "sent_model_allreview.pt")
            bestmodelvalue=(valid_acc+valid_acc1+valid_acc2)/3


	# In[ ]:


	#Try the best performing model on the test data
	#I trained 500 epochs and used the one with highest validation accuracy
    model.load_state_dict(torch.load("sent_model_allreview.pt"))
    model.eval()
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    test_loss1, test_acc1 = evaluate(model, test_iterator1, criterion)
    test_loss2, test_acc2 = evaluate(model, test_iterator2, criterion)
    temp=[]
    temp.extend([test_acc,test_acc1,test_acc2])
    all3.append(temp)
