import urllib.request
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re
import gzip
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

#load input dataset
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="C:/Users/HNUGW/Downloads/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
'''
this will download the file from the target website
'''
#unzip the dataset 
def un_gz(aclImdb_v1.tar.gz):
    f_name = aclImdb_v1.tar.gz.replace(".gz","")
    g_file=gzip.GzipFile(aclImdb_v1.tar.gz)
    open(aclImdb_v1.tar.gz,"w+").write(g_file.read())
    g_file.close()
'''
this will help our target file unzip form .gz to .tar
'''
#unzip the dateset from 
import tarfile
def un_tar(aclImdb_v1.tar.gz):
    untar zip file
    tar = tarfile.open(aclImdb_v1.tar.gz)
    names = tar.getnames()
    if os.path.isdir(aclImdb_v1.tar.gz + "_files"):
        pass
    else:
        os.mkdir(aclImdb_v1.tar.gz + "_files")
    for name in names:
        tar.extract(name, aclImdb_v1.tar.gz + "_files/")
    tar.close()
'''
this will help our target file .tar unzip and released the dataset
'''

#clean the dataset,remove all non alpha numeric characters from the review column of dataset and only keep entries with label 
df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
df = df[df.label != 'unsup']
df['label']=df.label.map({'pos': 1, 'neg': 0})

#tokenize the reviews in our target dataset
max_features = 20000
max_len = 100
tokenizer = Tokenizer(num_words=max_features, lower=True, split=' ')
tokenizer.fit_on_texts(df['review'])

'''
tokenize is a way we used in the NLP.  
Given a character sequence and a defined document unit, 
tokenization is the task of chopping it up into pieces, called tokens , 
perhaps at the same time throwing away certain characters, such as punctuation.
'''
#create training input X. only use df where 'type' is train
X= tokenizer.texts_to_sequences(df[df['type'] == 'train']['review'])
X = pad_sequences(X, maxlen=max_len)

#TRAINING 
model = Sequential()
model.add(Embedding(max_features,100,mask_zero=True))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.summary()
'''
this will create a model with embedding ->LSTM->Dense
'''

#Start the training the model 
epochs = 2
batch_size = 32
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=epochs, batch_size=batch_size, verbose=1)

#testing the model using the test data
X_test= tokenizer.texts_to_sequences(df[df['type'] == 'test']['review'])
X_test = np.array(X_test)
X_test = pad_sequences(X_test, maxlen=max_len)
y_test = df[df['type']=='test']['label'].values
preds= model.predict(X_test)
preds_binary = (preds > 0.5).astype(int)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,preds_binary)

# This will finish the training and do some sample prediction 
text1 = "I disliked the movie. The acting was worse."
text= np.array([text1])
model.save('new_model.model')
import tensorflow as tf
model = tf.keras.models.load_model('new_model.model')
text= tokenizer.texts_to_sequences(text)
text = pad_sequences(text, maxlen=max_len)
prediction = model.predict(text)
print(prediction)
