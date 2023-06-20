#Include all the libraries
import numpy as np 
import pandas as pd 

from statistics import mean
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

import string 

#Sklearn Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import CountVectorizer

#Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import gensim.downloader as api

# Load the pre-trained Word2Vec model
model = api.load("glove-twitter-25")

#train test split and MLP model call
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from keras_preprocessing.sequence import pad_sequences


class Main:
    def __init__(self):
        pass

    #clean function and for removing stop words
    def clean_text(self,text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        text = ''.join(word for word in text if not word.isdigit())

        # Remove double quotes
        text = text.replace('"', '')
        text = text.replace('“', '')
        text = text.replace('”', '')

        # Remove single quotes
        text = text.replace("'", "")
        text = text.replace("’","")

        #Removing single character
        """text = [word for word in text if len(word) > 1]
        text = ' '.join(text)"""

        # Remove special characters
        text = text.replace('\r', '').replace('\n', ' ').replace('\t', '').strip()
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [word for word in tokens if not word.lower() in stopwords.words('english')]
        
        # Join the tokens back into a string
        clean_text = ' '.join(tokens)
        return clean_text
    
    #function for creating the embedding
    def create_embeddings(self,df,column):
        
        # Create embeddings for each text column in the DataFrame
        if column in df.columns:
            embeddings = []
            for text in df[column]:
                if isinstance(text, str):
                    words = text.split()
                    embedding = np.zeros(model.vector_size)
                    count = 0
                    for word in words:
                        if word in model:
                            embedding += model[word]
                            count += 1
                    if count != 0:
                        embedding /= count
                    embeddings.append(embedding)
                else:
                    embeddings.append(np.zeros(model.vector_size))
            df_copy = df.copy()
            df_copy['comment_text_embedding'] = embeddings
            return df_copy
        
        return df

    #MLP model train
    def train_MLP(self,train_data, train_labels, num_layers):
    
        mlp_model = MLPClassifier(hidden_layer_sizes=[100]*(num_layers), max_iter=10, random_state=42)
        mlp_model.fit(train_data, train_labels)
        return mlp_model 
    
    def train_MLP_model(self,path_to_train_file, num_layers):
        train_txt = pd.read_csv(path_to_train_file).fillna('unknown')
         #Dropping the unnecessary columns from datasert
        train_txt = train_txt.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1)

        """Code for the Part 1 run"""
        #calling the clean function clean the dataset
        ##------uncomment at end
        train_txt['comment_text']= train_txt['comment_text'].apply(self.clean_text)

        #embedding for training
        train_txt_embedding = self.create_embeddings(train_txt,'comment_text')
        X = train_txt_embedding.comment_text_embedding.tolist()
        y =train_txt_embedding.toxic.tolist()

        # Pad the sequences with zeros to make them all the same length
        max_seq_len = max(len(seq) for seq in X)
        padded_train_data = pad_sequences(X, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post', value=0.0)

        # Train the MLP model with num_layers hidden layers
        mlp_model = self.train_MLP(padded_train_data, y, num_layers)
        print("model trained")
        return mlp_model


    def test_MLP_model(self,path_to_test_file, MLP_model):

        #uncomment later
        test_txt = pd.read_csv(path_to_test_file).fillna('unknown')
        test_txt['comment_text']= test_txt['comment_text'].apply(self.clean_text)
        test_label = pd.read_csv("test_labels.csv")

        #embedding for test_data
        test_txt_embedding = self.create_embeddings(test_txt,'comment_text')

        # converting to list
        test_val = test_txt_embedding.comment_text_embedding.tolist()
        test_label =test_label.toxic.tolist()

        #calculate the accuracy 
        max_seq_len = max(len(seq) for seq in test_val)
        padded_test_data = pad_sequences(test_val, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post', value=0.0)
        # Evaluate the model on the validation set

        #toxic probability
        prob = MLP_model.predict_proba(padded_test_data)
        toxic_prob = prob[:,1]
        test_txt['toxic_prob'] = toxic_prob

        #prediction for the model 
        prediction = MLP_model.predict(padded_test_data)
        prediction = np.round(prediction).astype(int)
        prediction = np.where(prediction == 0, 'non-toxic', 'toxic')
        test_txt['class prediction'] = prediction

        score = MLP_model.score(padded_test_data, test_label)
        #print('MLP model accuracy:', score)
        #return dataframe and accuracy 
        return test_txt,score

if __name__ == '__main__':
    main = Main()
    #--------------------1----------------------#
    #Train a 2 layer MLP model on the entire train set.   
    model_2layer = main.train_MLP_model("train.csv",2)
    test_df_2layer,accuracy_2layer = main.test_MLP_model('test.csv',model_2layer)

    #storing the test file in test_2layer csv
    test_df_2layer.to_csv('test_2layer.csv')
    print('MLP model accuracy for 2 layer:', accuracy_2layer)

    #--------------------2----------------------#
    #Train a 3 layer MLP model on the entire train set. 
    model_3layer = main.train_MLP_model("train.csv",3)
    test_df_3layer,accuracy_3layer = main.test_MLP_model('test.csv',model_3layer)
    #storing the test file in test_2layer csv
    test_df_3layer.to_csv('test_3layer.csv')
    print('MLP model accuracy for 3 layer:', accuracy_3layer)