#importing Important Library 
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
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

#Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader as api


class Main:
#function to compare text 
    def __init__(self):
        pass

    #clean function and for removing stop words
    def clean_text(self, text):
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
    
    #function for getting the most common word 
    def find_common_word(self,string_data,k):
        split = string_data.split()
        #using counter function to calculate the frequency of words
        Count = Counter(split)
        most_occur = Count.most_common(k)
        most_occur_word = [x[0] for x in most_occur]
        return most_occur_word
    
    def similarity(self,list1, list2):
        #load the pretrained model 
        model = api.load("word2vec-google-news-300")

        #Get the embeddings or vectors of all the list in words 
        embeddings1 = np.array([model[word] for word in list1 if word in model])
        embeddings2 = np.array([model[word] for word in list2 if word in model])
        similarity = cosine_similarity(embeddings1,embeddings2)
        #printing the data frame for similarity between different word in data frame format
        #df = pd.DataFrame(similarity, index=list1, columns=list2)
        #print(df)

        #averaging the cosine similariy of all the words vectors 
        average_similarity = np.mean(similarity)
        print(average_similarity)    

    def compare_texts_w2v(self,file_one,file_two,k):
        #print(file_two.head())
        # Concatenate the values in the 'text' column into a single string
        NOT_TOXIC_subset= ' '.join(file_one['comment_text'].astype(str).tolist()).lower()
        TOXIC_subset= ' '.join(file_two['comment_text'].astype(str).tolist()).lower()
        
        #calling the function 
        list1 = self.find_common_word(NOT_TOXIC_subset,k)
        list2 = self.find_common_word(TOXIC_subset,k)
        self.similarity(list1,list2)

    #Calling the function to get overview and similary word for most common non-stop words
    def doc_overview_w2v(self,text_file,k,n):
        model = api.load("word2vec-google-news-300")

        with open(text_file, 'r') as f:
            contents = f.read().lower()
            contents = self.clean_text(contents)
            common_content_word = self.find_common_word(contents,k)
            Similar_word = []

            #common_content_word contain the most common word list 
            #getting the similarity of those common words
            for i in common_content_word:
                temp  = model.most_similar(i,topn=n)
                similar = [x[0] for x in temp]
                Similar_word.append(similar)
        #converting the data into form of data frame to make understand better
        df = pd.DataFrame({'word':common_content_word,'similar word':Similar_word})
        return df



if __name__ == '__main__':
    main = Main()
    #callint the train file as its not mentioned to use train or test.csv
    train_txt = pd.read_csv("train.csv").fillna('unknown')

    #Dropping the unnecessary columns from datasert
    train_txt = train_txt.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1)

    #Dividing the Toxic and non_toxic dataset
    Non_toxic_subset = train_txt[train_txt['toxic']==0]
    Toxic_subset = train_txt[train_txt['toxic']==1]

    """Code for the Part 1 run"""
    #calling the clean function clean the toxic and non-toxic dataset
    Non_toxic_subset['comment_text']= Non_toxic_subset['comment_text'].apply(main.clean_text)
    Toxic_subset['comment_text']= Toxic_subset['comment_text'].apply(main.clean_text)
    
    #printing the Cosine similarity 
    print("Cosine Similarity:")
    print("compare_texts_word2vec(NOT_TOXIC_subset, TOXIC_subset, k = 5):")
    print(main.compare_texts_w2v(Non_toxic_subset,Toxic_subset,5))

    print("compare_texts_word2vec(NOT_TOXIC_subset, TOXIC_subset, k = 10):")
    print(main.compare_texts_w2v(Non_toxic_subset,Toxic_subset,10))
    
    print("compare_texts_word2vec(NOT_TOXIC_subset, TOXIC_subset t, k = 20):")
    print(main.compare_texts_w2v(Non_toxic_subset,Toxic_subset,20))

    #printing the  similary word of all the documents
    """Code for the Part 2 run"""
    pd.set_option('display.max_columns', None)
    print("--------------------------------------------")
    print(main.doc_overview_w2v("warofworlds.txt",5,5))

    print("--------------------------------------------")
    print(main.doc_overview_w2v("on_liberty.txt",5,5))

    print("--------------------------------------------")
    print(main.doc_overview_w2v("kingarthur.txt",5,10))





