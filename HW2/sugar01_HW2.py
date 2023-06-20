#importing necessary libraris
import re
import nltk
#imported stopword- as I have removed stop words from HW1 dictionary
nltk.download('stopwords')
import numpy as np
from nltk.corpus import stopwords

#Min distance formula for calculating min distance between word1 and word2
def min_distance(word1, word2):

    # create blank distance matrix(dm)
    dm = np.zeros((len(word1)+1,len(word2)+1))

    #initialize 
    for i in range(len(word1)+1):
        dm[i][0] = i
    for j in range(len(word2)+1):
        dm[0][j] = j

    #Recurrence Relation iterate for the both word in dictionary 
    for i in range(1,len(word1)+1):
        for j in range(1,len(word2)+1):
                temp = None
                if word1[i-1]==word2[j-1]:
                    temp = 0
                else:
                    #As mentioned in class 2
                    temp = 2
                dm[i][j] = min(dm[i-1][j] + 1,
                                dm[i][j-1] + 1,
                                dm[i-1][j-1] + temp)
    # returning the last column in the matrix for distnace                         
    return dm[i][j]

#Spell checked function for checking spelling mistakes
def spell_checker():
    print("------------------------------")
    print("Welcome to the spell checker!")
    print("Please enter a text to check spelling or enter quit to exit the program.")
    print("--------------------------")
    sent = "True"

    while(sent!="quit"):
        print("\nEnter text to be checked:",end='')
        sent = str(input()).lower()

        #Checking for the quit command then return the goodbye
        if sent=="quit":
            return print("Goodbye!")
        
        #Removing punctuations and numerical value
        sent = re.sub(r'[^\w\s]','',sent)
        sent_li = sent.split()

        #Removing stop word
        stop_words = set(stopwords.words('english'))
        sent_wo_stpword = []
        for i in sent_li:
            if i not in stop_words:
                sent_wo_stpword.append(i)
        #print(sent_wo_stpword)
        
        # opening the dicitonary file in read mode
        with open('dictionary.txt', 'r') as f:
            content = f.read().split()
        miss_spell =[]

        #Checking if the miss spell word is in dictionary 
        for i in sent_wo_stpword:
            if i not in content:
                miss_spell.append(i)

        # the misspelled word count
        if len(miss_spell)==0:
            print("No misspellings detected!")
        else:
            print("Misspelling - Suggestion")

            #Two empty dicitionary
            # 1  getting the related word min distnace for each missspelled word and its distance 
            # 2  storing the min distance of each word in sentence
            match_dict = {}
            final_dict = {}
            for i in miss_spell:
                for j in content:
                    match_dict.update({j:min_distance(i,j)})
                final_dict.update({i:min(match_dict, key=match_dict.get)})
            
            #printing the final miss-spelled related word from dictionary 
            for keys,value in final_dict.items():
                print(keys+" - "+ value)


#calling the spell checker function
spell_checker()