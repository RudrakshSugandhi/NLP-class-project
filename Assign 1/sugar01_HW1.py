#importing major library
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def process_regex(path):
    #opening the file in read mode
    with open(path, 'r') as f:
        contents = f.read().lower()

        # Sub function to subsitutue the regual expression with or
        cont_reg = re.sub(r'([a-z]{1,}(?<!y|h|f|s|c))our', r'\1or', contents, count=0, flags=0)

        #(Dr. , Mr., Ms., Mrs.) with appropriate expansions of words (Doctor, Mister, Miss, Misses)
        rep = {'dr.':'doctor','mr.':'mister','ms.':"miss","mrs.":"misses"}
        for key in rep.keys():
            cont_reg_rep= cont_reg.replace(key,rep[key])

    #Writing the converted text in file in 'w' mode
    with open('regex.txt', 'w') as f:
        print(cont_reg_rep,file=f)
    #calling the normalize function 
    normalize_text('regex.txt')

def normalize_text(path):
    with open(path, 'r') as f:

        # lower case
        contents = f.read().lower()

        #removing numbers
        contents = re.sub(r'\d+','',contents)

        #caputring one words in contents or removing puctuation
        contents = re.sub(r'[^\w\s]','',contents)

        #removing white space 
        contents = contents.strip(" ")

        #removing stop words
        stop_words = set(stopwords.words('english'))
        contents = [contents][0].split()
        content_w_stpword = []
        for i in contents:
            if i not in stop_words:
                content_w_stpword.append(i)

        #remove duplicates
        content_w_stpword = sorted(set(content_w_stpword))
        upt_content = []

        # '_' removing the special character from the strings
        for i in content_w_stpword:
            replace = i.replace("_","")
            upt_content.append(replace)
        # re-sort after updates
        updated_content = sorted(upt_content)
        #tried to convert into python dicitonary- Changed after suggestion
        #print(updated_content)
        #my_dict = dict()   
        #for index, value in enumerate(updated_content):
        #    my_dict[index] = value
        #print(my_dict)
    #writing the file in dicitinary.txt file
    with open('dictionary.txt', 'w') as f:
        for item in updated_content:
            f.write("%s\n"% item)
        print("Done")

#calling the function with path as the argument 
# this will generate two files regex.txt and dictionary.txt
process_regex('/Volumes/Drive D/Purdue Courses/Spring 2023/NLP/Assign 1/book.txt')
