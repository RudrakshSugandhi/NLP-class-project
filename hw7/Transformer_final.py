#installing the transformers and dataset from hugging face 
!pip install transformers
!pip install datasets

#importing the modules 
import datasets
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

#importing the imdb dataset 
dataset = datasets.load_dataset("imdb")
#removing the value in dataframe where text value is None
dataset = dataset.filter(lambda example: example['text'] is not None and example['label'] is not None)
# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(dataset['train'])

#getting the random 100 sample having 1 as label and having 0 as label
df_1 = df[df['label']==1].sample(100)
df_0 = df[df['label']==0].sample(100)
df = pd.concat([df_1, df_0])

#shuffle all the values in the 200 data rows
final_df = df.sample(frac=1)

#shorten the sequence lenght of string as model distilbert-base-uncased-finetuned-sst-2-english can only
# it can only predict 512 string lenght
max_seq_length = 512
final_df['text'] = final_df['text'].str.slice(stop=max_seq_length)

#convert the dataframe text rows into list of strings
text_list = final_df['text'].tolist()

#Load the 3 pre-trained model for text classification
classifier_1 = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
classifier_2 = pipeline('text-classification', model='lvwerra/distilbert-imdb')
classifier_3 = pipeline('text-classification', model='aychang/roberta-base-imdb')

#define a list with 200 text samples to classify 

# Classify some example text
result_1 = classifier_1(text_list)
result_2 = classifier_2(text_list)
result_3 = classifier_3(text_list)

#for this model distilbert-base-uncased-finetuned-sst-2-english
def pred_first_two_classifer(results): 
  model_result =[]
  for result in results:
    model_result.append(result['label'])
  y_pred = [1 if label == 'POSITIVE' else 0 for label in model_result]
  #getting the original label 
  y_true = final_df['label'].tolist() 
  return y_pred,y_true

#for this model distilbert-base-uncased-finetuned-sst-2-english
def pred_third(results):
  model_result =[]
  for result in results:
    model_result.append(result['label'])
  y_pred = [1 if label == 'pos' else 0 for label in model_result]
  #getting the original label 
  y_true = final_df['label'].tolist()
  return y_pred,y_true

#calulating the predicted value and true value for each model 
y_pred_1, y_true_1 = pred_first_two_classifer(result_1)
y_pred_2, y_true_2 = pred_first_two_classifer(result_2)
y_pred_3, y_true_3 = pred_third(result_3)

print(y_pred_3)

def accuracy_matrix(y_pred,y_true):
  # Compute accuracy
  accuracy = accuracy_score(y_true, y_pred)

  # Compute precision
  precision = precision_score(y_true, y_pred)

  # Compute recall
  recall = recall_score(y_true, y_pred)
  return accuracy,precision,recall

# accruarcy precision and recall of all the three model. 
print(accuracy_matrix(y_pred_1, y_true_1))
print(accuracy_matrix(y_pred_2, y_true_2))
print(accuracy_matrix(y_pred_3, y_true_3))

