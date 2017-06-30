import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl

# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

# Output printing out first 5 columns
df.head()

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head(10) # returns (rows, columns)

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

# Convert all strings to their lower case form.
lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)

# Removing all punctuations
sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
print(sans_punctuation_documents)

# Tokenization
preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)

# Count frequencies
frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_counts = Counter(i)
    #frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)
    #frequency_list.append(frequency_counts)
pprint.pprint(frequency_list)
#pprint.pprint(frequency_list)

# call CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
    #from sklearn.feature_extraction.text import CountVetorizer
count_vector = CountVectorizer()
    #count_vector = CountVectorizer()
print(count_vector)

# apply count_vector to documents
count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()
    # doc_array = count_vector.transform(documents).toarray()
doc_array

frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
    # freq_mx = pd.DataFrame(doc_array), columns = count_vector
frequency_matrix

# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.
    # X_train is our training data for the 'sms_message' column.
    # y_train is our training data for the 'label' column
    # X_test is our testing data for the 'sms_message' column.
    # y_test is our testing data for the 'label' column Print out the number of rows we have in each our training and testing data.
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
    # print('xxx':{}.format(df.shape[0]))
print('training rows: {}'.format(X_train.shape[0]))
print('testing rows: {}'.format(X_test.shape[0]))
print('training rows: {}'.format(y_train.shape[0]))
print('testing rows: {}'.format(y_test.shape[0]))

# Bayes Theorem
# P(D)
p_diabetes = 0.01

# P(~D)
p_no_diabetes = 0.99

# Sensitivity or P(Pos|D)
p_pos_diabetes = 0.9

# Specificity or P(Neg/~D)
p_neg_no_diabetes = 0.9

# P(Pos)
p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1 - p_neg_no_diabetes))
print(format(p_pos))

# P(D|Pos)
p_diabetes_pos = (p_diabetes * p_pos_diabetes) / p_pos
print(format(p_diabetes_pos))

# P(Pos/~D)
p_pos_no_diabetes = 0.1

# P(~D|Pos)
p_no_diabetes_pos = (p_no_diabetes * p_pos_no_diabetes) / p_pos
print(format(p_no_diabetes_pos))


# Naive Bayes
# P(J)
p_j = 0.5

# P(F/J)
p_j_f = 0.1

# P(I/J)
p_j_i = 0.1

p_j_text = p_j * p_j_f * p_j_i
print(p_j_text)

# P(G)
p_g = 0.5

# P(F/G)
p_g_f = 0.7

# P(I/G)
p_g_i = 0.2

p_g_text = p_g * p_g_f * p_g_i
print(p_g_text)

p_f_i = p_j_text + p_g_text
print('Probability of words freedom and immigration being said are: ', format(p_f_i))

p_j_fi = p_j_text / p_f_i
print('The probability of Jill Stein saying the words Freedom and Immigration: ', format(p_j_fi))

p_g_fi = p_g_text / p_f_i
print('The probability of Gary Johnson saying the words Freedom and Immigration: ', format(p_g_fi))

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

# predict training model
predictions = naive_bayes.predict(testing_data)

# Compute the accuracy, precision, recall and F1 scores of your model using your test data 'y_test' and the predictions
# you made earlier stored in the 'predictions' variable.
    # true/false = not spam/spam
    # positive/negative = correct/incorrct 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    #  number of correct predictions / total number of predictions
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
    # [True Positives/(True Positives + False Positives)]
print('Precision score: ', format(precision_score(y_test, predictions)))
    # [True Positives/(True Positives + False Negatives)]
print('Recall score: ', format(recall_score(y_test, predictions)))
    # precision and recall, two metrics can be combined to get the F1 score
print('F1 score: ', format(f1_score(y_test, predictions)))
