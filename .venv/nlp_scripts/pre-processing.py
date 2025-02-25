import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pandas as pd
import nltk as nltk

corpus = pd.read_csv('C:/Users\chris\PycharmProjects\PythonProject\.venv\data\Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv')
complaints = []
vocabulary = []

for record in corpus:
    r_tokenized = word_tokenize(record)
    complaint = [word.lower() for word in r_tokenized if word.isalpha()]
    complaints.append(complaint)
    for word in complaint:
        if word not in vocabulary:
            vocabulary.append(word)

len_vocabulary = len(vocabulary)
print(len_vocabulary)
print(vocabulary)
