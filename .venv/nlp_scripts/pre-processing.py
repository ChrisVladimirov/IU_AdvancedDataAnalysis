import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pandas as pd
import nltk as nltk
nltk.download('punkt')
nltk.download('punkt_tab')

corpus = pd.read_csv('C:/Users\chris\PycharmProjects\PythonProject\.venv\data\Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv')
complaints = []
vocabulary = []

for record in corpus.itertuples(index=False, name=None):
    record_text = " ".join([str(value) for value in record])
    r_tokenized = word_tokenize(record_text)
    complaint = [word.lower() for word in r_tokenized if word.isalpha()]
    complaints.append(complaint)
    for word in complaint:
        if word not in vocabulary:
            vocabulary.append(word)

len_vocabulary = len(vocabulary)
print(len_vocabulary)
print("\n".join(vocabulary))
