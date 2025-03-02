import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pandas as pd
import nltk as nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
nltk.download('stopwords')

corpus = pd.read_csv('C:/Users\chris\PycharmProjects\PythonProject\.venv\data\Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv')
custom_stopWords = {'nan', 'inc'}
complaints = []
vocabulary = set() #storing unique terms from across the corpus

stop_words = set(stopwords.words('english'))

for record in corpus.itertuples(index=False, name=None):
    record_text = " ".join([str(value) for value in record])
    r_tokenized = word_tokenize(record_text)
    complaint = [word.lower() for word in r_tokenized if word.isalpha() and word not in stop_words and word
                 not in custom_stopWords and len(word) >= 2]
    complaints.append(" ".join(complaint))
    vocabulary.update(complaint)

#print(vocabulary)
#print(len(vocabulary))
