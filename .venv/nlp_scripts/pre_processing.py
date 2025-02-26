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
complaints = []

stop_words = set(stopwords.words('english'))

for record in corpus.itertuples(index=False, name=None):
    record_text = " ".join([str(value) for value in record])
    r_tokenized = word_tokenize(record_text)
    complaint = [word.lower() for word in r_tokenized if word.isalpha() and word not in stop_words and len(word) >= 2]
    complaints.append(" ".join(complaint))
