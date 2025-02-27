from sklearn.decomposition import TruncatedSVD
from TF_IDF import dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

vectorizer = TfidfVectorizer(use_idf=True, max_features=5, smooth_idf=True)
model = vectorizer.fit_transform(dataframe)

LSA_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=10)
lsa = LSA_model.fit_transform(model)
l = lsa[0]

#for i, topic in enumerate(lsa):
#    print("Topic ",i," : ", topic * 100)
