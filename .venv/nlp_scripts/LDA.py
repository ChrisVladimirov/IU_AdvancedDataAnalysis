from nlp_scripts.TF_IDF import dataframe
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer(use_idf=True, max_features=5, smooth_idf=True)
model = vectorizer.fit_transform(dataframe)

lda_model = LatentDirichletAllocation(n_components=100, learning_method='online', random_state=42, max_iter=1)
lda_top = lda_model.fit_transform(model)

# Convert the numpy array to a string
lda_top_str = np.array2string(lda_top)
np.savetxt('output_LDA1.txt', lda_top)

#print("Review 1: ")
#with open('output_LDA1.txt', 'w') as file:
#    file.write(lda_top_str)

# Calculate Coherence Score using the 'c_v' method
coherence_model_cv = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_cv = coherence_model_cv.get_coherence()

print(f'C_V Coherence Score: {coherence_cv}')
