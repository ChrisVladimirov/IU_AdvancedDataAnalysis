import math

if __name__ == '__main__':

    from sklearn.decomposition import TruncatedSVD
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora.dictionary import Dictionary
    import pandas as pd
    from nltk.tokenize import word_tokenize
    import nltk as nltk
    from nltk.corpus import stopwords
    import numpy as np

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = pd.read_csv('C:\\Users\chris\PycharmProjects\PythonProject\.venv\data\Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv')
    complaints = []
    vocabulary = set() #storing unique terms from across the corpus

    stop_words = set(stopwords.words('english'))

    for record in corpus.itertuples(index=False, name=None):
        record_text = " ".join([str(value) for value in record])
        r_tokenized = word_tokenize(record_text)
        complaint = [word.lower() for word in r_tokenized if word.isalpha() and word not in stop_words and len(word) >= 2]
        complaints.append(" ".join(complaint))
        vocabulary.update(complaint)

    vect = TfidfVectorizer(min_df=1)
    data = vect.fit_transform(complaints)

    dataframe = pd.DataFrame(data.toarray(), columns=vect.get_feature_names_out())

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Ensure each document in 'complaints' is a list of tokens
    texts = [doc.split() for doc in complaints]

    # Create a Gensim Dictionary and Corpus
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Use the existing TF-IDF matrix 'dataframe' without re-vectorizing
    model = dataframe.values  # Convert to numpy array

    # Reduce dimensions using TruncatedSVD
    n_components = 10
    LSA_model = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=10)
    lsa = LSA_model.fit_transform(model)

    # Extract Topics from LSA
    topics = []
    for i in range(n_components):
        topic_terms = LSA_model.components_[i]
        terms = [dataframe.columns[idx] for idx in topic_terms.argsort()[-10:]]
        topics.append(terms)

    # Calculate Coherence Score using the 'c_v' method
    coherence_model_cv = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()

    print(f'C_V Coherence Score: {coherence_cv}')

    topic_importance = np.abs(lsa).sum(axis=0)
    topic_percentage = (topic_importance / topic_importance.sum()) * 100

    # Save the topics and their percentages to a .txt file
    with open('LSA_topicsImportance.txt', 'w') as f:
        for i, topic in enumerate(topics):
            f.write(f"Topic {i + 1} ({topic_percentage[i]:.2f}% importance):\n")
            f.write(" ".join(topic) + "\n\n")

    # Print the topics and their percentages
    for i, topic in enumerate(topics):
        print(f"Topic {i + 1} ({topic_percentage[i]:.2f}% importance):")
        print(" ".join(topic))
        print()
