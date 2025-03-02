if __name__ == '__main__':
    from nlp_scripts.pre_processing import vocabulary
    from nlp_scripts.TF_IDF import dataframe
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    from pre_processing import complaints

    #max_features -> how much of the vocabulary to be considered in the vectorizer: the full vocabulary
    vectorizer = TfidfVectorizer(use_idf=True, max_features=len(vocabulary), smooth_idf=True)
    model = vectorizer.fit_transform(dataframe)

    n_components=10 #number of topics: smaller count -> greater interpretability
    lda_model = LatentDirichletAllocation(n_components=n_components, learning_method='online', random_state=42, max_iter=1)
    lda_top = lda_model.fit_transform(model)

    texts = [doc.split() for doc in complaints]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    topics = []
    for i in range(n_components):
        topic_terms = lda_model.components_[i]
        terms = [dataframe.columns[idx] for idx in topic_terms.argsort()[-10:]]
        topics.append(terms)

    coherence_model_cv = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()

    print(f'C_V Coherence Score: {coherence_cv}')

    topic_importance = lda_top.sum(axis=0)
    topic_percentage = (topic_importance / topic_importance.sum()) * 100

    with open('LDA_topicsImportance.txt', 'w') as f:
        for i, topic in enumerate(topics):
            f.write(f"Topic {i + 1} ({topic_percentage[i]:.2f}% importance):\n")
            f.write(" ".join(topic) + "\n\n")

    for i, topic in enumerate(topics):
        print(f"Topic {i + 1} ({topic_percentage[i]:.2f}% importance):")
        print(" ".join(topic))
        print()
