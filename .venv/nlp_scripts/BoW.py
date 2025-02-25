import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from pre_processing import vocabulary

vect = CountVectorizer()
data = vect.fit_transform(vocabulary)

data = pd.DataFrame(data.toarray(), columns=vect.get_feature_names_out())
print(data)
