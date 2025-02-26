import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pre_processing import vocabulary

vect = TfidfVectorizer()
data = vect.fit_transform(vocabulary)

data = pd.DataFrame(data.toarray(), columns=vect.get_feature_names_out())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#print(data)
with open('output.txt', 'w') as file:
    file.write(data.to_string(index=False))
