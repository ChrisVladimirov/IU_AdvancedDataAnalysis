import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from pre_processing import complaints

vect = CountVectorizer(binary=True)
data = vect.fit_transform(complaints)

data = pd.DataFrame(data.toarray(), columns=vect.get_feature_names_out())
#print(data)
with open('output_BoW.txt', 'w') as file:
    file.write(data.to_string(index=False))
