import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pre_processing import complaints

vect = TfidfVectorizer(min_df=1)
dataframe = vect.fit_transform(complaints)

dataframe = pd.DataFrame(dataframe.toarray(), columns=vect.get_feature_names_out())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

with open('output_TF-IDF.txt', 'w') as file:
    file.write(dataframe.to_string(index=False))
