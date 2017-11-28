import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from agglomerative import Agglomerative

if __name__ == "__main__":
    data = pd.read_csv('CencusIncome/CencusIncome.data.txt', sep='\s*,\s*', na_values=["?"], engine='python')
    print(data.columns.tolist())
    print(data.shape)
    data = data.dropna(how='any')  # deleted any missing value
    del data['education']  # deleted categorial education because it's same with education-num
    print(data.shape)
    the_data = data.loc[:, 'age':'native-country']
    print(the_data.columns)
    label = data.loc[:, 'class']

    encoder = preprocessing.LabelEncoder()
    for col in ["workclass", "marital-status", "occupation", "relationship", "race", "sex",
                "native-country"]:
        the_data[col] = encoder.fit_transform(the_data[col])

    model = AgglomerativeClustering(linkage="average", n_clusters=2)
    model.fit(the_data)
    print(model)
