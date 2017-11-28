import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from agglomerative import Agglomerative

if __name__ == "__main__":
    data = pd.read_csv('CencusIncome/CencusIncome.data.txt', sep='\s*,\s*', na_values=["?"], engine='python')
    data = data.dropna(how='any')  # deleted any missing value
    del data['education']  # deleted categorial education because it's same with education-num
    the_data = data.loc[:, 'age':'native-country']
    label = data.loc[:, 'class']

    encoder = preprocessing.LabelEncoder()
    label = encoder.fit_transform(label)
    for col in ["workclass", "marital-status", "occupation", "relationship", "race", "sex",
                "native-country"]:
        the_data[col] = encoder.fit_transform(the_data[col])

    print(the_data.shape)
    testing_count = 10000
    model = AgglomerativeClustering(linkage="ward", n_clusters=2)
    model.fit(the_data.head(n=testing_count))
    testing = label[0:testing_count]
    #print(testing)
    #print(model.labels_)
    count = 0
    for idx, label in enumerate(model.labels_):
        if label == testing[idx]:
            count += 1
    print("Correct: ",count)
    print("Accuracy: ",count*100.0/testing_count)