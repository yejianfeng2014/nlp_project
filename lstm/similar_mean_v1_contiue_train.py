''' 继续训练 '''

from sklearn.feature_extraction.text import CountVectorizer
import  json

dic_map = {}

with open('jsonFile.json', 'r') as file:
    load = json.load(file)

    print(type(load))
    print(load)

for key in load.keys():
    dic_map[key] = int(load.get(key))

print(type(dic_map))
print(len(dic_map))
counts_vectorizer = CountVectorizer(max_features=10000 - 1,vocabulary=dic_map)

# counts_vectorizer.fit()