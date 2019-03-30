
import pandas as pd

from keras import layers, Model

from sklearn.feature_extraction.text import CountVectorizer
import itertools

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re

import  json
# 读取数据


# 查看数据

import os

print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv', encoding='utf-8')

df_train['id'] = df_train['id'].apply(str)

df_test = pd.read_csv('../input/test.csv', encoding='utf-8')

df_test['test_id'] = df_test['test_id'].apply(str)

df_all = pd.concat((df_train, df_test),sort=False)

df_all['question1'].fillna('', inplace=True)
df_all['question2'].fillna('', inplace=True)

# 查看原始数据的格式

print(df_all.shape)

# Create Vocab

counts_vectorizer = CountVectorizer(max_features=10000 - 1).fit(
    itertools.chain(df_all['question1'], df_all['question2']))


# counts_vectorizer.

other_index = len(counts_vectorizer.vocabulary_)


print(counts_vectorizer.vocabulary_.keys)


save_map = {}

for a in counts_vectorizer.vocabulary_.keys():
    save_map[a] = str (counts_vectorizer.vocabulary_.get(a))


print(len(save_map))

to_saveMap = counts_vectorizer.vocabulary_

jsObj = json.dumps(save_map)

fileObject = open('jsonFile.json', 'w')
fileObject.write(jsObj)

# pre_data


words_tokenizer = re.compile(counts_vectorizer.token_pattern)


# words_tokenizer.


def create_padded_seqs(texts, maxlen=10):
    seqs = texts.apply(lambda s:
                       [counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
                        for w in words_tokenizer.findall(s.lower())]
                       )

    return pad_sequences(seqs, maxlen)


X1_train, X1_val, X2_train, X2_val, y_train, y_val = \
    train_test_split(create_padded_seqs(df_all[df_all['id'].notnull()]['question1']),
                     create_padded_seqs(df_all[df_all['id'].notnull()]['question2']),
                     df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     stratify=df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     test_size=0.3, random_state=1989)

# import  keras.layers.IN

input1_tensor = layers.Input(X1_train.shape[1:])
input2_tensor = layers.Input(X2_train.shape[1:])

words_embedding_layer = layers.Embedding(X1_train.max() + 1, 100)

seq_embedding_layer = layers.LSTM(256, activation='tanh')

seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))

# 该层接收一个列表的同shape张量，并返回它们的逐元素积的张量，shape不变。
merge_layer = layers.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])

dense1_layer  = layers.Dense(16, activation='sigmoid')(merge_layer)

output = layers.Dense(1, activation='sigmoid')(dense1_layer)


model = Model([input1_tensor,input2_tensor],output)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())


model.fit([X1_train, X2_train], y_train,
          validation_data=([X1_val, X2_val], y_val),
          batch_size=128, epochs=10, verbose=1)


model.save(filepath='./model_similar_v2.h5')

evaluate = model.evaluate([X1_val, X2_val], y_val, )

print(evaluate)

