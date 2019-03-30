import pandas as pd

from keras import layers, Model

from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import json

# 读取数据


# 查看数据

import os

max_features = 95000
maxlen = 15

print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv', encoding='utf-8')

df_train['id'] = df_train['id'].apply(str)

df_test = pd.read_csv('../input/test.csv', encoding='utf-8')

df_test['test_id'] = df_test['test_id'].apply(str)

df_all = pd.concat((df_train, df_test), sort=False)

# train_X = train_df["question_text"].fillna("_##_").values

trian_q_1 = df_all['question1'].fillna('_##_').values
trian_q_2 = df_all['question2'].fillna('_##_').values

# train_X = df_all["question1"].fillna("_##_").values

# 查看原始数据的格式

print(df_all.shape)


tokenizer = Tokenizer(num_words=max_features)


tokenizer.fit_on_texts(list(trian_q_1))  # 用训练集去训练这个分词器，生成id_word 和word_id 等信息 如果测试集没有对应的单词会丢弃（这个很严重。。。）
tokenizer.fit_on_texts(list(trian_q_1))  # 用训练集去训练这个分词器，生成id_word 和word_id 等信息 如果测试集没有对应的单词会丢弃（这个很严重。。。）

json_str = tokenizer.to_json()

with open("./word_dic.json", 'w', encoding='utf-8') as json_file:
    json.dump(json_str, json_file, ensure_ascii=False)

trian_q_1 = tokenizer.texts_to_sequences(trian_q_1)
trian_q_2 = tokenizer.texts_to_sequences(trian_q_2)


trian_q_1 = pad_sequences(trian_q_1, maxlen=maxlen)
trian_q_2 = pad_sequences(trian_q_2, maxlen=maxlen)

train_y = df_all['is_duplicate'].values


X1_train, X1_val, X2_train, X2_val, y_train, y_val = \
    train_test_split(trian_q_1,
                     trian_q_1,
                     train_y,
                     test_size=0.3)

# import  keras.layers.IN

# print(X1_train.shape())
# print(X2_train.shape())

print("first line1:", X1_train[0])
print('first_line2', X2_train[0])

input1_tensor = layers.Input(X1_train.shape[1:], name='input_1')
input2_tensor = layers.Input(X2_train.shape[1:], name='input_2')

words_embedding_layer = layers.Embedding(X1_train.max() + 1, 100)

seq_embedding_layer = layers.LSTM(256, activation='tanh')

seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))

# 该层接收一个列表的同shape张量，并返回它们的逐元素积的张量，shape不变。
merge_layer = layers.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])

dense1_layer = layers.Dense(16, activation='sigmoid')(merge_layer)

output = layers.Dense(1, activation='sigmoid', name='out_put')(dense1_layer)

model = Model([input1_tensor, input2_tensor], output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit([X1_train, X2_train], y_train,
          validation_data=([X1_val, X2_val], y_val),
          batch_size=128, epochs=1, verbose=1)

model.save(filepath='./model_similar.h5')

# evaluate = model.evaluate([X1_val, X2_val], y_val)
#
# print(evaluate)

# model.predict()
