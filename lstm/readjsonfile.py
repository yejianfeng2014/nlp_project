'''

    从一个json 文件中恢复 keras 的分词工具，tokenizer

'''

from keras_preprocessing.text import tokenizer_from_json
import json

str_file = './hmm.json'

''' 文件的路径是需要读取的json 文件 '''
def get_token(filePaht=str_file):
    with open(str_file, 'r', encoding='utf-8') as f:
        print("Load str file from {}".format(str_file))
        str1 = f.read()
        r = json.loads(str1)

    tokenizer = tokenizer_from_json(r)

    return tokenizer


token = get_token(str_file)

print(token.index_word)

print(token.word_index)

#
# Tokenizer.token
# tokenizer_from_json
