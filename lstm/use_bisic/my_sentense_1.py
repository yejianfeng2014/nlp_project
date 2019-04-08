'''spaCy use'''


import  spacy

#加载英文模型
nlp = spacy.load('en')

# python -m spacy.en.download all


# 分词功能
test_doc = nlp(u"it's word tokenize test for spacy")

print(test_doc)


print(len(test_doc))


# 英文断句

test_doc_sentenses = nlp(u'Natural language processing (NLP) deals with the application of computational models to text or speech data. Application areas within NLP include automatic (machine) translation between languages; dialogue systems, which allow a human to interact with a machine using natural language; and information extraction, where the goal is to transform unstructured text into structured (database) representations that can be searched and browsed in flexible ways. NLP technologies are having a dramatic impact on the way people interact with computers, on the way people interact with each other through the use of language, and on the way people access the vast amount of linguistic data now in electronic form. From a scientific viewpoint, NLP involves fundamental questions of how to structure formal models (for example statistical models) of natural language phenomena, and of how to design algorithms that implement these models.')


for sent in test_doc_sentenses.sents:
    print(sent)

# 词干化（Lemmatize):
test_doc = nlp(u"you are best. it is lemmatize test for spacy. I love these books")

for token in test_doc:
    print(token, token.lemma_, token.lemma)



#词性标注：
for token in test_doc:
   print(token, token.pos_, token.pos)


# 名词短语提取

test_doc = nlp(u'Natural language processing (NLP) deals with the application of computational models to text or speech data. Application areas within NLP include automatic (machine) translation between languages; dialogue systems, which allow a human to interact with a machine using natural language; and information extraction, where the goal is to transform unstructured text into structured (database) representations that can be searched and browsed in flexible ways. NLP technologies are having a dramatic impact on the way people interact with computers, on the way people interact with each other through the use of language, and on the way people access the vast amount of linguistic data now in electronic form. From a scientific viewpoint, NLP involves fundamental questions of how to structure formal models (for example statistical models) of natural language phenomena, and of how to design algorithms that implement these models.')

for np in test_doc.noun_chunks:
    print(np)

