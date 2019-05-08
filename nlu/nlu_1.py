# 使用下面的地址访问
# http://127.0.0.1:5000/
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(u"This is a sentence.")

nlp
displacy.serve(doc, style="dep")