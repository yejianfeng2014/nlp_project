# 使用下面的地址访问
# http://127.0.0.1:5000/
import spacy
from spacy import displacy

text = """But Google is starting from behind. The company made a late push
into hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa
software, which runs on its Echo and Dot devices, have clear leads in
consumer adoption."""

nlp = spacy.load("custom_ner_model")
doc = nlp(text)
displacy.serve(doc, style="ent")