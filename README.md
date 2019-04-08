# nlp_project
npl 相关的数据处理和模型生成

1,生成了字典表文件，
2，可以从字典表生成分词器



安装spacy的相关东西

pip install spacy

# Download best-matching version of specific model for your spaCy installation
python -m spacy download en_core_web_sm

# Out-of-the-box: download best-matching default model and create shortcut link
python -m spacy download en

# Download exact model version (doesn't create shortcut link)
python -m spacy download en_core_web_sm-2.1.0 --direct


# 安装nlu 

git clone git@github.com:RasaHQ/rasa_nlu.git
cd rasa_nlu
pip install -r requirements.txt
python setup.py install


AttributeError: 'Agent' object has no attribute 'train_online'