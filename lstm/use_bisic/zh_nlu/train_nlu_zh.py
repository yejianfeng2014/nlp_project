from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config


# 训练模型
def train():
    # 示例数据
    training_data = load_data('data/demo-rasa.json')
    # pipeline配置
    trainer = Trainer(config.load("data/config_pretrained_embeddings_spacy.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist('./models/demo/')
    print(model_directory)
    predict(model_directory)

    # 识别意图


def predict(model_directory):
    from rasa_nlu.model import Metadata, Interpreter

    interpreter = Interpreter.load(model_directory)
    # 使用加载的interpreter处理文本
    print(interpreter.parse("i'm looking for a place in the north of town"))


if __name__ == '__main__':
    train()
