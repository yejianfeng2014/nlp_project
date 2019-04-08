from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy


# 训练模型
def train():
    # 示例数据
    training_data = load_data('data/museum.json')
    # pipeline配置
    trainer = Trainer(config.load("sample_configs/museum_config.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('./models/demo/')
    print(model_directory)
    predict(model_directory)

    # 识别意图


def predict(model_directory):
    from rasa_nlu.model import Metadata, Interpreter

    interpreter = Interpreter.load(model_directory)
    # 使用加载的interpreter处理文本
    print(interpreter.parse(u"这里有什么好看的展览"))


if __name__ == '__main__':
    train()
