''' 意图识别的预测'''

    # 识别意图

model_path = './models/demo/default/model_20190508-090944'


def predict(model_directory):
    from rasa_nlu.model import Metadata, Interpreter

    interpreter = Interpreter.load(model_directory)
    # 使用加载的interpreter处理文本
    print(interpreter.parse("i'm looking for a place in the north of town"))


if __name__ == '__main__':
    predict(model_path)
