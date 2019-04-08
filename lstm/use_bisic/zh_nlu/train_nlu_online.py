
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.channels.console import ConsoleInputChannel

# from  rasa_core.channels.console import CmdlineInput

def train_online(input_channel=ConsoleInputChannel(),
                 interpreter=RasaNLUInterpreter("models/demo/default/model_20180701-171646"),
                 domain_file="data/config_crf_custom_features.yml",
                 training_data_file="data/museum_story.md"):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), KerasPolicy()],
                  interpreter=interpreter)

    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)


if __name__ == '__main__':
    train_online()

