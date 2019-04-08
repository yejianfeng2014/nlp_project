
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy



def train_dialogue(domain_file="data/domain.yml",
                   model_path="models/dialogue",
                   training_data_file="stories.md"):
    agent = Agent(domain_file, policies=[MemoizationPolicy()])
    agent.train(
        training_data_file,
        max_history=3,
        epochs=100,
        batch_size=50,
        augmentation_factor=50,
        validation_split=0.2
    )

    agent.persist(model_path)


if __name__ == '__main__':
    train_dialogue()
