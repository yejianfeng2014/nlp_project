from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.model import Trainer

builder = ComponentBuilder(use_cache=True)      # will cache components between pipelines (where possible)

training_data = load_data('./../../data/demo-rasa.json')

trainer = Trainer(config.load("./../../sample_configs/config_pretrained_embeddings_spacy_duckling.yml"), builder)
trainer.train(training_data)
model_directory = trainer.persist('./projects/default/')  # Ret