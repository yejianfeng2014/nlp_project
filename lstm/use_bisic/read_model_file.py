from rasa_nlu.model import Interpreter
from rasa_nlu import config

# For simplicity we will load the same model twice, usually you would want to use the metadata of
# different models

from rasa_nlu.components import ComponentBuilder

builder = ComponentBuilder(use_cache=True)

model_directory = "./projects/default/default/model_20190404-154250"

interpreter = Interpreter.load(model_directory, builder)     # to use the builder, pass it as an arg when loading the model
# the clone will share resources with the first model, as long as the same builder is passed!
interpreter_clone = Interpreter.load(model_directory, builder)


my_str = "I want to grab lunch"

parse = interpreter_clone.parse(my_str)

# attributes = interpreter_clone.default_output_attributes(my_str)

print(my_str)


# print(attributes)
