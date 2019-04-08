

#
# import pytest
# import tempfile
# from jsonschema import ValidationError
# from rasa_nlu

from rasa_nlu import training_data,utils
from rasa_nlu.training_data.formats.rasa import validate_rasa_nlu_data


def example_training_data_is_valid():
    demo_json = './../../data/demo-rasa.json'
    data = utils.read_json_file(demo_json)
    validate_rasa_nlu_data(data)


#查看意图文件的数据内容
def demo_data(filename):
    td = training_data.load_data(filename)

    # assert td.intents == {"affirm", "greet", "restaurant_search", "goodbye"}
    print(td.intents)

    # assert td.entities == {"location", "cuisine"}

    print(td.entities)
    # assert len(td.training_examples) == 42

    print(len(td.training_examples))
    # assert len(td.intent_examples) == 42

    print(len(td.intent_examples))
    # assert len(td.entity_examples) == 11

    print(len(td.entity_examples))
    #
    # assert td.entity_synonyms == {'Chines': 'chinese',
    #                               'Chinese': 'chinese',
    #                               'chines': 'chinese',
    #                               'vegg': 'vegetarian',
    #                               'veggie': 'vegetarian'}

    print(td.entity_synonyms)
    #
    # assert td.regex_features == [{"name": "greet", "pattern": r"hey[^\s]*"},
    #                              {"name": "zipcode", "pattern": r"[0-9]{5}"}]
    #

    print(td.regex_features)


data_file = './../../data/demo-rasa.json'

demo_data(data_file)

example_training_data_is_valid()