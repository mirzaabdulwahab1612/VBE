import json
from json import JSONEncoder

class DummyObject(object):
    def __init__(self, file_name=None):
        if file_name is None:
            pass
        else:
            json_data = open(file_name, 'r')
            config_file = json.load(json_data)
            json_data.close()
            self.set_attributes(config_file)

    def set_attributes(self, config_file):
        for c_name, c_value in config_file.items():
            setattr(self, c_name, c_value)
            if type(getattr(self, c_name)) == dict:
                sub_obj = DummyObject()
                sub_obj = sub_obj.set_attributes(getattr(self, c_name))
                setattr(self, c_name, sub_obj)
        return self

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class DummyEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__
