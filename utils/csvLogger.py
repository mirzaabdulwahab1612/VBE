import os
import pandas as pd

class csvLogger():

  def __init__(self, results_dir):

    self._data = []
    self._save_path = results_dir

  def write(self, data):
    self._data.append(data)
    df = pd.DataFrame(self._data)
    df.to_csv(self._save_path, index=False)