#!/bin/bash

sudo apt-get -y update
sudo apt-get -y install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get -y install libglib2.0-0
sudo apt-get install -y libsm6 libxext6 libxrender-dev
sudo apt-get install -y python-cloudpickle
sudo apt-get install -y python-dill
sudo apt-get install -y libharfbuzz-dev
sudo apt-get update && apt-get install -y python3-opencv
sudo apt-get install ffmpeg libsm6 libxext6  -y
python -m pip  install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --user
python -m pip install --upgrade pip --user
python -m pip install opencv-python --user
python -m pip install progressbar2 --user
python -m pip install tqdm --user
python -m pip install lxml --user
python -m pip install cloudpickle --user
python -m pip install dill --user
python -m pip install click --user
python -m pip install tensorflow --user
python -m pip install tensorflow-gpu --user
python -m pip install pandas --user
python -m pip install Pillow --user
python -m pip install -e . --user
python -m pip install PyHamcrest --user
python -m pip install gym --user
python -m pip install atari_py==0.2.6 --user
python -m pip install gym[atari] --user
python -m pip install gym[accept-rom-license] --user
python -m pip install scikit-image --user
python -m pip install sklearn --user
python -m pip install -U matplotlib --user
python -m pip install tensorboardX --user
python -m pip install gym_super_mario_bros --user
python -m pip install matplotlib --user
python -m pip install scipy --user