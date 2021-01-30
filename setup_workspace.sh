#!/bin/bash

# setup script for setting up workspace on a linux machine
# NOTE: install detectron2 step does not work for mac, follow steps in link below
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source

git config --local core.hooksPath .githooks/

read -p "Download and init miniconda? (y/n)?: " input
if [[ $input == "y" ]]; then
  # install conda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
  bash ~/miniconda.sh -b -p $HOME/miniconda
  
  # setup conda env
  source ~/miniconda/bin/activate
  conda init
fi

# create new conda env
conda create -y --name prostate python=3.6
echo "conda activate prostate" >> ~/.bashrc
conda activate prostate

# install detectron2 + dependencies
pip install pyyaml==5.1 pycocotools==2.0.1 gdown
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -y pandas
conda install -y -c conda-forge opencv
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
# make sure installation contains no errors
python -m detectron2.utils.collect_env

# install rest of dependencies
conda env update --name prostate --file env.yml
