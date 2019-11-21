# Indoor localization with neural networks and co-training regressor

## Overview

This repository contains my work on a project during my master studies. The goal of the project was to find a way to use semi-supervised learning to try and determine location given the RSSI readings. Dataset is hosted on Kaggle [3], provided by [1].

There are two experiments I did with the data:

1) Experiment with neural network:
  - This experiment is based off of kernel [4], it trains autoencoder on unlabeled data and then decoder is replaced with regressor. Two scenarios are tested, one with basic features from dataset, other with new features.
2) Experiment with co-training regressor:
  - This experiment is based on CoReg, a co-training algorithm for regressors suggested in [2]. CoReg is implemented and tested on two different scenarios, one with basic features from dataset, other with new, extracted features.
## Repository structure

"processing-data" folder contains Python script and data used for data processing and visualization. Aside from plotting data, it also scales all the RSSI readings, converts original location parameter to x and y coordinates, as well as extracting extra features, as seen [1].

## External dependencies:

  - sklearn
  - pandas
  - tensorflow 2.0
  - numpy
  - matplotlib
  - seaborn
  
## References

1. M. Mohammadi, A. Al-Fuqaha, M. Guizani, J. Oh, “Semi-supervised Deep Reinforcement Learning in Support of IoT and Smart City Services,” IEEE Internet of Things Journal, Vol. PP, No. 99, 2017.
2. Zhou, Zhi-Hua, and Ming Li. "Semi-Supervised Regression with Co-Training." In IJCAI, vol. 5, pp. 908-913. 2005.
3. BLE RSSI Dataset for Indoor localization: https://www.kaggle.com/mehdimka/ble-rssi-dataset
4. Better indoor localization WIP: https://www.kaggle.com/mehdimka/ble-rssi-dataset 
