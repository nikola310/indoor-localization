# Indoor localization with neural networks and co-training regressor

## Overview

Faculty project for subject "Business Informatics". The goal of this project was to find a way to use semi-supervised learning to try and determine location given the RSSI readings. Dataset is hosted on [Kaggle][3], provided by [Mohammadi et al.][1].

"explore-data" script is used for data processing and visualization. Aside from plotting data, it also scales all the RSSI readings, converts original location parameter to x and y coordinates, as well as extracting mutual differences of iBeacon RSSI values, as seen in paper by [Mohammadi et al.][1] This amounts to 78 new features which were tested later on. 

There are two experiments presented here:

1) Experiment with neural network:
   - This experiment is based off of [kernel from Kaggle][4], it trains autoencoder on unlabeled data and then decoder is replaced with regressor. For comparison, two scenarios were tested, one with basic features from dataset, other with new features.
2) Experiment with co-training regressor:
   - This experiment is based on CoReg, a co-training algorithm for regressors suggested by [Zhou et al.][2] CoReg is implemented and tested on two different scenarios, one with basic features from dataset, other with new, extracted features.

Results obtained with these experiments are similar to the result obtained in [Kaggle kernel][4].

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
4. Better indoor localization WIP: https://www.kaggle.com/ryches/better-indoor-localization-wip
5. COREG in Python: https://github.com/nealjean/coreg

[1]: https://arxiv.org/pdf/1810.04118.pdf
[2]: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/ijcai05.pdf
[3]: https://www.kaggle.com/mehdimka/ble-rssi-dataset
[4]: https://www.kaggle.com/ryches/better-indoor-localization-wip
[5]: https://github.com/nealjean/coreg

