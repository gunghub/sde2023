### Reimplementation of Score-Based Generative Modeling through Stochastic Differential Equations

##### Overview

- `train.ipynb` is mainly used to train the models, including the baseline model and our score-based SDE model.
- `evaluate.ipynb` is used to sample from the trained models and evaluate the sampling results using the FID metric.
- `model.py` contains the UNet neural network.
- `sde.py` contains the functions and utilities needed for SDE.

##### Usage

- Execute `train.ipynb` to train baseline model and score-based SDE model. They will be saved in your project directory.

- Execute `evaluate.ipynb` to sample from the trained models and see the results. Sampling results are evaluated by FID. 

##### Reference

During development, we referred to the [score-based SDE code repository by Yang Song](https://github.com/yang-song/score_sde). 