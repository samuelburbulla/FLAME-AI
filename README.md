# FLAME-AI

My submission to the [**2023 FLAME AI Challenge**: *Flow Physics Crowd-science Challenge for Turbulent Super-resolution*](https://flame-ai-workshop.github.io).
The submission was ranked #10 on the [Kaggle leaderboard](https://www.kaggle.com/competitions/2023-flame-ai-challenge/leaderboard).



## Content

Training script: `flame.py`

Model parameters: `working/model.pt`

Submission file: `working/submission.csv`


## Usage

Requirements: `torch`, `numpy`, `pandas`, `matplotlib`.

Train the model by running `python flame.py`.

If you want run the script with trained model weights, set `load_model_weights = True` in `flame.py`.


## Visualization

### Input
![](https://github.com/samuelburbulla/FLAME-AI/blob/main/working/lr.png?raw=true)

### Prediction
![](https://github.com/samuelburbulla/FLAME-AI/blob/main/working/hr_pred.png?raw=true)

### Ground truth
![](https://github.com/samuelburbulla/FLAME-AI/blob/main/working/hr.png?raw=true)