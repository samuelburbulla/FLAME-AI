# FLAME-AI

My submission to the [**2023 FLAME AI Challenge**: *Flow Physics Crowd-science Challenge for Turbulent Super-resolution*](https://flame-ai-workshop.github.io).
The submission was ranked #10 on the [Kaggle leaderboard](https://www.kaggle.com/competitions/2023-flame-ai-challenge/leaderboard).


## Model description

The proposed model is a specialized convolutional neural network (CNN) designed for the task of upsampling low-resolution images.
First, the model inputs of size 16x16x4 are passed through an upsampling layer with bilinear interpolation to increase the spatial resolution to 128x128x4 pixels.
Subsequently, a sequence of convolutional layers processes the upsampled data to improve prediction accuracy with respect to the given the training data.
We carefully optimized the hyper-parameters of the architecture and found that 4 layers mapping to 16 latent channels, together with a large kernel size of 15 and padding of 7, are a good trade-off between accuracy and training time.
Notably, we added a skip connection around the convolutional layer block.
The model is trained in two stages: First, we perform 10000 steps using Adam on a random batch of size 16 and, afterwards, fine-tune with LBFGS on batches of size 128 for another 1000 iterations.


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