# Double Pendulum Neural Network Project

This project demonstrates training a neural network using PyTorch to balance a double pendulum. The pendulum starts close to the upright position with a slight random tilt at the beginning of each episode. During training, a live visualization shows the pendulum as the policy learns.

## Usage

Install the required dependencies:

```bash
pip install torch numpy matplotlib
```

Then run the training script:

```bash
python -m pendulum_nn.train
```

A matplotlib window will display the double pendulum while training progresses, and a plot of episode rewards will appear at the end.
