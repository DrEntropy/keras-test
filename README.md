# keras-test

A learning exercise exploring Keras 3's multi-backend support, comparing JAX and PyTorch performance on a Mac Studio M4 and a Google Colab T4 GPU. This uses example code from Keras 3 documentation.

## Notebooks

### mnist.ipynb — MNIST Digit Classification

A CNN for handwritten digit recognition (28x28 grayscale images). Achieves ~99.1% test accuracy after 10 epochs.

**Model:** Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dropout(0.5) → Dense(10, softmax)

### cats_dogs.ipynb — Cats vs Dogs Binary Classification

An Xception-inspired architecture for classifying 180x180 color images. Uses residual blocks with separable convolutions, data augmentation, and model checkpointing.  Training not run to completion yet, just used for performance comparison. 

## Backend Performance Comparison

Keras 3 lets you swap backends via an environment variable. Timing is per training step:

| Task      | Backend | Mac Studio M4 | Colab T4 GPU |
| --------- | ------- | -------------- | ------------ |
| MNIST     | JAX     | 12 ms/step     | 2 ms/step    |
| MNIST     | PyTorch | 20 ms/step     | 10 ms/step   |
| Cats/Dogs | JAX     | ~15 s/step     | ~2 s/step    |
| Cats/Dogs | PyTorch | ~1 s/step      | did not test |

For the smaller MNIST model, on the mac, suprisingly JAX(cpu) was slightly  faster then PyTorch(mps).   On the T4 GPU, JAX was about 5x faster than PyTorch. For the larger Cats/Dogs model, JAX on the mac was much slower than PyTorch, but on the T4 GPU it was much faster. 

## Setup

Requires Python 3.12+. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Key dependencies: keras, jax, torch, torchvision, tensorflow, matplotlib, jupyter.

The cats/dogs notebook downloads the Microsoft Cats and Dogs dataset (~7.8 GB) on first run.
