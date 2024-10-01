# mnist-c

trying to implement a feature complete mnist in pure c

## demo

```bash
gcc -o nn nn.c -lm
./nn
```

```
epoch 1, accuracy: 90.84%, avg loss: 0.4684
epoch 2, accuracy: 92.61%, avg loss: 0.2339
epoch 3, accuracy: 93.77%, avg loss: 0.1863
epoch 4, accuracy: 94.46%, avg loss: 0.1539
epoch 5, accuracy: 95.03%, avg loss: 0.1300
epoch 6, accuracy: 95.34%, avg loss: 0.1117
epoch 7, accuracy: 95.78%, avg loss: 0.0973
epoch 8, accuracy: 96.02%, avg loss: 0.0857
epoch 9, accuracy: 96.17%, avg loss: 0.0762
epoch 10, accuracy: 96.38%, avg loss: 0.0683
epoch 11, accuracy: 96.51%, avg loss: 0.0617
epoch 12, accuracy: 96.57%, avg loss: 0.0560
epoch 13, accuracy: 96.62%, avg loss: 0.0511
epoch 14, accuracy: 96.65%, avg loss: 0.0468
epoch 15, accuracy: 96.73%, avg loss: 0.0430
epoch 16, accuracy: 96.78%, avg loss: 0.0397
epoch 17, accuracy: 96.86%, avg loss: 0.0368
epoch 18, accuracy: 96.98%, avg loss: 0.0342
epoch 19, accuracy: 97.04%, avg loss: 0.0318
epoch 20, accuracy: 97.07%, avg loss: 0.0297

```

## todo

- [ ] very slow compared to even python libraries - optimize
- [ ] implement model saving and inference from saved model

## notes

three layers - input, hidden and output
neurons in each layer determined by the characteristics of the dataset we are using

mnist dataset - 28x28 greyscale matrix handwritten digits - 60000 training images and 10000 testing images
flattening results in -> array or 784 -> input layer
10 neurons (0-9) in output layer

hidden layer -> 256 neurons (why?)

### processing input data

read images and labels from IDX file format

open idx in binary mode
read and convert the header information (num images, rows, columns)
allocate memory and read the image pixel data and labels

### implementing neural network structure

layer struct to represent each layer in the network

### references

- [dataset link](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download)
- https://github.com/konrad-gajdus/miniMNIST-c
