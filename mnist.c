#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define EPOCHS 20
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct {
  float *weights; // flattened array representing the weight matrix
  float *biases;  // array for the biases of each neuron
  int input_size,
      output_size; // number of inputs to layer and neurons in the layer
} Layer;

typedef struct {
  Layer hidden, output;
} Network;

typedef struct {
  unsigned char *images, *labels;
  int nImages;
} InputData;

void read_mnist_images(const char *filename, unsigned char **images,
                       int *nImages) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    exit(1);
  }

  int tmp, rows, cols;
  fread(&tmp, sizeof(int), 1, file);
  fread(nImages, sizeof(int), 1, file);
  *nImages = __builtin_bswap32(*nImages);

  fread(&rows, sizeof(int), 1, file);
  fread(&cols, sizeof(int), 1, file);

  rows = __builtin_bswap32(rows);
  rows = __builtin_bswap32(cols);

  *images = malloc((*nImages) * IMAGE_SIZE * IMAGE_SIZE);
  fread(*images, sizeof(unsigned char), (*nImages) * IMAGE_SIZE * IMAGE_SIZE,
        file);
  fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels,
                       int *nLabels) {
  FILE *file = fopen(filename, "rb");
  if (!file)
    exit(1);

  int tmp;
  fread(&tmp, sizeof(int), 1, file);
  fread(nLabels, sizeof(int), 1, file);
  *nLabels = __builtin_bswap32(*nLabels);
  *labels = malloc(*nLabels);
  fread(*labels, sizeof(unsigned char), *nLabels, file);
  fclose(file);
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    for (int k = 0; k < INPUT_SIZE; k++) {
      unsigned char temp = images[i * INPUT_SIZE + k];
      images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
      images[j * INPUT_SIZE + k] = temp;
    }
    unsigned char temp = labels[i];
    labels[i] = labels[j];
    labels[j] = temp;
  }
}

// initialzie the weights and biases of the init_layer function

void init_layer(Layer *layer, int in_size, int out_size) {
  int n = in_size * out_size;
  float scale = sqrtf(2.0f / in_size);

  layer->input_size = in_size;
  layer->output_size = out_size;
  layer->weights = malloc(n * sizeof(float));
  layer->biases = calloc(out_size, sizeof(float));

  // use he initialzation to set the weights and biases to 0
  for (int i = 0; i < n; i++) {
    layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
  }
}

// forward propagation / forward pass

// computing the output of a nural netwrok given an input
// involves moving the input data through each layer of the network and applying
// linear transformations and activation funcitons (relu, softmax, etc.)

void forward_pass(Layer *layer, float *input, float *output) {
  for (int i = 0; i < layer->output_size; i++) {
    output[i] = layer->biases[i];
    for (int j = 0; j < layer->input_size; j++) {
      output[i] += input[j] * layer->weights[j * layer->output_size + i];
    }
  }
}

// activation functions - defins how a weighted sum of an input is transformed
// into an output from a node in a network
// relu activation - rectified linear unit in the hidden layer

// for (int 1 = 0; i < HIDDEN_SIZE; i++) {
//   hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;
// }

// mathematically relu(x) = max(0,x)

// softmax activation - apply it to the output layers to compute the
// possibilities
void softmax(float *input, int size) {
  float max = input[0], sum = 0;
  for (int i = 1; i < size; i++) {
    if (input[i] > max)
      max = input[i];
  }
  for (int i = 0; i < size; i++) {
    input[i] = expf(input[i] - max);
    sum += input[i];
  }
  for (int i = 0; i < size; i++) {
    input[i] /= sum;
  }
}

// mathematically sig(z)_i = e^z_i/epsilon_j=1^K(e^z_j)

// forward propogation generates predictions necessary for trining and inference
// computes loss to quantify the error in training
// facilitates backpropagation by utilizing actications and intermediate outputs
// to compute gradients

// backpropagation / backwatd pass
// start by propagating the error from the output layer until reaching the input
// layer updates weights and biases based on the gradients

void backward_pass(Layer *layer, float *input, float *output_grad,
                   float *input_grad, float lr) {
  for (int i = 0; i < layer->output_size; i++) {
    for (int j = 0; j < layer->input_size; j++) {
      int idx = j * layer->output_size + i;
      // gradient calculation for weight update
      // gradient of loss wrt weight = gradient of loss wrt ouput * input
      float grad = output_grad[i] * input[j];
      // weight update
      // new weight = old weight-(learning rate * gradient of loss wrt weight)
      layer->weights[idx] -= lr * grad;
      // input gradient
      // gradient of loss wrt input j = sum of (gradient of loss wrt each output
      // i * weight connecting the input to the output) over all outputs
      if (input_grad) {
        input_grad[j] += output_grad[i] * layer->weights[idx];
      }
    }
    // bias update
    // new bias = old bias-(learning rate * gradient of loss wrt bias)
    layer->biases[i] -= lr * output_grad[i];
  }
}

// back propagation computes gradients to see how much each weight and bias
// affects the network's error updates params to adjust weights and biases to
// reduce the error in future predictions
// propagates error back through the layer allowing all layers to learn
//

// training
// training logic for a single input-label pair

void train(Network *net, float *input, int label, float lr) {
  float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];
  float output_grad[OUTPUT_SIZE] = {0}, hidden_grad[HIDDEN_SIZE] = {0};

  // forward pass - input2hidden
  forward_pass(&net->hidden, input, hidden_output);
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    hidden_output[i] =
        hidden_output[i] > 0 ? hidden_output[i] : 0; // relu activation
  }

  // forward pass - hidden2output
  forward_pass(&net->output, hidden_output, final_output);
  softmax(final_output, OUTPUT_SIZE);

  // compute ouput gradient
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    output_grad[i] = final_output[i] - (i == label);
  }

  // backward pass - output2hidden
  backward_pass(&net->output, hidden_output, output_grad, hidden_grad, lr);

  // backpropagate through relu
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0; // relu derivative
  }

  // backward pass - hidden2input
  backward_pass(&net->hidden, input, hidden_grad, NULL, lr);
}

int predict(Network *net, float *input) {
  float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

  forward_pass(&net->hidden, input, hidden_output);
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0; // relu
  }

  forward_pass(&net->output, hidden_output, final_output);
  softmax(final_output, OUTPUT_SIZE);

  int max_index = 0;
  for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (final_output[i] > final_output[max_index]) {
      max_index = i;
    }
  }
  return max_index;
}

int main() {
  Network net;
  InputData data = {0};
  float learning_rate = LEARNING_RATE, img[INPUT_SIZE];
  srand(time(NULL));
  init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
  init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);
  read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
  read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

  shuffle_data(data.images, data.labels, data.nImages);

  int train_size = (int)(data.nImages * TRAIN_SPLIT);
  int test_size = data.nImages - train_size;

  // training loop
  // training usually spans multiple epochs
  // each epoch involves running the entire dataset through the network

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    float total_loss = 0;
    for (int i = 0; i < train_size; i += BATCH_SIZE) {
      for (int j = 0; j < BATCH_SIZE && i + j < train_size; j++) {
        int idx = i + j;
        for (int k = 0; k < INPUT_SIZE; k++)
          img[k] = data.images[idx * INPUT_SIZE + k] / 255.0f;

        train(&net, img, data.labels[idx], learning_rate);

        // compute loss for monitoring
        float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];
        forward_pass(&net.hidden, img, hidden_output);
        for (int k = 0; k < HIDDEN_SIZE; k++) {
          hidden_output[k] =
              hidden_output[k] > 0 ? hidden_output[k] : 0; // relu
        }
        forward_pass(&net.output, hidden_output, final_output);
        softmax(final_output, OUTPUT_SIZE);

        total_loss += -logf(final_output[data.labels[idx]] + 1e-10f);
      }
    }
    // evaluate on test set
    int correct = 0;
    for (int i = train_size; i < data.nImages; i++) {
      for (int k = 0; k < INPUT_SIZE; k++) {
        img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
      }
      if (predict(&net, img) == data.labels[i]) {
        correct++;
      }
    }
    printf("epoch %d, accuracy: %.2f%%, avg loss: %.4f\n", epoch + 1,
           (float)correct / test_size * 100, total_loss / train_size);
  }

  free(net.hidden.weights);
  free(net.hidden.biases);
  free(net.output.weights);
  free(net.output.biases);
  free(data.images);
  free(data.labels);

  return 0;
}
