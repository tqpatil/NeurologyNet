# Neurology

![image](https://github.com/tqpatil/NeurologyNet/assets/34226808/f0eb27d6-c078-46a3-9c05-334c166c4df7)

<img width="380" alt="image" src="https://github.com/tqpatil/NeurologyNet/assets/34226808/79ffe2f7-d055-416e-ac4c-d2653456e8d4">

Creating an infrastructure for building neural nets fully in C.

## Features
- Multi-threaded training with configurable thread pool
- Parallelized layer computations (FC, Conv, Activation, MaxPool, Softmax)

## Layer types
- Fully Connected (FC) Layer
	Params: 
	1. Input size (int)
	2. Output size (int)
- Activation Layer
	Params:
	1. Activation Function (activation)
	2. Derivative of Activation Function (activation_p)
	3. Input size (int)
- Conv2D (Convolutional)
	Params:
	1. Num filters (int)
	2. Filter rows (int)
	3. Filter cols (int)
	4. Num channels (int)
	5. Stride (int)
	6. Padding (int)
- MaxPool Layer
	Params:
	1. Num channels (int)
	2. Input height (int)
	3. Input width (int)
	4. Pool rows (int)
	5. Pool cols (int)
	6. Stride (int)
- Flatten Layer
	Params:
	1. Num filters (int)
	2. Height (int)
	3. Width (int)
- Network
	Params:
	1. Loss function (loss)
	2. Derivative of Loss function (loss_prime)
## Error Functions
	1. Mean Squared Error(mean_squared_error(), mean_squared_prime())
	2. Crossentropy (cross_entropy_loss(), cross_entropy_prime())
## Activation Functions
	1. Tanh(tanh_activation(), tanh_p())
	2. ReLu(relu_activation(), relu_p())
## Key Functions
	1. addLayer(Network *net, Layer *layer); // Adds layer to network
	2. initFC(int input_size, int output_size); //Creates fully connected layer
	3. initActivation(activation funcA, activation_p funcB, int input_size); // Create activation layer (given activation function)
	4. fit(Network *net, int num_samples, int sample_length, int networkOutputSize, double xtrain[][], double ytrain[][], int epochs, double learning_rate);
	5. double **predict(Network *net, int num_samples, int sample_length, double data[][]);
	6. destroyNetwork(Network *net); // Object destroyer
	7. initConv2D(int num_filters, int filter_rows, int filter_cols, int num_channels, int stride, int padding);
	8. initMaxPool(int num_channels, int input_height, int input_width, int pool_rows, int pool_cols, int stride);
	9. initFlatten(int num_filters, int height, int width);
	10. setThreadPoolSize(Network *net, int num_threads);
 
### To Do:
  	1. Adam Optimizer
   	2. Regularization / Dropout

