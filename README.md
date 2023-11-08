# Neurology
Creating an infrastructure for building neural nets fully in C.
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
- Network
	Params:
	1. Loss function (loss)
	2. Derivative of Loss function (loss_prime)
## Error Functions
	1. Mean Squared Error(mean_squared_error(), mean_squared_prime())
	2. Crossentropy (in progress)
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
### To Do:
	1. Convolution Layer
 	2. Pooling Layer
  	3. Adam Optimizer
   	4. Softmax/Crossentropy loss
    	5. OpenCL support
