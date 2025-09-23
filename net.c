#include "net.h"

#define NUM_IMAGES 10000
#define IMAGE_SIZE 784
#define NUM_IMAGES_TRAIN 60000

static double*** Conv_forprop(Layer *layer, double ***input_data);
static double*** Conv_backprop(Layer *layer, double*** output_error, double learning_rate);
static double* FC_backprop(Layer *layer, double *output_error, double learning_rate);
static double* FC_forprop(Layer *layer, double *input_data);
static double* activation_backprop(Layer *layer, double *output_error, double learning_rate);
static double* activation_forprop(Layer *layer, double *input_data);

double mean_squared_error(double* expected, double* result, int array_length)
{
	double error = 0.0;
	for (int i = 0; i < array_length; ++i) {
		double d = expected[i] - result[i];
		error += d * d;
	}
	return error / array_length;
}

void mean_squared_prime(double* expected, double* result, int array_length, double *output)
{
	for (int i = 0; i < array_length; ++i)
		output[i] = (2.0 * (expected[i] - result[i])) / array_length;
}

void relu_activation(double *input, int input_size, double *result)
{
	for (int i = 0; i < input_size; ++i)
		result[i] = input[i] > 0.0 ? input[i] : 0.0;
}

void relu_p(double *input, int input_size, double *result)
{
	for (int i = 0; i < input_size; ++i)
		result[i] = input[i] > 0.0 ? 1.0 : 0.0;
}

void tanh_activation(double *input, int input_size, double *result)
{
	for (int i = 0; i < input_size; ++i)
		result[i] = tanh(input[i]);
}

void tanh_p(double *input, int input_size, double *result)
{
	for (int i = 0; i < input_size; ++i) {
		double t = tanh(input[i]);
		result[i] = 1.0 - (t * t);
	}
}

Network* initNetwork(loss Loss, loss_prime Loss_prime)
{
	Network *net = malloc(sizeof(*net));
	if (!net)
		return NULL;
	net->loss_function = Loss;
	net->loss_function_prime = Loss_prime;
	net->head = NULL;
	net->tail = NULL;
	net->num_layers = 0;
	net->visualizer = 0;
	return net;
}

void addLayer(Network *net, Layer* layer)
{
	if (!net || !layer)
		return;
	if (!net->head) {
		net->head = net->tail = layer;
		layer->prev = layer->next = NULL;
	} else {
		net->tail->next = layer;
		layer->prev = net->tail;
		layer->next = NULL;
		net->tail = layer;
	}
	net->num_layers++;
}

double** predict(Network *net, int num_samples, int sample_size, double input_data[num_samples][sample_size])
{
	if (!net || !net->head || !net->tail)
		return NULL;

	double **result = malloc(num_samples * sizeof(double*));
	for (int i = 0; i < num_samples; ++i)
		result[i] = malloc(net->tail->output_size * sizeof(double));

	for (int i = 0; i < num_samples; ++i) {
		int in_size = net->head->input_size;
		double *input = malloc(in_size * sizeof(double));
		for (int j = 0; j < in_size; ++j)
			input[j] = input_data[i][j];

		Layer *curr = net->head;
		double *output = NULL;
		while (curr) {
			output = curr->forward_prop(curr, input);
			input = output;
			curr = curr->next;
		}

		for (int m = 0; m < net->tail->output_size; ++m)
			result[i][m] = output[m];

		curr = net->head;
		while (curr) {
			if (curr->input) {
				free(curr->input);
				curr->input = NULL;
			}
			if (curr->next == NULL && curr->output) {
				free(curr->output);
				curr->output = NULL;
			}
			curr = curr->next;
		}
	}

	return result;
}

void fit(Network *net, int num_samples, int sample_size, int sizeOfOutput, double x_train[num_samples][sample_size], double y_train[num_samples][sizeOfOutput], int epochs, double learning_rate)
{
	if (!net || !net->head || !net->tail)
		return;

	int input_shape = net->head->input_size;
	double *grad = malloc(net->tail->output_size * sizeof(double));

	for (int ep = 0; ep < epochs; ++ep) {
		double epoch_error = 0.0;

		for (int s = 0; s < num_samples; ++s) {
			double *input = malloc(input_shape * sizeof(double));
			for (int k = 0; k < input_shape; ++k)
				input[k] = x_train[s][k];

			Layer *curr = net->head;
			double *output = NULL;
			while (curr) {
				output = curr->forward_prop(curr, input);
				input = output;
				curr = curr->next;
			}

			epoch_error += net->loss_function(y_train[s], output, net->tail->output_size);
			net->loss_function_prime(y_train[s], output, net->tail->output_size, grad);

			curr = net->tail;
			double *e = grad;
			while (curr) {
				double *next_e = curr->backward_prop(curr, e, learning_rate);
				e = next_e;
				curr = curr->prev;
			}
			/* Make sure grad points to the latest buffer returned by backprop */
			grad = e;

			curr = net->head;
			while (curr) {
				if (curr->input) {
					free(curr->input);
					curr->input = NULL;
				}
				if (curr->next == NULL && curr->output) {
					free(curr->output);
					curr->output = NULL;
				}
				curr = curr->next;
			}
		}

		epoch_error /= num_samples;
		printf("epoch %d of %d, error=%f\n", ep + 1, epochs, epoch_error);
	}

	free(grad);
}

Layer* initActivation(activation a, activation_p ap, int input_size)
{
	Layer *layer = malloc(sizeof(*layer));
	if (!layer)
		return NULL;
	layer->forward_prop = activation_forprop;
	layer->backward_prop = activation_backprop;
	layer->Activation = a;
	layer->Ddx_activation = ap;
	layer->input_size = input_size;
	layer->output_size = input_size;
	layer->input = NULL;
	layer->output = NULL;
	layer->type = 1;
	return layer;
}

Layer* initConv2D(int num_filters, int filter_rows, int filter_cols, int num_channels, int stride, int padding)
{
	Layer *layer = malloc(sizeof(*layer));
	if (!layer)
		return NULL;

	if (num_filters <= 0 || filter_rows <= 0 || filter_cols <= 0 || num_channels <= 0 || stride <= 0) {
		free(layer);
		return NULL;
	}

	layer->channels = num_channels;
	layer->stride = stride;
	layer->padding = padding;
	layer->input = NULL;
	layer->output = NULL;
	layer->type = 2;
	layer->num_filters = num_filters;
	layer->filter_rows = filter_rows;
	layer->filter_cols = filter_cols;

	layer->convFilters = malloc(sizeof(double***) * num_filters);
	if (!layer->convFilters) {
		free(layer);
		return NULL;
	}

	for (int i = 0; i < num_filters; ++i) {
		layer->convFilters[i] = malloc(sizeof(double**) * filter_rows);
		if (!layer->convFilters[i])
			goto conv_alloc_fail;

		for (int r = 0; r < filter_rows; ++r) {
			layer->convFilters[i][r] = malloc(sizeof(double*) * filter_cols);
			if (!layer->convFilters[i][r])
				goto conv_alloc_fail;

			for (int c = 0; c < filter_cols; ++c) {
				layer->convFilters[i][r][c] = malloc(sizeof(double) * num_channels);
				if (!layer->convFilters[i][r][c])
					goto conv_alloc_fail;

				for (int ch = 0; ch < num_channels; ++ch) {
					double b = sqrt(6.0) / sqrt(4.0 + 4.0);
					double a = -b;
					double random_double = a + (((double)rand() / RAND_MAX) * (b - a));
					layer->convFilters[i][r][c][ch] = random_double;
				}
			}
		}
	}

	return layer;

conv_alloc_fail:
	for (int ii = 0; ii < num_filters; ++ii) {
		if (!layer->convFilters[ii])
			continue;
		for (int rr = 0; rr < filter_rows; ++rr) {
			if (!layer->convFilters[ii][rr])
				continue;
			for (int cc = 0; cc < filter_cols; ++cc)
				free(layer->convFilters[ii][rr][cc]);
			free(layer->convFilters[ii][rr]);
		}
		free(layer->convFilters[ii]);
	}
	free(layer->convFilters);
	free(layer);
	return NULL;
}

Layer* initFC(int input_size, int output_size)
{
	Layer* layer = malloc(sizeof(*layer));
	if (!layer)
		return NULL;

	layer->input_size = input_size;
	layer->output_size = output_size;

	layer->bias = malloc(output_size * sizeof(double));
	if (!layer->bias) {
		free(layer);
		return NULL;
	}

	for (int i = 0; i < output_size; ++i)
		layer->bias[i] = ((double)rand() / RAND_MAX) * 0.6 - 0.3;

	layer->weights = malloc(input_size * sizeof(double*));
	if (!layer->weights) {
		free(layer->bias);
		free(layer);
		return NULL;
	}

	for (int i = 0; i < input_size; ++i) {
		layer->weights[i] = malloc(output_size * sizeof(double));
		if (!layer->weights[i]) {
			for (int j = 0; j < i; ++j)
				free(layer->weights[j]);
			free(layer->weights);
			free(layer->bias);
			free(layer);
			return NULL;
		}

		for (int j = 0; j < output_size; ++j) {
			double b = sqrt(6.0) / sqrt(input_size + output_size);
			double a = -b;
			double random_double = a + (((double)rand() / RAND_MAX) * (b - a));
			layer->weights[i][j] = random_double;
		}
	}

	layer->forward_prop = FC_forprop;
	layer->backward_prop = FC_backprop;
	layer->input = NULL;
	layer->output = NULL;
	layer->type = 0;
	return layer;
}

static double*** Conv_forprop(Layer *layer, double ***input_data)
{
	(void)layer;
	(void)input_data;
	return NULL;
}

static double*** Conv_backprop(Layer *layer, double*** output_error, double learning_rate)
{
	(void)layer;
	(void)output_error;
	(void)learning_rate;
	return NULL;
}

static double* activation_forprop(Layer *layer, double *input_data)
{
	if (!layer)
		return NULL;

	double *result = malloc(layer->output_size * sizeof(double));
	if (!result)
		return NULL;

	layer->input = input_data;
	layer->Activation(layer->input, layer->input_size, result);
	layer->output = result;
	return result;
}

static double* activation_backprop(Layer *layer, double *output_error, double learning_rate)
{
	(void)learning_rate;
	double act[layer->output_size];
	layer->Ddx_activation(layer->input, layer->input_size, act);
	for (int i = 0; i < layer->input_size; ++i)
		act[i] *= output_error[i];

	double *temp = realloc(output_error, layer->input_size * sizeof(double));
	if (!temp)
		return NULL;

	for (int i = 0; i < layer->input_size; ++i)
		temp[i] = act[i];

	return temp;
}

static double* FC_forprop(Layer *layer, double *input_data)
{
	double *result = malloc(layer->output_size * sizeof(double));
	layer->input = input_data;
	for (int col = 0; col < layer->output_size; ++col) {
		double acc = 0.0;
		for (int row = 0; row < layer->input_size; ++row)
			acc += layer->weights[row][col] * input_data[row];
		result[col] = acc + layer->bias[col];
	}
	layer->output = result;
	return result;
}

static double* FC_backprop(Layer *layer, double *output_error, double learning_rate)
{
	double *input_error = malloc(layer->input_size * sizeof(double));
	for (int i = 0; i < layer->input_size; ++i) {
		double acc = 0.0;
		for (int j = 0; j < layer->output_size; ++j)
			acc += layer->weights[i][j] * output_error[j];
		input_error[i] = acc;
	}

	double *weights_error = malloc(layer->input_size * layer->output_size * sizeof(double));
	for (int i = 0; i < layer->input_size; ++i) {
		for (int j = 0; j < layer->output_size; ++j)
			weights_error[i * layer->output_size + j] = layer->input[i] * output_error[j];
	}

	for (int i = 0; i < layer->output_size; ++i)
		layer->bias[i] += learning_rate * output_error[i];

	for (int i = 0; i < layer->input_size; ++i)
		for (int j = 0; j < layer->output_size; ++j)
			layer->weights[i][j] += learning_rate * weights_error[i * layer->output_size + j];

	free(weights_error);

	double *interim = realloc(output_error, layer->input_size * sizeof(double));
	if (!interim) {
		free(input_error);
		return NULL;
	}

	for (int i = 0; i < layer->input_size; ++i)
		interim[i] = input_error[i];

	free(input_error);
	return interim;
}

void destroyNetwork(Network *net)
{
	if (!net)
		return;

	Layer *curr = net->head;
	while (curr) {
		Layer *next = curr->next;
		if (curr->type == 0) {
			for (int i = 0; i < curr->input_size; ++i)
				free(curr->weights[i]);
			free(curr->weights);
			free(curr->bias);
		} else if (curr->type == 2) {
			for (int f = 0; f < curr->num_filters; ++f) {
				if (!curr->convFilters[f])
					continue;
				for (int r = 0; r < curr->filter_rows; ++r) {
					if (!curr->convFilters[f][r])
						continue;
					for (int c = 0; c < curr->filter_cols; ++c)
						free(curr->convFilters[f][r][c]);
					free(curr->convFilters[f][r]);
				}
				free(curr->convFilters[f]);
			}
			free(curr->convFilters);
		}
		free(curr);
		curr = next;
	}

	free(net);
}

void enableVisualizer(Network *net, int flag)
{
	if (!net)
		return;
	net->visualizer = flag ? 1 : 0;
}

void reshapeImages(uint8_t *flatData, double (*reshapedData)[NUM_IMAGES][28][28])
{
	for (int i = 0; i < NUM_IMAGES; ++i)
		for (int row = 0; row < 28; ++row)
			for (int col = 0; col < 28; ++col)
				(*reshapedData)[i][row][col] = flatData[i * IMAGE_SIZE + row * 28 + col];
}

int reverseInt (int i)
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int main(void)
{
	// Seed random for weight init
	srand((unsigned)time(NULL));

	FILE *file = fopen("MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", "rb");
	if (!file) {
		perror("Error opening test images file");
		return 1;
	}

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	size_t x = fread((char*)&magic_number, sizeof(magic_number), 1, file);
	magic_number = reverseInt(magic_number);
	x = fread((char*)&number_of_images, sizeof(number_of_images), 1, file);
	number_of_images = reverseInt(number_of_images);
	x = fread((char*)&n_rows, sizeof(n_rows), 1, file);
	n_rows = reverseInt(n_rows);
	x = fread((char*)&n_cols, sizeof(n_cols), 1, file);
	n_cols = reverseInt(n_cols);

	double (*testImages)[28][28] = malloc(number_of_images * sizeof(*testImages));
	if (!testImages) {
		fprintf(stderr, "Out of memory for testImages\n");
		fclose(file);
		return 1;
	}

	for (int i = 0; i < number_of_images; ++i)
		for (int r = 0; r < n_rows; ++r)
			for (int c = 0; c < n_cols; ++c) {
				unsigned char temp = 0;
				x = fread((char*)&temp, sizeof(temp), 1, file);
				testImages[i][r][c] = (double) temp;
			}
	fclose(file);

	FILE *file2 = fopen("MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte", "rb");
	if (!file2) {
		perror("Error opening train images file");
		free(testImages);
		return 1;
	}

	int magic_number_train = 0;
	int number_of_images_train = 0;
	int n_rows_train = 0;
	int n_cols_train = 0;
	x = fread((char*)&magic_number_train, sizeof(magic_number_train), 1, file2);
	magic_number_train = reverseInt(magic_number_train);
	x = fread((char*)&number_of_images_train, sizeof(number_of_images_train), 1, file2);
	number_of_images_train = reverseInt(number_of_images_train);
	x = fread((char*)&n_rows_train, sizeof(n_rows_train), 1, file2);
	n_rows_train = reverseInt(n_rows_train);
	x = fread((char*)&n_cols_train, sizeof(n_cols_train), 1, file2);
	n_cols_train = reverseInt(n_cols_train);

	double (*train_images)[28][28] = malloc(number_of_images_train * sizeof(*train_images));
	if (!train_images) {
		fprintf(stderr, "Out of memory for train_images\n");
		free(testImages);
		fclose(file2);
		return 1;
	}

	for (int i = 0; i < number_of_images_train; ++i)
		for (int r = 0; r < n_rows_train; ++r)
			for (int c = 0; c < n_cols_train; ++c) {
				unsigned char temp = 0;
				x = fread((char*)&temp, sizeof(temp), 1, file2);
				train_images[i][r][c] = (double) temp;
			}
	fclose(file2);

	FILE *file3 = fopen("MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte", "rb");
	if (!file3) {
		perror("Error opening test labels file");
		free(testImages);
		free(train_images);
		return 1;
	}
	int magic_number_labels = 0;
	x = fread((char*)&magic_number_labels, sizeof(magic_number_labels), 1, file3);
	magic_number_labels = reverseInt(magic_number_labels);
	int number_of_labels = 0;
	x = fread((char*)&number_of_labels, sizeof(number_of_labels), 1, file3);
	number_of_labels = reverseInt(number_of_labels);
	double *test_labels = malloc(number_of_labels * sizeof(double));
	for (int i = 0; i < number_of_labels; ++i) {
		unsigned char temp = 0;
		x = fread((char*)&temp, sizeof(temp), 1, file3);
		test_labels[i] = (double) temp;
	}
	fclose(file3);

	FILE *file4 = fopen("MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte", "rb");
	if (!file4) {
		perror("Error opening train labels file");
		free(testImages);
		free(train_images);
		free(test_labels);
		return 1;
	}
	int magic_number_labels_train = 0;
	x = fread((char*)&magic_number_labels_train, sizeof(magic_number_labels_train), 1, file4);
	magic_number_labels_train = reverseInt(magic_number_labels_train);
	int number_of_labels_train = 0;
	x = fread((char*)&number_of_labels_train, sizeof(number_of_labels_train), 1, file4);
	number_of_labels_train = reverseInt(number_of_labels_train);
	double *train_labels = malloc(number_of_labels_train * sizeof(double));
	for (int i = 0; i < number_of_labels_train; ++i) {
		unsigned char temp = 0;
		x = fread((char*)&temp, sizeof(temp), 1, file4);
		train_labels[i] = (double) temp;
	}
	fclose(file4);

	// Train on a subset of our data for quick testing
	int num_train = 2000;
	if (num_train > number_of_images_train)
		num_train = number_of_images_train;

	
	double (*x_train)[IMAGE_SIZE] = malloc(num_train * sizeof(*x_train));
	double (*y_train)[10] = malloc(num_train * sizeof(*y_train));
	if (!x_train || !y_train) {
		fprintf(stderr, "Out of memory for training arrays\n");
		free(testImages);
		free(train_images);
		free(test_labels);
		free(train_labels);
		free(x_train);
		free(y_train);
		return 1;
	}

	for (int i = 0; i < num_train; ++i) {
		for (int r = 0; r < 28; ++r)
			for (int c = 0; c < 28; ++c)
				x_train[i][r * 28 + c] = train_images[i][r][c] / 255.0;

		for (int k = 0; k < 10; ++k)
			y_train[i][k] = 0.0;
		int lbl = (int)train_labels[i];
		if (lbl >= 0 && lbl < 10)
			y_train[i][lbl] = 1.0;
	}

	Network *net = initNetwork(mean_squared_error, mean_squared_prime);
	Layer *l1 = initFC(IMAGE_SIZE, 128);
	Layer *a1 = initActivation(relu_activation, relu_p, 128);
	Layer *l2 = initFC(128, 10);
	addLayer(net, l1);
	addLayer(net, a1);
	addLayer(net, l2);
	int epochs = 5;
	double lr = 0.01;
	fit(net, num_train, IMAGE_SIZE, 10, x_train, y_train, epochs, lr);

	// evaluation
	int num_test_eval = 1000;
	if (num_test_eval > number_of_images)
		num_test_eval = number_of_images;

	double (*x_test)[IMAGE_SIZE] = malloc(num_test_eval * sizeof(*x_test));
	if (!x_test) {
		fprintf(stderr, "Out of memory for x_test\n");
		destroyNetwork(net);
		free(testImages);
		free(train_images);
		free(test_labels);
		free(train_labels);
		free(x_train);
		free(y_train);
		return 1;
	}
	for (int i = 0; i < num_test_eval; ++i) {
		for (int r = 0; r < 28; ++r)
			for (int c = 0; c < 28; ++c)
				x_test[i][r * 28 + c] = testImages[i][r][c] / 255.0;
	}

	double **pred = predict(net, num_test_eval, IMAGE_SIZE, x_test);
	int correct = 0;
	for (int i = 0; i < num_test_eval; ++i) {
		int pred_label = 0;
		double best = pred[i][0];
		for (int k = 1; k < 10; ++k) {
			if (pred[i][k] > best) {
				best = pred[i][k];
				pred_label = k;
			}
		}
		if (pred_label == (int)test_labels[i])
			++correct;
	}
	printf("Test accuracy on %d samples: %.2f%%\n", num_test_eval, 100.0 * correct / num_test_eval);

	// Cleanup/free allocated memory
	for (int i = 0; i < num_test_eval; ++i)
		free(pred[i]);
	free(pred);
	free(x_test);
	free(x_train);
	free(y_train);
	free(testImages);
	free(train_images);
	free(test_labels);
	free(train_labels);
	destroyNetwork(net);
	return 0;
}
