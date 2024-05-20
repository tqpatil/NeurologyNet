#include "net.h"
#define NUM_IMAGES 10000
#define IMAGE_SIZE 784 // 28x28
#define NUM_IMAGES_TRAIN 60000
//nvcc compiles files with C++ linkage. 
// If you want to link and call functions from a C file, cudatest must be declared extern “C”.
double mean_squared_error(double* expected, double* result, int array_length){
	double error=0;
	for (int i=0; i<array_length; i++){
		error += ((expected[i] - result[i]) * (expected[i] - result[i]));
	}
	error = error/array_length;
	return error;
}
void mean_squared_prime(double* expected, double* result, int array_length, double *output){
	for (int i=0; i<array_length; i++){
		output[i] = (2* (expected[i] - result[i]))/array_length;
	}
	
}
void relu_activation(double *input, int input_size, double *result){
	for(int i=0; i<input_size; i++){
		if(input[i] > 0){
			result[i] = input[i];
		}
		else{
			result[i] = 0;
		}
	}
}
void relu_p(double *input, int input_size, double *result){
	for(int i=0; i<input_size; i++){
		if(input[i]>0){
			result[i] = 1;
		}
		else{
			result[i] = 0;
		}
	}
}
void tanh_activation(double *input, int input_size, double *result){
	for (int i=0; i< input_size; i++){
		double temp = tanh(input[i]);
		result[i] = temp;
	}
}
void tanh_p(double *input, int input_size, double *result){
	double temp;
	for (int i=0; i< input_size; i++){
		temp = tanh(input[i]);
		result[i] = (1-(temp * temp));
	}
}

Network* initNetwork(loss Loss, loss_prime Loss_prime){ // Initialize network as a linked list of layers
	Network *net = (Network *)malloc(sizeof(Network));
	if (net == NULL){
		fprintf(stderr, "Failed to allocate network\n");
		return NULL;
	}
	net->loss_function = Loss;
	net->loss_function_prime = Loss_prime;
	net->head = NULL;
	net->tail = NULL;
	net->num_layers = 0;
	return net;
}
void addLayer(Network *net, Layer* layer){ // Add layer to end of network
	if (layer == NULL){
		fprintf(stderr, "Cannot add NULL layer to network\n");
		return; 
	}
	if (net->head == NULL){
		net->head = layer;
		net->tail = layer;
		layer->next = NULL;
		layer->prev = NULL;
		net->num_layers += 1;
	}
	else{
		net->tail->next = layer;
		layer->prev = net->tail; 
		net->tail = layer;
		layer->next = NULL;
		net->num_layers += 1;
	}
}
double** predict(Network *net, int num_samples, int sample_size, double input_data[num_samples][sample_size]){ // run forward pass
	double **result = (double**)malloc(num_samples*sizeof(double *));
	for (int i=0; i<num_samples; i++){
		result[i] = (double*)malloc(net->tail->output_size*sizeof(double));
	}
	for(int i=0; i<num_samples; i++){
		double *input = (double*) malloc(net->head->input_size * sizeof(double));
        	double *output;
		for (int j=0; j<net->head->input_size; j++){
			input[j] = input_data[i][j];
		}
		Layer *curr = net->head;
		while(curr != NULL){
			output= curr->forward_prop(curr, input);
			input = output;
			curr= curr->next;
		}
		for (int m=0; m<net->tail->output_size; m++){
			result[i][m]= output[m];
		}
		curr= net->head;
        	while(curr != NULL){
                	if(curr->input != NULL){
                        	free(curr->input);
                        	curr->input = NULL;
                	}
                	if(curr->next == NULL){
                        	if(curr->output != NULL){
                                	free(curr->output);
                                	curr->output = NULL;
                        	}
                	}
                	curr= curr->next;
        	}
		//copy output into result[i];	

	}
	/*
	printf("fml\n");
	Layer *curr= net->head;
	while(curr != NULL){
		if(curr->input != NULL){
			free(curr->input);
			curr->input = NULL;
		}
		if(curr->next == NULL){
			if(curr->output != NULL){
				free(curr->output);
				curr->output = NULL;
			}
		}
		curr= curr->next;
	}
	*/
	return result;
}
void fit(Network *net, int num_samples, int sample_size, int sizeOfOutput, double x_train[num_samples][sample_size], double y_train[num_samples][sizeOfOutput], int epochs, double learning_rate){ // Train network through backpropagation for n epochs
	int input_shape = net->head->input_size; 
	double *t;
	double error;
	double *e=(double *)malloc(net->tail->output_size * sizeof(double)); 
	for (int i=0; i < epochs; i++){
		error = 0;
		for (int j=0; j<num_samples; j++){
			double *input = (double*) malloc(input_shape * sizeof(double));
        		double *output;
			for (int k=0; k<input_shape; k++){
				input[k] = x_train[j][k];
			}
			Layer *curr = net->head;
			while (curr != NULL){
				output = curr->forward_prop(curr, input);
				input = output;
				curr = curr->next;
			}
			
			error+= net->loss_function(y_train[j], output, net->tail->output_size);
			net->loss_function_prime(y_train[j], output, net->tail->output_size, e);
			curr = net->tail;
			while(curr != NULL){
				t = curr->backward_prop(curr, e, learning_rate);
				e=t;
				curr = curr->prev;
			}
			curr= net->head;
                	while(curr != NULL){
                        	if(curr->input != NULL){
                                	free(curr->input);
                                	curr->input = NULL;
                        	}
                        	if(curr->next == NULL){
                                	if(curr->output != NULL){
                                        	free(curr->output);
                                        	curr->output = NULL;
                                	}
                        	}
                        	curr= curr->next;
                	}

		}
		error = error/num_samples; 
		printf("epoch %d of %d, error=%f\n", i+1, epochs, error); 
	}
	free(e);
}
Layer* initActivation(activation a, activation_p ap, int input_size){ // Using generic type Layer, set fields for Activation layer
	Layer *layer = (Layer *)malloc(sizeof(Layer));
	if (layer==NULL){
		fprintf(stderr, "Failed to allocate activation layer\n");
		return NULL;
	}
	layer->forward_prop = activation_forprop;
	layer->backward_prop = activation_backprop;
	layer->Activation = a;
	layer->Ddx_activation = ap;
	layer->input_size = input_size;
	layer->output_size = input_size;
	layer->input = NULL;
	layer->output = NULL;
	layer->type = 1;
	//set layer forward prop and back prop functions
	return layer;
}
// What about channels on an immage?
// alternative is just to have the filters be 4d array but make the channels 1 for grayscale images
// i think this is better 
// figure out the exact logistics for this because allocating the filters as a pointer map probably
// risks memory fragmentation. Maybe want to just keep the 4d array as a 1d list and use index arithmetic. 
Layer* initConv2D(int num_filters, int filter_rows, int filter_cols, int num_channels, int stride, int padding){
	srand(time(NULL));
	Layer *layer = (Layer *)malloc(sizeof(Layer));
	if (layer==NULL){
		fprintf(stderr, "Failed to allocate activation layer\n");
		return NULL;
	}
	if((num_filters <= 0) || (filter_rows <= 0) || (filter_cols <=0) || (num_channels <= 0) || (stride <= 0)){
		fprintf(stderr, "None of the params to: initConv2D may be <= 0. ");
		free(layer);
		return NULL;
	}
	// layer->forward_prop = Conv_forprop; // damn, all my function pointers are gonna have to be different now
	// layer->backward_prop = Conv_backprop;
	layer->channels = num_channels;
	layer->stride = stride; 
	layer->padding = padding;
	layer->input = NULL;
	layer->output = NULL;
	layer->type = 2;
	layer->num_filters = num_filters;
	layer->filter_rows = filter_rows;
	layer->filter_cols = filter_cols;
	layer->convFilters = (double ****) malloc(sizeof(double***)*num_filters);
	if(layer->convFilters == NULL){
		fprintf(stderr, "Failed to allocate memory\n");
		free(layer);
		return NULL;
	}
	for(int i=0; i< num_filters; i++){
		layer->convFilters[i] = (double***) malloc(sizeof(double**) * filter_rows);
		if (layer->convFilters[i] == NULL) { // Testing this is painful... 
			for (int j = 0; j < i; j++) {
				for(int w=0; w<filter_rows; w++){
					for(int y = 0; y<filter_cols; y++){
						free(layer->convFilters[j][w][y]);
					}
					free(layer->convFilters[j][w]);
				}
                free(layer->convFilters[j]);
            }
            free(layer->convFilters);
            free(layer);
            return NULL;
		}
		for(int k=0; k< filter_rows; k++){
			layer->convFilters[i][k] = (double **) malloc(sizeof(double*) * filter_cols);
			if (layer->convFilters[i][k] == NULL) {
				for (int j = 0; j<k; j++){
					free(layer->convFilters[i][j]);
				}
				free(layer->convFilters[i]);
				for (int j = 0; j < i; j++) {
					for(int m =0; m<filter_rows; m++){
						for(int y = 0; y<filter_cols; y++){
							free(layer->convFilters[j][m][y]);
						}
						free(layer->convFilters[j][m]);
					}
					free(layer->convFilters[j]);
            	}
				free(layer->convFilters);
				free(layer);
				return NULL;
			}
			for(int p=0; p<filter_cols; p++){
				layer->convFilters[i][k][p] = (double *) malloc(sizeof(double) * num_channels);
				if (layer->convFilters[i][k][p] == NULL) {
					for (int j = 0; j<p; j++){
						free(layer->convFilters[i][k][j]);
					}
					free(layer->convFilters[i][k]);
					for (int j =0; j<k; j++){
						free(layer->convFilters[i][j]);
					}
					free(layer->convFilters[i]);
					for (int j = 0; j < i; j++) {
						for(int m =0; m<filter_rows; m++){
							for(int y = 0; y<filter_cols; y++){
								free(layer->convFilters[j][m][y]);
							}
							free(layer->convFilters[j][m]);
						}
						free(layer->convFilters[j]);
					}
					free(layer->convFilters);
					free(layer);
					return NULL;
				}
				for(int w=0; w<num_channels; w++){
					double b = sqrt(6)/sqrt(4 + 4); 
					double a = -1* sqrt(6)/sqrt(4 + 4);
					double random_double = a+(((double)rand() / RAND_MAX) * (b-a));
					layer->convFilters[i][k][p][w] = random_double;
				}
			}
		}
	}
	return layer;
	// double random_double = ((double)rand() / RAND_MAX)*0.6- 0.3;
}
Layer* initFC(int input_size, int output_size){ // For fully connected layer, initialize and randomize weights, set forprop and backprop
	srand(time(NULL));
	Layer* layer= (Layer *) malloc(sizeof(Layer));
	if (layer == NULL){
		fprintf(stderr, "Failed to allocate memory\n");
		return NULL;
	}
	layer->input_size = input_size;
	layer->output_size = output_size;
	layer->bias = (double *) malloc(output_size * sizeof(double));
        if(layer->bias == NULL){
            fprintf(stderr, "Memory allocation failed.\n");
			free(layer);
			return NULL;
        }
	for (int i=0; i<output_size; i++){
        double random_double = ((double)rand() / RAND_MAX)*0.6- 0.3;
		layer->bias[i] = random_double;
	}
	layer->weights = (double **)malloc(input_size * sizeof(double *));
	if (layer->weights == NULL){
		printf("Failed to allocate memory\n");
		free(layer);
		return NULL;
	}
	for (int i = 0; i < input_size; i++) {
        	layer->weights[i] = (double*)malloc(output_size * sizeof(double));
        	if (layer->weights[i] == NULL) {
            		fprintf(stderr, "Memory allocation failed.\n");
            	// You may need to free previously allocated memory here
            		for (int j = 0; j < i; j++) {
                		free(layer->weights[j]);
            		}
            		free(layer->weights);
            		free(layer);
            		return NULL;
        	}
		for (int j = 0; j < output_size; j++) {
			double b = sqrt(6)/sqrt(input_size + output_size);
			double a = -1* sqrt(6)/sqrt(input_size + output_size); // Xavier Weight initialization
			double random_double = a+(((double)rand() / RAND_MAX) * (b-a));
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
static double*** Conv_forprop(Layer *layer, double ***input_data){
	return NULL;
}
static double*** Conv_backprop(Layer *layer, double*** output_error, double learning_rate){
	return NULL;
}
static double* activation_forprop(Layer *layer, double *input_data){ // forward pass for activation layer
	double *result = (double *)malloc(layer->output_size * sizeof(double));
	if (layer == NULL){
		fprintf(stderr, "Failed to forward propagate layer\n");
		return NULL;
	}
	if (result == NULL){
		fprintf(stderr, "Out of memory, failed to propagate layer\n");
		return NULL;
	}
	layer->input = input_data;
	layer->Activation(layer->input, layer->input_size, result);
	layer->output = result;
	return result;
}
static double* activation_backprop(Layer *layer, double *output_error, double learning_rate){ // backward pass for activation layer
	double act[layer->output_size];
	layer->Ddx_activation(layer->input, layer->input_size, act);
	for (int i=0; i<layer->input_size; i++){
		act[i] = act[i] * output_error[i];	
	}
	double *temp=(double*)realloc(output_error, layer->input_size * sizeof(double));
	if (temp == NULL){
		fprintf(stderr, "Out of memory error\n");
		return NULL;
	}
	for (int i=0; i<layer->input_size; i++){
		temp[i] = act[i];
	}
	return temp;
}
static double* FC_forprop(Layer *layer, double *input_data){ // forward prop for Dense layer 
	double *result = (double *)malloc(layer->output_size * sizeof(double));
	layer->input = input_data;
	for (int col = 0; col < layer->output_size; col++) {
        	result[col] = 0;
        	for (int row = 0; row < layer->input_size; row++) {
            		result[col] += layer->weights[row][col] * input_data[row];
        	}
		result[col] += layer->bias[col];
    	}
	layer->output = result;
	return result;

}
static double* FC_backprop(Layer *layer, double *output_error, double learning_rate){ // backprop for Dense layer
	double input_error[layer->input_size];
	for (int i = 0; i < layer->input_size; i++) {
        	input_error[i] = 0;
        	for (int j = 0; j < layer->output_size; j++) {
            		input_error[i] += layer->weights[i][j] * output_error[j];
        	}
    	}
	double weights_error[layer->input_size][layer->output_size];
	for (int i = 0; i < layer->input_size; i++) {
        	for (int j = 0; j < layer->output_size; j++) {
            		weights_error[i][j] = layer->input[i] * output_error[j];
        	}
    	}
	for (int i=0; i< layer->output_size; i++){
		layer->bias[i] += (learning_rate * output_error[i]);

	}
	for (int i=0; i< layer->input_size;i++){
		for (int j=0; j< layer->output_size; j++){
			layer->weights[i][j] += (learning_rate * weights_error[i][j]);
		}
	}
	double* interim = (double*)realloc(output_error, layer->input_size*sizeof(double));
	for(int i=0; i< layer->input_size; i++){
		interim[i] = input_error[i];
	}
	return interim;
}
void destroyNetwork(Network *net){
	Layer *curr = net->head;
	Layer *next;
	while (curr != NULL){
		next = curr->next;
		if (curr->type==0){
			for (int i=0; i<curr->input_size; i++){
				free(curr->weights[i]);
			}
				free(curr->weights);
			free(curr->bias);
		}
		else{
			if(curr->type == 2){
				fprintf(stderr, "!!!!!!");
				// free(the boy); (he aint do nothing)
			}
		}
		free(curr);
		curr = next;
	}
	free(net);
}
void enableVisualizer(Network *net, int flag){
	if(flag == 1){
		net->visualizer = 1;
	}
	else{
		net->visualizer = 0; 
	}
}
void reshapeImages(uint8_t *flatData, double (*reshapedData)[NUM_IMAGES][28][28]) {
    for (int i = 0; i < NUM_IMAGES; ++i) {
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                (*reshapedData)[i][row][col] = flatData[i * IMAGE_SIZE + row * 28 + col];
            }
        }
    }
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
int main(){
	
	FILE *file = fopen("MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", "rb");
	if (!file) {
		perror("Error opening file");
		return 1;
    }
	double testImages[NUM_IMAGES][28][28];
	int magic_number=0;
	int number_of_images=0;
	int n_rows=0;
	int n_cols=0;
	// x = fread(&magic_number, sizeof(uint32_t), 1, file);
	size_t x = fread((char*)&magic_number,sizeof(magic_number),1,file); 
	magic_number= reverseInt(magic_number);
	x = fread((char*)&number_of_images,sizeof(number_of_images),1,file);
	number_of_images= reverseInt(number_of_images);
	x = fread((char*)&n_rows,sizeof(n_rows),1,file);
	n_rows= reverseInt(n_rows);
	x = fread((char*)&n_cols,sizeof(n_cols),1,file);
	n_cols= reverseInt(n_cols);
	for(int i=0;i<number_of_images;++i){
		for(int r=0;r<n_rows;++r){
			for(int c=0;c<n_cols;++c){
				unsigned char temp=0;
				x = fread((char*) &temp,sizeof(temp),1,file);
				testImages[i][r][c] = (double) temp;

			}
		}
	}
	fclose(file);
	// return 0;
	FILE *file2 = fopen("MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte", "rb");
	if (!file2) {
		perror("Error opening file");
		return 1;
    }
	double train_images[NUM_IMAGES_TRAIN][28][28];
	int magic_number_train=0;
	int number_of_images_train=0;
	int n_rows_train=0;
	int n_cols_train=0;
	x = fread((char*)&magic_number_train,sizeof(magic_number_train),1,file2); 
	magic_number_train= reverseInt(magic_number_train);
	x = fread((char*)&number_of_images_train,sizeof(number_of_images_train),1,file2);
	number_of_images_train= reverseInt(number_of_images_train);
	// printf("%d\n",number_of_images_train);
	x = fread((char*)&n_rows_train,sizeof(n_rows_train),1,file2);
	n_rows_train= reverseInt(n_rows_train);
	x = fread((char*)&n_cols_train,sizeof(n_cols_train),1,file2);
	n_cols= reverseInt(n_cols_train);
	for(int i=0;i<number_of_images;++i){
		for(int r=0;r<n_rows;++r){
			for(int c=0;c<n_cols;++c){
				unsigned char temp=0;
				x = fread((char*) &temp,sizeof(temp),1,file2);
				train_images[i][r][c] = (double) temp;
			}
		}
	}
	fclose(file2);
	FILE *file3 = fopen("MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte", "rb");
	if (!file3) {
		perror("Error opening file");
		return 1;
    }
	int magic_number_labels= 0;
    x = fread((char *)&magic_number_labels, sizeof(magic_number_labels),1, file3);
    magic_number_labels = reverseInt(magic_number_labels);
	int number_of_labels=0;
	x = fread((char *)&number_of_labels, sizeof(number_of_labels),1,file3);
	number_of_labels = reverseInt(number_of_labels);
	double test_labels[number_of_labels];
	for(int i=0; i< number_of_labels;i++){
		unsigned char temp = 0;
		x = fread((char *) &temp, sizeof(temp), 1,file3);
		test_labels[i]= (double) temp;
	}
	fclose(file3);
	FILE *file4 = fopen("MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte", "rb");
	if (!file4) {
		perror("Error opening file");
		return 1;
    }
	int magic_number_labels_train= 0;
    x = fread((char *)&magic_number_labels_train, sizeof(magic_number_labels_train),1, file4);
    magic_number_labels_train = reverseInt(magic_number_labels_train);
	int number_of_labels_train=0;
	x = fread((char *)&number_of_labels_train, sizeof(number_of_labels_train),1,file4);
	number_of_labels_train = reverseInt(number_of_labels_train);
	double train_labels[number_of_labels_train];
	for(int i=0; i< number_of_labels_train;i++){
		unsigned char temp = 0;
		x = fread((char *) &temp, sizeof(temp), 1,file4);
		train_labels[i]= (double) temp;
	}
	fclose(file4);
	
	Layer *layerConv = initConv2D(3, 24, 24, 3, 2, 2);
	Network *net = initNetwork(mean_squared_error, mean_squared_prime);
	Layer *layer = initFC(2, 4);
	Layer *layertwo = initActivation(tanh_activation, tanh_p, 3);
	Layer *layerthree = initFC(4, 1);
	Layer *layerf = initActivation(tanh_activation, tanh_p, 1);
	addLayer(net, layer);
	addLayer(net, layertwo);
	addLayer(net, layerthree);
	addLayer(net, layerf);
	// addLayer(net, layerthree);
	// addLayer(net, layerf);
	//this one down below this is supposed to be commented
	/*
	double **input = (double**)malloc(4*sizeof(double*));
	for (int i=0; i<4; i++){
		input[i] = (double *)malloc(2* sizeof(double));
	}
	*/
	//This one down here can be uncommented
	/*
	double input[4][2];
	input[0][0] = 0.0;
	input[0][1] = 0.0;
	input[1][0] = 0.0;
	input[1][1] = 1.0;
	input[2][0] = 1.0;
	input[2][1] = 0.0;
	input[3][0] = 1.0;
	input[3][1] = 1.0;
	*/
	// comment down is ok 
	/*
	double **expected = (double **) malloc(4*sizeof(double*));
	for (int i=0; i<4; i++){
		expected[i] = (double *) malloc(sizeof(double));
	}
	*/
	//uncomment down
	/*
	double expected[4][1];
	expected[0][0] = 1;
	expected[1][0] = 0;
	expected[2][0] = 0;
	expected[3][0] = 1;
	fit(net,4,2,1,input, expected, 1000, 0.1);
	double **out=predict(net, 4,2,input);
	for(int i=0; i<4; i++){
		printf("%f\n", out[i][0]);
	}
	free(out[0]);
	free(out[1]);
	free(out[2]);
	free(out[3]);
	free(out);
	*/
	destroyNetwork(net);
	return 0;
	//end code

	/*double *result;
	double* resultone;
	double *input = (double*)malloc(784*sizeof(double));
	double *output = (double*)malloc(10*sizeof(double));
	double **holder = (double**)malloc(sizeof(double*));
	for (int i=0; i<784; i++){
		input[i] = 2;
	}
	for (int i=0; i<10; i++){
		output[i] = 0.5;
	}
	result = layer->forward_prop(layer, input);
	for (int i=0; i<layer->output_size; i++){
		printf("%f\n", result[i]);
	}
	double learning_rate = 0.1;

	resultone = layer->backward_prop(layer, output, learning_rate);
	for (int i=0; i<784; i++){
		printf("%f\n", resultone[i]);
	}
*/
}
// in mnist, each grayscale pixel value has a max of 255
