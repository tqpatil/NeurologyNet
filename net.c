#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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
double mean_squared_error(double* expected, double* result, int array_length);
void mean_squared_prime(double* expected, double* result, int array_length, double *output);
typedef double (*loss)(double*, double*, int);
typedef void (*loss_prime)(double*, double*, int, double*);
typedef struct Network Network; 
typedef struct Layer Layer;
static double* FC_backprop(Layer *layer, double *output_error, double learning_rate);
static double* FC_forprop(Layer *layer, double *input_data);
static double* activation_backprop(Layer *layer, double *output_error, double learning_rate);
static double* activation_forprop(Layer *layer, double *input_data);
typedef void (*activation)(double*, int, double*);
typedef void (*activation_p)(double*, int, double*);
typedef double* (*forward_prop)(Layer*, double*);
typedef double* (*backward_prop)(Layer*, double*, double);
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
typedef struct Network{
	Layer *head;
	Layer *tail;
	loss loss_function;
	loss_prime loss_function_prime;
	int num_layers;
} Network;
typedef struct Layer{
	double **weights;// must be deallocated
	double *bias;//must be deallocated
	double *input; // must be deallocated
	double *output; // must be deallocated
	int input_size;
	int output_size;
	int type;
	activation Activation;
	activation_p Ddx_activation;
	forward_prop forward_prop;
	backward_prop backward_prop;
	Layer *next; 
	Layer *prev; 
	
} Layer;

Network* initNetwork(loss Loss, loss_prime Loss_prime){
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
void addLayer(Network *net, Layer* layer){
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
double** predict(Network *net, int num_samples, int sample_size, double input_data[num_samples][sample_size]){
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

void fit(Network *net, int num_samples, int sample_size, int sizeOfOutput, double x_train[num_samples][sample_size], double y_train[num_samples][sizeOfOutput], int epochs, double learning_rate){
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
Layer* initActivation(activation a, activation_p ap, int input_size){
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

Layer* initFC(int input_size, int output_size){
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
			double b = sqrt(6/(input_size + output_size));
			double a = -1 * a;
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
static double* activation_forprop(Layer *layer, double *input_data){
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
static double* activation_backprop(Layer *layer, double *output_error, double learning_rate){
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
static double* FC_forprop(Layer *layer, double *input_data){
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
static double* FC_backprop(Layer *layer, double *output_error, double learning_rate){
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
		free(curr);
		curr = next;
	}
	free(net);
}
int main(){
	Network *net = initNetwork(mean_squared_error, mean_squared_prime);
	Layer *layer = initFC(2, 3);
	Layer *layertwo = initActivation(tanh_activation, tanh_p, 3);
	Layer *layerthree = initFC(3, 1);
	Layer *layerf = initActivation(tanh_activation, tanh_p, 1);
	addLayer(net, layer);
	addLayer(net, layertwo);
	addLayer(net, layerthree);
	addLayer(net, layerf);
	/*
	double **input = (double**)malloc(4*sizeof(double*));
	for (int i=0; i<4; i++){
		input[i] = (double *)malloc(2* sizeof(double));
	}
	*/
	double input[4][2];
	input[0][0] = 0.0;
	input[0][1] = 0.0;
	input[1][0] = 0.0;
	input[1][1] = 1.0;
	input[2][0] = 1.0;
	input[2][1] = 0.0;
	input[3][0] = 1.0;
	input[3][1] = 1.0;
	/*
	double **expected = (double **) malloc(4*sizeof(double*));
	for (int i=0; i<4; i++){
		expected[i] = (double *) malloc(sizeof(double));
	}
	*/
	double expected[4][1];
	expected[0][0] = 0.0;
	expected[1][0] = 1.0;
	expected[2][0] = 1.0;
	expected[3][0] = 0.0;
	fit(net,4,2,1,input, expected, 1000, 0.05);
	double **out=predict(net, 4,2,input);
	for(int i=0; i<4; i++){
		printf("%f\n", out[i][0]);
	}
	free(out[0]);
	free(out[1]);
	free(out[2]);
	free(out[3]);
	free(out);
	destroyNetwork(net);
	/*
	printf("these work\n");
	for (int i=0; i < layer->input_size; i++){
		free(layer->weights[i]);
	}
	printf("2\n");
	free(layer->weights);
	for (int i=0; i<layerthree->input_size; i++){
		free(layerthree->weights[i]);
	}
	printf("4\n");
	free(layerthree->weights);
	free(layer->bias);
	free(layerthree->bias);
	free(layer);
	free(layertwo);
	free(layerthree);
	free(net);
	*/

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
	free(resultone);
	free(layer->bias);
	for (int i=0; i < layer->input_size; i++){
		free(layer->weights[i]);
	}
	free(layer->input);
	free(layer->output);
	free(layer->weights);
	free(layer);
*/
}
// in mnist, each grayscale pixel value has a max of 255
