#ifndef NET_H
#define NET_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
float mean_squared_error(float* expected, float* result, int array_length);
void mean_squared_prime(float* expected, float* result, int array_length, float *output);
typedef float (*loss)(float*, float*, int);
typedef void (*loss_prime)(float*, float*, int, float*);
typedef struct Network Network; 
typedef struct Layer Layer;
static float* FC_backprop(Layer *layer, float *output_error, float learning_rate);
static float* FC_forprop(Layer *layer, float *input_data);
static float* activation_backprop(Layer *layer, float *output_error, float learning_rate);
static float* activation_forprop(Layer *layer, float *input_data);
typedef void (*activation)(float*, int, float*);
typedef void (*activation_p)(float*, int, float*);
typedef float* (*forward_prop)(Layer*, float*);
typedef float* (*backward_prop)(Layer*, float*, float);
typedef struct Network{
        Layer *head;
        Layer *tail;
        loss loss_function;
        loss_prime loss_function_prime;
	int visualizer; 
        int num_layers;
} Network;
void enableVisualizer(Network* net, int flag);
typedef struct Layer{
        // Maybe a bool isConvolutional and a corresponding pointer to a conv2d layer with attributes to clean up code
	// Same thing for a flatten layer and pooling layer depending on complexity
	float **weights;// must be deallocated
        float *bias;//must be deallocated
        float *input; // must be deallocated
        float *output; // must be deallocated
        int input_size;
        int output_size;
        int type;
        activation Activation;
        activation_p Ddx_activation;
        forward_prop forward_prop;
        backward_prop backward_prop;
        int num_filters;
        int filter_rows;
        int filter_cols; 
        int channels;      
        float ****convFilters; // Need to deallocate
        // float convFilters[num_filters][filter_rows][filter_cols]; // fill with random values 
        Layer *next; 
        Layer *prev; 
        
} Layer;
#endif
