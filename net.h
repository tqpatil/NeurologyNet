#ifndef NET_H
#define NET_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
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
        int num_filters;
        int filter_rows;
        int filter_cols;       
        double ***convFilters; // Need to deallocate
        // double convFilters[num_filters][filter_rows][filter_cols]; // fill with random values 
        Layer *next; 
        Layer *prev; 
        
} Layer;
#endif
