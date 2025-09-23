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

typedef struct Layer Layer;
typedef struct Network Network;

typedef void (*activation)(double*, int, double*);
typedef void (*activation_p)(double*, int, double*);
typedef double* (*forward_prop)(Layer*, double*);
typedef double* (*backward_prop)(Layer*, double*, double);

struct Network {
    Layer *head;
    Layer *tail;
    loss loss_function;
    loss_prime loss_function_prime;
    int visualizer;
    int num_layers;
};

Network* initNetwork(loss Loss, loss_prime Loss_prime);
void addLayer(Network* net, Layer* layer);
double** predict(Network *net, int num_samples, int sample_size, double input_data[num_samples][sample_size]);
void fit(Network *net, int num_samples, int sample_size, int sizeOfOutput, double x_train[num_samples][sample_size], double y_train[num_samples][sizeOfOutput], int epochs, double learning_rate);
void enableVisualizer(Network* net, int flag);

struct Layer {
    double **weights;
    double *bias;
    double *input;
    double *output;
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
    int stride;
    int padding;
    double ****convFilters;
    Layer *next;
    Layer *prev;
};

Layer* initActivation(activation a, activation_p ap, int input_size);
Layer* initConv2D(int num_filters, int filter_rows, int filter_cols, int num_channels, int stride, int padding);
Layer* initFC(int input_size, int output_size);
void destroyNetwork(Network *net);

#endif
