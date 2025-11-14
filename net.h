#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>

typedef struct ThreadPool ThreadPool;
typedef struct WorkerJob WorkerJob;

struct WorkerJob {
	void (*task)(void*);
	void *arg;
};

struct ThreadPool {
	pthread_t *threads;
	WorkerJob *job_queue;
	int queue_size;
	int queue_head;
	int queue_tail;
	int num_threads;
	int shutdown;
	pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_cond_t complete_cond;
    int tasks_in_progress;
};

double mean_squared_error(double* expected, double* result, int array_length);
void mean_squared_prime(double* expected, double* result, int array_length, double *output);
double cross_entropy_loss(double* expected, double* logits, int array_length);
void cross_entropy_prime(double* expected, double* logits, int array_length, double *output);

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
	ThreadPool *thread_pool;
	int visualizer;
	int num_layers;
};

Network* initNetwork(loss Loss, loss_prime Loss_prime);
void setThreadPoolSize(Network* net, int num_threads);
void addLayer(Network* net, Layer* layer);
double** predict(Network *net, int num_samples, int sample_size, double input_data[num_samples][sample_size]);
void fit(Network *net, int num_samples, int sample_size, int sizeOfOutput, double x_train[num_samples][sample_size], double y_train[num_samples][sizeOfOutput], int epochs, double learning_rate);
void enableVisualizer(Network* net, int flag);
double *forward_sample(Network *net, double *input_flat, int channels, int height, int width);
double evaluate(Network *net, int num_samples, double *x_flat, double *y_flat, int channels, int height, int width, int num_classes);
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
    int input_height;
    int input_width;
    double ****convFilters;
    Layer *next;
    Layer *prev;
};

Layer* initActivation(activation a, activation_p ap, int input_size);
Layer* initConv2D(int num_filters, int filter_rows, int filter_cols, int num_channels, int stride, int padding);
Layer* initFC(int input_size, int output_size);
Layer* initFlatten(int num_filters, int height, int width);
Layer* initMaxPool(int num_channels, int input_height, int input_width, int pool_rows, int pool_cols, int stride);
void destroyNetwork(Network *net);

#endif
