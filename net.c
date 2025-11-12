#include "net.h"

#define NUM_IMAGES 10000
#define IMAGE_SIZE 784
#define NUM_IMAGES_TRAIN 60000
#define THREAD_POOL_DEFAULT 4
#define JOB_QUEUE_SIZE 10000

static void* worker_thread(void *arg);
static void thread_pool_enqueue(ThreadPool *pool, void (*task)(void*), void *arg);
static void thread_pool_wait(ThreadPool *pool);

static ThreadPool *g_pool = NULL;

typedef struct { 
	Layer *layer; 
	double ***active_input; 
	double **out2d; 
	int f; 
	int out_h; 
	int out_w; 
} ConvForTask;
static void conv_forprop_task_fn(void *arg) {
	ConvForTask *t = (ConvForTask*)arg;
	Layer *ly = t->layer;
	int out_h = t->out_h, out_w = t->out_w;
	if (!t || !ly || !t->active_input) {
		// fprintf(stderr, "[conv_forprop_task_fn] NULL arg/layer/active_input\n");
		free(t);
		return;
	}
	if (t->f == 0) {
		// print once per task batch for the first filter to reduce noise
		// fprintf(stderr, "[conv_forprop_task_fn] layer=%p num_filters=%d filter_rows=%d filter_cols=%d channels=%d stride=%d padding=%d active_input=%p\n",
		//         (void*)ly, ly->num_filters, ly->filter_rows, ly->filter_cols, ly->channels, ly->stride, ly->padding, (void*)t->active_input);
		// if (t->active_input[0] && t->active_input[0][0])
		//     fprintf(stderr, "[conv_forprop_task_fn] sample input[0][0]=%f\n", t->active_input[0][0][0]);
	}
	for (int h=0; h<out_h; ++h) {
		for (int w=0; w<out_w; ++w) {
			double sum = 0.0;
			for (int c=0; c<ly->channels; ++c) {
				for (int kh=0; kh<ly->filter_rows; ++kh) {
					for (int kw=0; kw<ly->filter_cols; ++kw) {
						int ih = h * ly->stride + kh;
						int iw = w * ly->stride + kw;
						sum += t->active_input[c][ih][iw] * ly->convFilters[t->f][kh][kw][c];
					}
				}
			}
			t->out2d[h][w] = sum;
		}
	}
	free(t);
}

typedef struct { 
	Layer *layer; 
	int channel; 
	int out_h; 
	int out_w; 
	int in_h; 
	int in_w; 
	double ***output_error; 
	double ***input_error;
} ConvInputTask;
static void conv_input_task_fn(void *arg) {
	ConvInputTask *t = (ConvInputTask*)arg;
	Layer *ly = t->layer;
	int c = t->channel;
	for (int f = 0; f < ly->num_filters; ++f) {
		for (int h = 0; h < t->out_h; ++h) {
			for (int w = 0; w < t->out_w; ++w) {
				double error_val = t->output_error[f][h][w];
				for (int kh = 0; kh < ly->filter_rows; ++kh) {
					for (int kw = 0; kw < ly->filter_cols; ++kw) {
						int ih = h * ly->stride + kh;
						int iw = w * ly->stride + kw;
						t->input_error[c][ih][iw] += error_val * ly->convFilters[f][kh][kw][c];
					}
				}
			}
		}
	}
	free(t);
}

typedef struct { Layer *layer; int f; int out_h; int out_w; int in_h; int in_w; double learning_rate; double ***output_error; } ConvBackTask;
static void conv_back_task_fn(void *arg) {
	ConvBackTask *t = (ConvBackTask*)arg;
	Layer *ly = t->layer;
	int f = t->f;
	for (int h = 0; h < t->out_h; ++h) {
		for (int w = 0; w < t->out_w; ++w) {
			double error_val = t->output_error[f][h][w];
			for (int c = 0; c < ly->channels; ++c) {
				for (int kh = 0; kh < ly->filter_rows; ++kh) {
					for (int kw = 0; kw < ly->filter_cols; ++kw) {
						int ih = h * ly->stride + kh;
						int iw = w * ly->stride + kw;
						// layer->input was set to point to the original 3D input
						// (stored as an opaque pointer). Cast back to double***
						// and index by [channel][row][col].
						double ***input_3d = (double***)ly->input;
						if (ly->padding > 0) {
							int orig_ih = ih - ly->padding;
							int orig_iw = iw - ly->padding;
							if (orig_ih >= 0 && orig_ih < t->in_h && orig_iw >= 0 && orig_iw < t->in_w) {
								ly->convFilters[f][kh][kw][c] += t->learning_rate * error_val * input_3d[c][orig_ih][orig_iw];
							}
						} else {
							if (ih >= 0 && ih < t->in_h && iw >= 0 && iw < t->in_w)
								ly->convFilters[f][kh][kw][c] += t->learning_rate * error_val * input_3d[c][ih][iw];
						}
					}
				}
			}
		}
	}
	free(t);
}

typedef struct { 
	Layer *layer; 
	double *input; 
	double *result; 
	int start; 
	int end; 
} FCForTask;
static void fc_forprop_task_fn(void *arg) {
	FCForTask *t = (FCForTask*)arg;
	for (int col =t->start; col < t->end; ++col) {
		double acc = 0.0;
		for (int row = 0; row < t->layer->input_size; ++row)
			acc += t->layer->weights[row][col]*t->input[row];
		t->result[col] = acc + t->layer->bias[col];
	}
	free(t);
}

typedef struct {
	Layer *layer; 
	int start; 
	int end; 
	double *output_error; 
	double *dest; 
} InputErrTask;
static void input_err_task_fn(void *arg) {
	InputErrTask *t = (InputErrTask*)arg;
	for (int i = t->start; i < t->end; ++i) {
		double acc = 0.0;
		for (int j = 0; j < t->layer->output_size; ++j) {
			acc += t->layer->weights[i][j] * t->output_error[j];
		}
		t->dest[i] = acc;
	}
	free(t);
}

typedef struct { Layer *layer; int start_col; int end_col; double *input_vec; double *output_err; double lr; } WeightUpdTask;
static void weight_upd_task_fn(void *arg) {
	WeightUpdTask *t = (WeightUpdTask*)arg;
	for (int j = t->start_col; j < t->end_col; ++j) {
		t->layer->bias[j] += t->lr * t->output_err[j];
		for (int i = 0; i < t->layer->input_size; ++i) {
			t->layer->weights[i][j] += t->lr * t->input_vec[i] * t->output_err[j];
		}
	}
	free(t);
}

typedef struct { Layer *layer; double *in; double *out; int start; int len; } ActTask;
static void act_task_fn(void *arg) {
	ActTask *t = (ActTask*)arg;
	t->layer->Activation(t->in + t->start, t->len, t->out + t->start);
	free(t);
}

typedef struct { Layer *layer; double *in; double *out_err; int start; int len; } ActBackTask;
static void act_back_task_fn(void *arg) {
	ActBackTask *t = (ActBackTask*)arg;
	double *buf = malloc(t->len * sizeof(double));
	t->layer->Ddx_activation(t->in + t->start, t->len, buf);
	for (int i = 0; i < t->len; ++i) {
		t->out_err[t->start + i] = buf[i] * t->out_err[t->start + i];
	}
	free(buf);
	free(t);
}

typedef struct { double *input_data; double *output; int *mask; int channel; int in_h; int in_w; int out_h; int out_w; int pool_h; int pool_w; int stride; } MaxPoolTask;
static void maxpool_task_fn(void *arg) {
	MaxPoolTask *t = (MaxPoolTask*)arg;
	int c = t->channel;
	for (int oh = 0; oh < t->out_h; ++oh) {
		for (int ow = 0; ow < t->out_w; ++ow) {
			double best = -INFINITY;
			int best_idx = 0;
			for (int ph = 0; ph < t->pool_h; ++ph) {
				for (int pw = 0; pw < t->pool_w; ++pw) {
					int ih = oh * t->stride + ph;
					int iw = ow * t->stride + pw;
					int idx = c * (t->in_h * t->in_w) + ih * t->in_w + iw;
					double val = t->input_data[idx];
					if (val > best) { 
						best = val; best_idx = idx; 
					}
				}
			}
			int out_index = c * (t->out_h * t->out_w) + oh * t->out_w + ow;
			t->output[out_index] = best;
			t->mask[out_index] = best_idx;
		}
	}
	free(t);
}

typedef struct { int *mask; double *out_err; double *dest; int start; int end; } MaxPoolBackTask;
static void maxpool_back_task_fn(void *arg) {
	MaxPoolBackTask *t = (MaxPoolBackTask*)arg;
	for (int i = t->start; i < t->end; ++i) {
		int idx = t->mask[i];
		t->dest[idx] += t->out_err[i];
	}
	free(t);
}


static double*** Conv_forprop(Layer *layer, double ***input_data);
static double*** Conv_backprop(Layer *layer, double*** output_error, double learning_rate);
static double* FC_backprop(Layer *layer, double *output_error, double learning_rate);
static double* FC_forprop(Layer *layer, double *input_data);
static double* activation_backprop(Layer *layer, double *output_error, double learning_rate);
static double* activation_forprop(Layer *layer, double *input_data);
static double* flatten_forprop(Layer *layer, double *input_data);
static double* flatten_backprop(Layer *layer, double *output_error, double learning_rate);
static double* maxpool_forprop(Layer *layer, double *input_data);
static double* maxpool_backprop(Layer *layer, double *output_error, double learning_rate);
static double* Conv_forward_wrapper(Layer *layer, double *input_data);
static double* Conv_backward_wrapper(Layer *layer, double *output_error, double learning_rate);

double mean_squared_error(double* expected, double* result, int array_length) {
	double error = 0.0;
	for (int i = 0; i < array_length; ++i) {
		double d = expected[i] - result[i];
		error += d * d;
	}
	return error / array_length;
}

void mean_squared_prime(double* expected, double* result, int array_length, double *output) {
	for (int i = 0; i < array_length; ++i) {
		output[i] = (2.0 * (expected[i] - result[i])) / array_length;
	}
}

typedef struct { double *logits; double *out_buf; int start; int len; double *partial_sum; } SoftmaxExpTask;
static void softmax_exp_task_fn(void *arg) {
	SoftmaxExpTask *t = (SoftmaxExpTask*)arg;
	double local_sum =0.0;
	for (int i = 0; i < t->len; ++i) {
		double v = t->logits[t->start + i];
		double e = exp(v);
		t->out_buf[t->start + i] = e;
		local_sum += e;
	}
	if (t->partial_sum) {
		*t->partial_sum = local_sum;
	}
	free(t);
}

typedef struct { double *logits_or_probs; double *expected; int start; int len; int total_len; double *out_grad; } SoftmaxGradTask;
static void softmax_grad_task_fn(void *arg) {
	SoftmaxGradTask *t = (SoftmaxGradTask*)arg;
	for (int i = 0; i < t->len; ++i) {
		int idx = t->start + i;
		t->out_grad[idx] = (t->logits_or_probs[idx] - t->expected[idx]) / (double)t->total_len;
	}
	free(t);
}

double cross_entropy_loss(double* expected, double* logits, int array_length) {
	if (!logits || !expected || array_length <= 0) return 0.0;
	double maxv = logits[0];
	for (int i = 1; i < array_length; ++i)
		if (logits[i] > maxv) maxv = logits[i];

	double *exp_buf = malloc(array_length * sizeof(double));
	double *logits_copy = malloc(array_length * sizeof(double));
	if (!exp_buf || !logits_copy) {
		free(exp_buf);
		free(logits_copy);
		return 0.0;
	}

	int threads = g_pool ? g_pool->num_threads : 1;
	if (threads <= 0) {
		threads = 1;
	}
	int chunk = (array_length + threads - 1) / threads;
	double *partials=NULL;
	if (g_pool) {
		partials = malloc(threads * sizeof(double));
	}

	for (int i = 0; i < array_length; ++i) {
		logits_copy[i] = logits[i] - maxv;
	}

	if (!g_pool) {
		double sum = 0.0;
		for (int i = 0; i < array_length; ++i) {
			exp_buf[i] = exp(logits_copy[i]);
			sum += exp_buf[i];
		}
		double loss = 0.0;
		for (int i = 0; i < array_length; ++i) {
			double p = exp_buf[i] / sum;
			if (expected[i] > 0.0) {
				loss -= expected[i] * log(p + 1e-15);
			}
		}
		free(exp_buf);
		free(logits_copy);
		return loss / array_length;
	}

	for (int t = 0; t < threads; ++t) {
		int start = t * chunk;
		if (start >= array_length) {
			break;
		} 
		int len = chunk;
		if (start + len > array_length) {
			len = array_length - start;
		}
		SoftmaxExpTask *task = malloc(sizeof(SoftmaxExpTask));
		task->logits = logits_copy; task->out_buf = exp_buf; task->start = start; task->len = len; task->partial_sum = &partials[t];
		thread_pool_enqueue(g_pool, softmax_exp_task_fn, task);
	}
	thread_pool_wait(g_pool);

	double sum = 0.0;
	for (int t = 0; t < threads; ++t) {
		sum += partials[t];
	}

	double loss = 0.0;
	for (int i = 0; i < array_length; ++i) {
		double p = exp_buf[i] / sum;
		if (expected[i] > 0.0) {
			loss -= expected[i] * log(p+1e-15);
		}
	}

	free(partials);
	free(exp_buf);
	free(logits_copy);
	return loss / array_length;
}

void cross_entropy_prime(double* expected, double* logits, int array_length, double *output) {
	if (!expected || !logits || !output || array_length <= 0){
		return;
	}
	double *logits_copy = malloc(array_length * sizeof(double));
	double *exp_buf = malloc(array_length * sizeof(double));
	double *probs = malloc(array_length * sizeof(double));
	if (!logits_copy || !exp_buf || !probs) {
		free(logits_copy);
		free(exp_buf);
		free(probs);
		return;
	}

	double maxv = logits[0];
	for (int i = 1; i < array_length; ++i) {
		if (logits[i] > maxv) {
			maxv = logits[i];
		}
	}
	for (int i = 0; i < array_length; ++i) {
		logits_copy[i] = logits[i] - maxv;
	}

	for (int i = 0; i < array_length; ++i) {
		exp_buf[i] = exp(logits_copy[i]);
	}

	double sum = 0.0;
	for (int i = 0; i < array_length; ++i) {
		sum += exp_buf[i];
	}

	for (int i = 0; i < array_length; ++i) {
		probs[i] = exp_buf[i] / sum;
	}

	for (int i = 0; i < array_length; ++i) {
		output[i] = (expected[i] - probs[i]) / (double)array_length;
	}

	free(logits_copy);
	free(exp_buf);
	free(probs);
}

void relu_activation(double *input, int input_size, double *result) {
	for (int i = 0; i < input_size; ++i) {
		result[i] = input[i] > 0.0 ? input[i] : 0.0;
	}
}

void relu_p(double *input, int input_size, double *result) {
	for (int i = 0; i < input_size; ++i) {
		result[i] = input[i] > 0.0 ? 1.0 : 0.0;
	}
}

void tanh_activation(double *input, int input_size, double *result) {
	for (int i = 0; i < input_size; ++i) {
		result[i] = tanh(input[i]);
	}
}

void tanh_p(double *input, int input_size, double *result) {
	for (int i = 0; i < input_size; ++i) {
		double t = tanh(input[i]);
		result[i] = 1.0 - (t * t);
	}
}

static void* worker_thread(void *arg) {
	ThreadPool *pool = (ThreadPool *)arg;
	
	while (1) {
		pthread_mutex_lock(&pool->lock);
		while (pool->queue_head == pool->queue_tail && !pool->shutdown){
			pthread_cond_wait(&pool->notify, &pool->lock);
		}
		if (pool->shutdown) {
			pthread_mutex_unlock(&pool->lock);
			break;
		}
		WorkerJob job = pool->job_queue[pool->queue_head];
		pool->queue_head = (pool->queue_head + 1)%pool->queue_size;
		
		pthread_mutex_unlock(&pool->lock);
		job.task(job.arg);
		pthread_mutex_lock(&pool->lock);
		pool->tasks_in_progress--;
		if (pool->tasks_in_progress == 0){
			pthread_cond_signal(&pool->complete_cond);
		}
		pthread_mutex_unlock(&pool->lock);
	}
	
	return NULL;
}

static void thread_pool_enqueue(ThreadPool *pool, void (*task)(void*), void *arg) {
	if (!pool || !task) {
		return;
	}
	
	pthread_mutex_lock(&pool->lock);
	
	int next_tail = (pool->queue_tail + 1) %pool->queue_size;
	if (next_tail == pool->queue_head) {
		pthread_mutex_unlock(&pool->lock);
		return;
	}
	
	pool->job_queue[pool->queue_tail].task = task;
	pool->job_queue[pool->queue_tail].arg = arg;
	pool->queue_tail = next_tail;
	pool->tasks_in_progress++;
	pthread_cond_signal(&pool->notify);
	pthread_mutex_unlock(&pool->lock);
}

static void thread_pool_wait(ThreadPool *pool) {
	if (!pool) {
		return;
	}
	pthread_mutex_lock(&pool->lock);
	while (pool->tasks_in_progress > 0) {
		pthread_cond_wait(&pool->complete_cond, &pool->lock);
	}
	pthread_mutex_unlock(&pool->lock);
}

static ThreadPool* thread_pool_create(int num_threads) {
	if (num_threads <= 0){
		return NULL;
	}

	ThreadPool *pool = malloc(sizeof(ThreadPool));
	if (!pool){
		return NULL;
	}

	pool->num_threads = num_threads;
	pool->queue_size = JOB_QUEUE_SIZE;
	pool->queue_head = 0;
	pool->queue_tail = 0;
	pool->shutdown = 0;

	pool->job_queue = malloc(pool->queue_size * sizeof(WorkerJob));
	if (!pool->job_queue) {
		free(pool);
		return NULL;
	}

	pool->threads = malloc(num_threads * sizeof(pthread_t));
	if (!pool->threads) {
		free(pool->job_queue);
		free(pool);
		return NULL;
	}

	pthread_mutex_init(&pool->lock, NULL);
	pthread_cond_init(&pool->notify, NULL);
	pthread_cond_init(&pool->complete_cond, NULL);
	pool->tasks_in_progress = 0;

	for (int i = 0; i < num_threads; ++i) {
		pthread_create(&pool->threads[i], NULL, worker_thread, pool);
	}
	return pool;
}

static void thread_pool_destroy(ThreadPool *pool) {
	if (!pool) {
		return;
	}
	pthread_mutex_lock(&pool->lock);
	pool->shutdown = 1;
	pthread_cond_broadcast(&pool->notify);
	pthread_mutex_unlock(&pool->lock);

	for (int i = 0; i < pool->num_threads; ++i){
		pthread_join(pool->threads[i], NULL);
	}
	pthread_mutex_destroy(&pool->lock);
	pthread_cond_destroy(&pool->notify);
	pthread_cond_destroy(&pool->complete_cond);
	free(pool->threads);
	free(pool->job_queue);
	free(pool);
}

Network* initNetwork(loss Loss, loss_prime Loss_prime) {
	Network *net = malloc(sizeof(*net));
	if (!net) {
		return NULL;
	}
	net->loss_function = Loss;
	net->loss_function_prime = Loss_prime;
	net->head = NULL;
	net->tail = NULL;
	net->num_layers = 0;
	net->visualizer = 0;
	net->thread_pool = thread_pool_create(THREAD_POOL_DEFAULT);
	g_pool = net->thread_pool;
	return net;
}

void addLayer(Network *net, Layer* layer) {
	if (!net || !layer) {
		return;
	}
	if (!net->head) {
		net->head = net->tail = layer;
		layer->prev = layer->next = NULL;
	} 
	else {
		net->tail->next = layer;
		layer->prev = net->tail;
		layer->next = NULL;
		net->tail = layer;
	}
	net->num_layers++;
}

void setThreadPoolSize(Network* net, int num_threads) {
	if (!net || num_threads <= 0) {
		return;
	}
	
	if (net->thread_pool){
		thread_pool_destroy(net->thread_pool);
	}
	net->thread_pool = thread_pool_create(num_threads);
	g_pool = net->thread_pool;
}

double** predict(Network *net, int num_samples, int sample_size, double input_data[num_samples][sample_size]) {
	if (!net || !net->head || !net->tail) {
		return NULL;
	}

	double **result = malloc(num_samples * sizeof(double*));
	for (int i = 0; i < num_samples; ++i) {
		result[i] = malloc(net->tail->output_size * sizeof(double));
	}

	for (int i = 0; i < num_samples; ++i) {
		int in_size = net->head->input_size;
		double *input = malloc(in_size * sizeof(double));
		for (int j = 0; j < in_size; ++j) {
			input[j] = input_data[i][j];
		}

		Layer *curr = net->head;
		double *output = NULL;
		while (curr) {
			output = curr->forward_prop(curr, input);
			input = output;
			curr = curr->next;
		}

		for (int m = 0; m < net->tail->output_size; ++m) {
			result[i][m] = output[m];
		}

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

void fit(Network *net, int num_samples, int sample_size, int sizeOfOutput, double x_train[num_samples][sample_size], double y_train[num_samples][sizeOfOutput], int epochs, double learning_rate) {
	if (!net || !net->head || !net->tail) {
		return;
	}
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
			// Make sure grad points to the latest buffer returned by backprop
			grad = e;

			// Free allocated buffers from forward/backward pass
			// Do NOT free layer->input as it points to data used during forward pass
			// Only free layer->output which are intermediate allocations
			curr = net->head;
			while (curr) {
				// Skip freeing input - it's owned by the caller or previous layer
				// Free output from all layers
				if (curr->output) {
					free(curr->output);
					curr->output = NULL;
				}
				curr->input = NULL;
				curr = curr->next;
			}
		}

		epoch_error /= num_samples;
		printf("epoch %d of %d, error=%f\n", ep + 1, epochs, epoch_error);
	}

	free(grad);
}

Layer* initActivation(activation a, activation_p ap, int input_size) {
	Layer *layer = malloc(sizeof(*layer));
	if (!layer) {
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
	return layer;
}

Layer* initConv2D(int num_filters, int filter_rows, int filter_cols, int num_channels, int stride, int padding) {
	Layer *layer = malloc(sizeof(*layer));
	if (!layer) {
		return NULL;
	}
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
		if (!layer->convFilters[i]) {
			goto conv_alloc_fail;
		}

		for (int r = 0; r < filter_rows; ++r) {
			layer->convFilters[i][r] = malloc(sizeof(double*) * filter_cols);
			if (!layer->convFilters[i][r]) {
				goto conv_alloc_fail;
			}

			for (int c = 0; c < filter_cols; ++c) {
				layer->convFilters[i][r][c] = malloc(sizeof(double) * num_channels);
				if (!layer->convFilters[i][r][c]) {
					goto conv_alloc_fail;
				}

				for (int ch = 0; ch < num_channels; ++ch) {
					double b = sqrt(6.0) / sqrt(4.0 + 4.0);
					double a = -b;
					double random_double = a + (((double)rand() / RAND_MAX) * (b - a));
					layer->convFilters[i][r][c][ch] = random_double;
				}
			}
		}
	}

	// Use small wrapper functions that match the forward_prop/backward_prop signatures (double*)
	layer->forward_prop = Conv_forward_wrapper;
	layer->backward_prop = Conv_backward_wrapper;

	return layer;

conv_alloc_fail: // goto usage (lol sorry)
	for (int ii = 0; ii < num_filters; ++ii) {
		if (!layer->convFilters[ii]) {
			continue;
		}
		for (int rr = 0; rr < filter_rows; ++rr) {
			if (!layer->convFilters[ii][rr]) {
				continue;
			}
			for (int cc = 0; cc < filter_cols; ++cc) {
				free(layer->convFilters[ii][rr][cc]);
			}
			free(layer->convFilters[ii][rr]);
		}
		free(layer->convFilters[ii]);
	}
	free(layer->convFilters);
	free(layer);
	return NULL;
}

Layer* initFC(int input_size, int output_size) {
	Layer* layer = malloc(sizeof(*layer));
	if (!layer) {
		return NULL;
	}
	layer->input_size = input_size;
	layer->output_size = output_size;

	layer->bias = malloc(output_size * sizeof(double));
	if (!layer->bias) {
		free(layer);
		return NULL;
	}

	// Initialize bias to small positive value to help with ReLU
	for (int i = 0; i < output_size; ++i) {
		layer->bias[i] = 0.1;
	}
	layer->weights = malloc(input_size * sizeof(double*));
	if (!layer->weights) {
		free(layer->bias);
		free(layer);
		return NULL;
	}

	for (int i = 0; i < input_size; ++i) {
		layer->weights[i] = malloc(output_size * sizeof(double));
		if (!layer->weights[i]) {
			for (int j=0; j<i; ++j) {
				free(layer->weights[j]);
			}
			free(layer->weights);
			free(layer->bias);
			free(layer);
			return NULL;
		}

		for (int j = 0; j < output_size; ++j) {
			double b = sqrt(6.0) / sqrt(input_size + output_size);
			double a = -b;
			double random_double = a + (((double)rand() /RAND_MAX) * (b - a));
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

Layer* initFlatten(int num_filters, int height, int width) {
	Layer *layer = malloc(sizeof(*layer));
	if (!layer) {
		return NULL;
	}
	layer->input_size = num_filters * height * width;
	layer->output_size = num_filters * height * width;
	layer->num_filters = num_filters;
	layer->filter_rows = height;
	layer->filter_cols = width;
	layer->forward_prop = flatten_forprop;
	layer->backward_prop = flatten_backprop;
	layer->input = NULL;
	layer->output = NULL;
	layer->type = 3;
	layer->weights = NULL;
	layer->bias = NULL;

	return layer;
}

Layer* initMaxPool(int num_channels, int input_height, int input_width, int pool_rows, int pool_cols, int stride) {
	Layer *layer = malloc(sizeof(*layer));
	if (!layer){
		return NULL;
	} 
	layer->channels = num_channels;
	layer->input_size = num_channels * input_height * input_width;
	layer->filter_rows = pool_rows;
	layer->filter_cols = pool_cols;
	layer->input_height = input_height;
	layer->input_width = input_width;
	int out_h = (input_height-pool_rows) / stride + 1;
	int out_w = (input_width-pool_cols) / stride + 1;
	layer->output_size = num_channels * out_h * out_w;
	layer->stride = stride;
	layer->padding = 0;
	layer->forward_prop = maxpool_forprop;
	layer->backward_prop = maxpool_backprop;
	layer->input = NULL;
	layer->output = NULL;
	layer->type = 4;
	return layer;
}

static double*** Conv_forprop(Layer *layer, double ***input_data) {
	int input_height = 28;
	int input_width = 28;
	int output_height = (input_height + 2 * layer->padding - layer->filter_rows) / layer->stride + 1;
	int output_width = (input_width + 2 * layer->padding - layer->filter_cols) / layer->stride + 1;

	if (!layer) {
		// fprintf(stderr, "[Conv_forprop] ERROR: NULL layer\n");
		return NULL;
	}
	if (!input_data) {
		// fprintf(stderr, "[Conv_forprop] ERROR: NULL input_data for layer %p\n", (void*)layer);
		return NULL;
	}
	// fprintf(stderr, "[Conv_forprop] layer=%p num_filters=%d filt=%dx%d channels=%d stride=%d padding=%d input_data=%p\n",
	//         (void*)layer, layer->num_filters, layer->filter_rows, layer->filter_cols, layer->channels, layer->stride, layer->padding, (void*)input_data);
	if (layer->channels > 0) {
		if (!input_data[0]) {
			// fprintf(stderr, "[Conv_forprop] ERROR: input_data[0] is NULL\n");
			return NULL;
		}
		if (!input_data[0][0]) {
			// fprintf(stderr, "[Conv_forprop] ERROR: input_data[0][0] is NULL\n");
			return NULL;
		}
		// fprintf(stderr, "[Conv_forprop] sample input[0][0]=%f\n", input_data[0][0][0]);
	}

	double ***output = malloc(layer->num_filters * sizeof(double**));
	for (int f = 0; f < layer->num_filters; ++f) {
		output[f] = malloc(output_height * sizeof(double*));
		for (int h = 0; h < output_height; ++h) {
			output[f][h] = malloc(output_width * sizeof(double));
		}
	}

	double ***padded_input = NULL;
	if (layer->padding > 0) {
		int padded_height = input_height + 2 * layer->padding;
		int padded_width = input_width + 2 * layer->padding;
		padded_input = malloc(layer->channels * sizeof(double**));
		for (int c = 0; c < layer->channels; ++c) {
			padded_input[c] = malloc(padded_height * sizeof(double*));
			for (int h = 0; h < padded_height; ++h) {
				padded_input[c][h] = malloc(padded_width * sizeof(double));
				for (int w = 0; w < padded_width; ++w) {
					padded_input[c][h][w] = 0.0;
				}
			}
			for (int h = 0; h < input_height; ++h) {
				for (int w = 0; w < input_width; ++w){
					padded_input[c][h + layer->padding][w + layer->padding] = input_data[c][h][w];
			
				}
			}
		}
	}

	    double ***active_input = padded_input ? padded_input : input_data;
    // Remember the original (un-padded) input so backprop can access
    // the pre-packed 3D input. Store it in layer->input (opaque pointer).
    layer->input = (double*)input_data;
	if (!g_pool) {
		for (int f = 0; f < layer->num_filters; ++f) {
			for (int h = 0; h < output_height; ++h) {
				for (int w = 0; w < output_width; ++w) {
					double sum = 0.0;
					for (int c = 0; c < layer->channels; ++c) {
						for (int kh = 0; kh < layer->filter_rows; ++kh) {
							for (int kw = 0; kw < layer->filter_cols; ++kw) {
								int ih = h * layer->stride + kh;
								int iw = w * layer->stride + kw;
								sum += active_input[c][ih][iw] * layer->convFilters[f][kh][kw][c];
							}
						}
					}
					output[f][h][w] = sum;
				}
			}
		}
	} else {
		for (int f = 0; f < layer->num_filters; ++f) {
			ConvForTask *t = malloc(sizeof(ConvForTask));
			t->layer = layer; 
			t->active_input = active_input; 
			t->out2d = output[f]; 
			t->f = f; 
			t->out_h = output_height; 
			t->out_w = output_width;
			thread_pool_enqueue(g_pool, conv_forprop_task_fn, t);
		}
		thread_pool_wait(g_pool);
	}

	if (padded_input) {
		int padded_height = input_height + 2 * layer->padding;
		int padded_width = input_width + 2 * layer->padding;
		for (int c=0; c<layer->channels; ++c) {
			for (int h = 0; h < padded_height; ++h) {
				free(padded_input[c][h]);
			}
			free(padded_input[c]);
		}
		free(padded_input);
	}

	layer->output = (double*)output;
	return output;
}

static double*** Conv_backprop(Layer *layer, double***output_error, double learning_rate) {
	int input_height = 28;
	int input_width = 28;
	int output_height = (input_height +2 * layer->padding - layer->filter_rows) / layer->stride + 1;
	int output_width = (input_width + 2 * layer->padding - layer->filter_cols) / layer->stride + 1;

	double ***input_error = malloc(layer->channels * sizeof(double**));
	int padded_height = input_height + 2 * layer->padding;
	int padded_width = input_width + 2 * layer->padding;
	for (int c = 0; c < layer->channels; ++c) {
		input_error[c] = malloc(padded_height * sizeof(double*));
		for (int h = 0; h < padded_height; ++h) {
			input_error[c][h] = malloc(padded_width * sizeof(double));
			for (int w = 0; w<padded_width; ++w) {
				input_error[c][h][w] = 0.0;
			}
		}
	}

	if (!g_pool) {
		for (int f = 0; f < layer->num_filters; ++f) {
			for (int h = 0; h < output_height; ++h) {
				for (int w = 0; w < output_width; ++w) {
					double error_val = output_error[f][h][w];
					for (int c = 0; c < layer->channels; ++c) {
						for (int kh = 0; kh < layer->filter_rows; ++kh) {
							for (int kw = 0; kw < layer->filter_cols; ++kw) {
								int ih = h * layer->stride + kh;
								int iw = w * layer->stride + kw;
								input_error[c][ih][iw] += error_val * layer->convFilters[f][kh][kw][c];
							}
						}
					}
				}
			}
		}
	} else {
		int chs = layer->channels;
		for (int c = 0; c < chs; ++c) {
			ConvInputTask *t = malloc(sizeof(ConvInputTask));
			t->layer = layer; t->channel = c; t->out_h = output_height; t->out_w = output_width; t->in_h = input_height; t->in_w = input_width; t->output_error = output_error; t->input_error = input_error;
			thread_pool_enqueue(g_pool, conv_input_task_fn, t);
		}
		thread_pool_wait(g_pool);
	}

	if (!g_pool) {
		for (int f = 0; f < layer->num_filters; ++f) { // absolutely horrifying nested loops
			for (int h = 0; h < output_height; ++h) {
				for (int w = 0; w < output_width; ++w) {
					double error_val = output_error[f][h][w];
					for (int c = 0; c < layer->channels; ++c) {
						for (int kh = 0; kh < layer->filter_rows; ++kh) {
							for (int kw = 0; kw < layer->filter_cols; ++kw) {
								int ih = h * layer->stride+kh;
								int iw = w * layer->stride+kw;
								// Use the stored 3D input pointer for correct indexing
								double ***input_3d = (double***)layer->input;
								if (layer->padding>0) {
									int orig_ih = ih - layer->padding;
									int orig_iw = iw - layer->padding;
									if (orig_ih >= 0 && orig_ih < input_height && orig_iw >= 0 && orig_iw < input_width) {
										layer->convFilters[f][kh][kw][c] += learning_rate * error_val * input_3d[c][orig_ih][orig_iw];
									}
								} else {
									if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width)
										layer->convFilters[f][kh][kw][c] += learning_rate * error_val * input_3d[c][ih][iw];
								}
							}
						}
					}
				}
			}
		}
	} else {
		for (int f = 0; f < layer->num_filters; ++f) {
			ConvBackTask *t = malloc(sizeof(ConvBackTask));
			t->layer = layer; 
			t->f = f; 
			t->out_h = output_height; 
			t->out_w = output_width; 
			t->in_h = input_height; 
			t->in_w = input_width; 
			t->learning_rate = learning_rate; 
			t->output_error = output_error;
			thread_pool_enqueue(g_pool, conv_back_task_fn, t);
		}
		thread_pool_wait(g_pool);
	}

	double ***output = malloc(layer->channels * sizeof(double**));
	for (int c = 0; c < layer->channels; ++c) {
		output[c] = malloc(input_height * sizeof(double*));
		for (int h = 0; h < input_height; ++h) {
			output[c][h] = malloc(input_width * sizeof(double));
			for (int w = 0; w < input_width; ++w) {
				output[c][h][w] = input_error[c][h + layer->padding][w + layer->padding];
			}
		}
	}

	for (int c = 0; c < layer->channels; ++c) {
		for (int h = 0; h < padded_height; ++h) {
			free(input_error[c][h]);
		}
		free(input_error[c]);
	}
	free(input_error);

	for (int f = 0; f < layer->num_filters; ++f) {
		for (int h = 0; h < output_height; ++h) {
			free(output_error[f][h]);
		}
		free(output_error[f]);
	}
	free(output_error);

	return output;
}

static double* activation_forprop(Layer *layer, double *input_data) {
	if (!layer) {
		return NULL;
	}
	double *result = malloc(layer->output_size * sizeof(double));
	if (!result) {
		return NULL;
	}
	layer->input = input_data;
	if (!g_pool) {
		layer->Activation(layer->input, layer->input_size, result);
		layer->output = result;
		return result;
	}
	int chunk = (layer->input_size + g_pool->num_threads - 1) / g_pool->num_threads;
	if (chunk < 1) chunk = 1;
	for (int s = 0; s < layer->input_size; s += chunk) {
		int len = chunk;
		if (s + len > layer->input_size){
			len = layer->input_size - s;
		}
		ActTask *t = malloc(sizeof(ActTask));
		t->layer = layer; t->in = input_data; t->out = result; t->start = s; t->len = len;
		thread_pool_enqueue(g_pool, act_task_fn, t);
	}
	thread_pool_wait(g_pool);
	layer->output = result;
	return result;
}

static double* activation_backprop(Layer *layer, double *output_error, double learning_rate) {
	(void)learning_rate;
	double *result = malloc(layer->input_size * sizeof(double));
	if (!result) {
		return NULL;
	} 
	if (!g_pool) {
		double act[layer->output_size];
		layer->Ddx_activation(layer->input, layer->input_size, act);
		for (int i=0; i<layer->input_size; ++i)
			result[i] = act[i] * output_error[i];
		free(output_error);
		return result;
	}
	int chunk = (layer->input_size + g_pool->num_threads - 1)/g_pool->num_threads;
	if (chunk < 1) {
		chunk = 1;
	}
	for (int s = 0; s < layer->input_size; s += chunk) {
		int len = chunk;
		if (s + len > layer->input_size){
			len = layer->input_size - s;
		}
	ActBackTask *t = malloc(sizeof(ActBackTask));
	t->layer=layer;
	t->in = layer->input;
	t->out_err = output_error; 
	t->start = s;
	t->len = len;
	thread_pool_enqueue(g_pool, act_back_task_fn, t);
	}
	thread_pool_wait(g_pool);
	// CRITICAL FIX: The tasks above update output_error in place. Copy it to result.
	for (int i = 0; i < layer->input_size; ++i) {
		result[i] = output_error[i];
	} 
	free(output_error);
	return result;
}

static double* FC_forprop(Layer *layer, double *input_data) {
	double *result = malloc(layer->output_size*sizeof(double));
	if (!result){
		return NULL;
	}
	layer->input = input_data;
	if (!g_pool) {
		for (int col = 0; col < layer->output_size; ++col) {
			double acc = 0.0;
			for (int row = 0; row < layer->input_size; ++row) {
				acc += layer->weights[row][col] * input_data[row];
			}
			result[col] = acc + layer->bias[col];
		}
		layer->output = result;
		return result;
	}


	int chunk = (layer->output_size + g_pool->num_threads - 1)/g_pool->num_threads;
	if (chunk < 1){
		chunk = 1;
	}
	for (int start = 0; start < layer->output_size; start += chunk) {
		int end = start + chunk;
		if (end > layer->output_size){
			end = layer->output_size;
		}
		FCForTask *t = malloc(sizeof(FCForTask));
		t->layer = layer; t->input = input_data; t->result = result; t->start = start; t->end = end;
		thread_pool_enqueue(g_pool, fc_forprop_task_fn, t);
	}
	thread_pool_wait(g_pool);
	layer->output = result;
	return result;
}

static double* FC_backprop(Layer *layer, double *output_error, double learning_rate) {
	double *input_error = malloc(layer->input_size * sizeof(double));
	if (!input_error) return NULL;

	if (!g_pool) {
		for (int i = 0; i < layer->input_size; ++i) {
			double acc = 0.0;
			for (int j = 0; j < layer->output_size; ++j) {
				acc += layer->weights[i][j] * output_error[j];
			}
			input_error[i] = acc;
		}

		for (int i = 0; i < layer->output_size; ++i) {
			layer->bias[i] += learning_rate * output_error[i];
		}

		for (int j = 0; j < layer->output_size; ++j) {
			for (int i = 0; i < layer->input_size; ++i)
				layer->weights[i][j] += learning_rate * layer->input[i] * output_error[j];
		}
	} else {

		int chunks = g_pool->num_threads > 0 ? g_pool->num_threads : 1;
		int chunk = (layer->input_size + chunks - 1) / chunks;
		for (int start = 0; start < layer->input_size; start += chunk) {
			int end = start + chunk; if (end > layer->input_size) end = layer->input_size;
			InputErrTask *t = malloc(sizeof(InputErrTask));
			t->layer = layer; t->start = start; t->end = end; t->output_error = output_error; t->dest = input_error;
			thread_pool_enqueue(g_pool, input_err_task_fn, t);
		}
		thread_pool_wait(g_pool);


		int out_chunks = g_pool->num_threads > 0 ? g_pool->num_threads : 1;
		int out_chunk = (layer->output_size + out_chunks - 1) / out_chunks;
		for (int start = 0; start < layer->output_size; start += out_chunk) {
			int end = start + out_chunk; if (end > layer->output_size) end = layer->output_size;
			WeightUpdTask *t = malloc(sizeof(WeightUpdTask));
			t->layer = layer; t->start_col = start; t->end_col = end; t->input_vec = layer->input; t->output_err = output_error; t->lr = learning_rate;
			thread_pool_enqueue(g_pool, weight_upd_task_fn, t);
		}
		thread_pool_wait(g_pool);
	}

	double *interim = realloc(output_error, layer->input_size * sizeof(double));
	if (!interim) {
		free(input_error);
		return NULL;
	}

	for (int i = 0; i < layer->input_size; ++i) {
		interim[i] = input_error[i];
	}

	free(input_error);
	return interim;
}

static double* flatten_forprop(Layer *layer, double *input_data) {
	double *result = malloc(layer->output_size * sizeof(double));
	if (!result) {
		return NULL;
	}

	int height = layer->filter_rows;
	int width = layer->filter_cols;
	int channels = layer->num_filters;

	double ***input_3d = (double***)input_data;
	int idx = 0;
	for (int c = 0; c < channels; ++c) {
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				result[idx++] = input_3d[c][h][w];
			}
		}
	}

	layer->input = input_data;
	layer->output = result;
	return result;
}

static double* flatten_backprop(Layer *layer, double *output_error, double learning_rate) {
	(void)learning_rate;

	int height = layer->filter_rows;
	int width = layer->filter_cols;
	int channels = layer->num_filters;

	double ***output = malloc(channels * sizeof(double**));
	for (int c = 0; c < channels; ++c) {
		output[c] = malloc(height * sizeof(double*));
		for (int h = 0; h < height; ++h) {
			output[c][h] = malloc(width * sizeof(double));
		}
	}

	int idx = 0;
	for (int c = 0; c < channels; ++c) {
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				output[c][h][w] = output_error[idx++];
			}
		}
	}

	free(output_error);
	return (double*)output;
}

static double* maxpool_forprop(Layer *layer, double *input_data) {
	int channels = layer->channels;
	int in_h = layer->input_height;
	int in_w = layer->input_width;
	int pool_h = layer->filter_rows;
	int pool_w = layer->filter_cols;
	int stride = layer->stride;
	int out_h = (in_h - pool_h) / stride + 1;
	int out_w = (in_w - pool_w) / stride + 1;
	double *output = malloc(layer->output_size * sizeof(double));
	if (!output) {
		return NULL;
	}
	int *mask = malloc(layer->output_size * sizeof(int));
	if (!mask) {
		free(output); return NULL; 
	}
	if (!g_pool) {
		for (int c = 0; c < channels; ++c) {
			for (int oh = 0; oh < out_h; ++oh) {
				for (int ow = 0; ow < out_w; ++ow) {
					double best = -INFINITY;
					int best_idx = 0;
					for (int ph = 0; ph < pool_h; ++ph) {
						for (int pw = 0; pw < pool_w; ++pw) {
							int ih = oh * stride + ph;
							int iw = ow * stride + pw;
							int idx = c * (in_h * in_w) + ih * in_w + iw;
							double val = input_data[idx];
							if (val > best) { best = val; best_idx = idx; }
						}
					}
					int out_index = c * (out_h * out_w) + oh * out_w + ow;
					output[out_index] = best;
					mask[out_index] = best_idx;
				}
			}
		}
	} else {
		for (int c = 0; c < channels; ++c) {
			MaxPoolTask *t = malloc(sizeof(MaxPoolTask));
			t->input_data = input_data; t->output = output; t->mask = mask; t->channel = c; t->in_h = in_h; t->in_w = in_w; t->out_h = out_h; t->out_w = out_w; t->pool_h = pool_h; t->pool_w = pool_w; t->stride = stride;
			thread_pool_enqueue(g_pool, maxpool_task_fn, t);
		}
		thread_pool_wait(g_pool);
	}
	layer->input = (double*)mask;
	layer->output = output;
	return output;
}

static double* maxpool_backprop(Layer *layer, double *output_error, double learning_rate) {
	(void)learning_rate;
	int in_size = layer->input_size;
	double *input_error = calloc(in_size, sizeof(double));
	if (!input_error){
		return NULL;
	}
	int *mask = (int*)layer->input;
	if (!g_pool) {
		for (int i = 0; i < layer->output_size; ++i) {
			int idx = mask[i];
			input_error[idx] += output_error[i];
		}
		free(output_error);
		return input_error;
	}
    
	int chunks = g_pool->num_threads > 0 ? g_pool->num_threads : 1;
	int chunk = (layer->output_size + chunks - 1) / chunks;
	for (int s = 0; s < layer->output_size; s += chunk) {
		int e = s + chunk; if (e > layer->output_size) e = layer->output_size;
		MaxPoolBackTask *t = malloc(sizeof(MaxPoolBackTask));
		t->mask = mask; t->out_err = output_error; t->dest = input_error; t->start = s; t->end = e;
	thread_pool_enqueue(g_pool, maxpool_back_task_fn, t);
	}
	thread_pool_wait(g_pool);
	free(output_error);
	return input_error;
}

typedef struct {
	Network *net;
	int sample_idx;
	double *x_sample;
	double *y_sample;
	double *output;
	double error;
	double learning_rate;
} ForwardPassJob;

void destroyNetwork(Network *net) {
	if (!net) {
		return;
	}

	if (net->thread_pool) {
		thread_pool_destroy(net->thread_pool);
	}

	Layer *curr = net->head;
	while (curr) {
		Layer *next = curr->next;
		if (curr->type == 0) {
			for (int i = 0; i < curr->input_size; ++i) {
				free(curr->weights[i]);
			}
			free(curr->weights);
			free(curr->bias);
		} else if (curr->type == 2) {
			for (int f = 0; f < curr->num_filters; ++f) {
				if (!curr->convFilters[f]) {
					continue;
				}
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

void enableVisualizer(Network *net, int flag) {
	if (!net) {
		return;
	}
	net->visualizer = flag ? 1 : 0;
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

int reverseInt (int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

double (*load_mnist_images(const char *path, int *num_images))[28][28] {
	FILE *file = fopen(path, "rb");
	if (!file){
		return NULL;
	}
	int magic_number = 0, n_rows = 0, n_cols = 0;
	fread(&magic_number, sizeof(magic_number), 1, file);
	magic_number = reverseInt(magic_number);
	fread(num_images, sizeof(*num_images), 1, file);
	*num_images = reverseInt(*num_images);
	fread(&n_rows, sizeof(n_rows), 1, file);
	n_rows = reverseInt(n_rows);
	fread(&n_cols, sizeof(n_cols), 1, file);
	n_cols = reverseInt(n_cols);
	double (*images)[28][28] = malloc(*num_images * sizeof(*images));
	for (int i = 0; i < *num_images; ++i) {
		for (int r = 0; r < n_rows; ++r){
			for (int c = 0; c < n_cols; ++c) {
				unsigned char temp = 0;
				fread(&temp, sizeof(temp), 1, file);
				images[i][r][c] = (double)temp;
			}
		}
	}
	fclose(file);
	return images;
}

double *load_mnist_labels(const char *path, int *num_labels) {
	FILE *file = fopen(path, "rb");
	if (!file) {
		return NULL;
	}
	int magic_number = 0;
	fread(&magic_number, sizeof(magic_number), 1, file);
	magic_number = reverseInt(magic_number);
	fread(num_labels, sizeof(*num_labels), 1, file);
	*num_labels = reverseInt(*num_labels);
	double *labels = malloc(*num_labels * sizeof(double));
	for (int i = 0; i < *num_labels; ++i) {
		unsigned char temp = 0;
		fread(&temp, sizeof(temp), 1, file);
		labels[i] = (double)temp;
	}
	fclose(file);
	return labels;
}

double*** flat_to_3d(double *flat_img, int channels, int height, int width) {
	double ***img_3d = malloc(channels * sizeof(double**));
	if (!img_3d) return NULL;
	for (int c = 0; c < channels; ++c) {
		img_3d[c] = malloc(height * sizeof(double*));
		if (!img_3d[c]) {
			for (int cc = 0; cc < c; ++cc) free(img_3d[cc]);
			free(img_3d);
			return NULL;
		}
		for (int h = 0; h < height; ++h) {
			img_3d[c][h] = malloc(width * sizeof(double));
			if (!img_3d[c][h]) {
				for (int hh = 0; hh < h; ++hh) free(img_3d[c][hh]);
				free(img_3d[c]);
				for (int cc = 0; cc < c; ++cc) {
					for (int hh = 0; hh < height; ++hh) free(img_3d[cc][hh]);
					free(img_3d[cc]);
				}
				free(img_3d);
				return NULL;
			}
			for (int w = 0; w < width; ++w) {
				img_3d[c][h][w] = flat_img[c * (height * width) + h * width + w];
			}
		}
	}
	return img_3d;
}

void free_3d(double ***img_3d, int channels, int height) {
	if (!img_3d) return;
	for (int c = 0; c < channels; ++c) {
		if (!img_3d[c]) continue;
		for (int h = 0; h < height; ++h) {
			if (img_3d[c][h]) free(img_3d[c][h]);
		}
		free(img_3d[c]);
	}
	free(img_3d);
}

// Conv wrapper definitions: call the conv implementations which use double*** but expose forward/back props as double* to work with Layer struct 
static double* Conv_forward_wrapper(Layer *layer, double *input_data) {
	double ***in3d = (double***)input_data;
	double ***out3d = Conv_forprop(layer, in3d);
	return (double*)out3d;
}

static double* Conv_backward_wrapper(Layer *layer, double *output_error, double learning_rate) {
	double ***err3d = (double***)output_error;
	double ***inerr3d = Conv_backprop(layer, err3d, learning_rate);
	return (double*)inerr3d;
}

/* Demo: train a CNN-style network where inputs are 28x28 images (single channel), 10 output classes */
static void fit_cnn(Network *net, int num_samples, int height, int width, int channels, double *x_train_flat, double *y_train_flat, int num_classes, int epochs, double learning_rate) {
	if (!net || !net->head || !net->tail) return;
	for (int epoch = 0; epoch < epochs; ++epoch) {
		double total_loss = 0.0;
		printf("Epoch %d: ", epoch + 1);
		fflush(stdout);

		size_t img_stride = (size_t)channels * height * width;
		size_t label_stride = (size_t)num_classes;

		for (int i = 0; i < num_samples; ++i) {
			double *sample_ptr = x_train_flat + (size_t)i * img_stride;
			double *norm_buf = malloc(img_stride * sizeof(double));
			if (!norm_buf) return;
			for (size_t k = 0; k < img_stride; ++k) norm_buf[k] = sample_ptr[k] / 255.0;
			double ***img_3d = flat_to_3d(norm_buf, channels, height, width);
			if (!img_3d) { free(norm_buf); return; }

			Layer *curr = net->head;
			double *output = (double*)img_3d;
			while (curr) {
				double *new_output = curr->forward_prop(curr, output);
				output = new_output;
				curr = curr->next;
			}

			double *grad = malloc(num_classes * sizeof(double));
			if (!grad) { free_3d(img_3d, channels, height); return; }
			double *label_ptr = y_train_flat + (size_t)i * label_stride;
			double loss_val = net->loss_function(label_ptr, output, num_classes);
			total_loss += loss_val;
			net->loss_function_prime(label_ptr, output, num_classes, grad);

			curr = net->tail;
			double *output_error = grad;
			while (curr) {
				double *new_error = curr->backward_prop(curr, output_error, learning_rate);
				if (!new_error) {
					free(output_error);
					free_3d(img_3d, channels, height);
					free(norm_buf);
					return;
				}
				output_error = new_error;
				curr = curr->prev;
			}

			curr = net->head;
			while (curr) {
				if (curr->output) {
					free(curr->output);
					curr->output = NULL;
				}
				curr->input = NULL;
				curr = curr->next;
			}

			free_3d(img_3d, channels, height);

			if (i % 100 == 0) printf(".");
			fflush(stdout);
		}

		printf(" Loss: %.4f\n", total_loss / num_samples);
	}
}

int main(void) {
	srand((unsigned)time(NULL)); // Random seed for weight initialization

	int number_of_images = 0, number_of_images_train = 0;
	double (*testImages)[28][28] = load_mnist_images("MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", &number_of_images);
	double (*train_images)[28][28] = load_mnist_images("MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte", &number_of_images_train);
	if (!testImages || !train_images){
		return 1;
	}

	int number_of_labels = 0, number_of_labels_train = 0;
	double *test_labels = load_mnist_labels("MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte", &number_of_labels);
	double *train_labels = load_mnist_labels("MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte", &number_of_labels_train);
	if (!test_labels || !train_labels) {
		free(testImages);
		free(train_images);
		return 1;
	}

	int num_train = 500;  // Smaller dataset for quick CNN test
	if (num_train > number_of_images_train) {
		num_train = number_of_images_train;
	}

	// Test 1: CNN with Conv2D + Flatten + FC
	printf("=== Testing CNN with Conv2D ===\n");
	printf("Building CNN: Conv2D(8 filters, 3x3) → Flatten → FC(128) → ReLU → FC(10)\n");
	
	Network *cnn = initNetwork(cross_entropy_loss, cross_entropy_prime);
	setThreadPoolSize(cnn, 4);
	
	// Create CNN layers
	Layer *conv1 = initConv2D(8, 3, 3, 1, 1, 0);  // 8 filters, 3x3 kernel, 1 channel, stride 1, no padding
	if (!conv1) {
		printf("ERROR: Failed to allocate Conv2D layer\n");
		return 1;
	}
	
	Layer *flatten = initFlatten(8, 26, 26);
	if (!flatten) {
		printf("ERROR: Failed to allocate Flatten layer\n");
		return 1;
	}
	
	Layer *fc1 = initFC(8 * 26 * 26, 128); 
	if (!fc1) {
		printf("ERROR: Failed to allocate FC layer\n");
		return 1;
	}
	
	Layer *act1 = initActivation(relu_activation, relu_p, 128);
	if (!act1) {
		printf("ERROR: Failed to allocate Activation layer\n");
		return 1;
	}
	
	Layer *fc2 = initFC(128, 10);
	if (!fc2) {
		printf("ERROR: Failed to allocate output FC layer\n");
		return 1;
	}
	
	addLayer(cnn, conv1);
	addLayer(cnn, flatten);
	addLayer(cnn, fc1);
	addLayer(cnn, act1);
	addLayer(cnn, fc2);
	
	printf("Preparing training data (labels only)...\n");
	// Well convert images to 3d using flat_to_3d
	double (*y_train)[10] = malloc(num_train * sizeof(*y_train));
	if (!y_train) {
		printf("ERROR: Failed to allocate training labels\n");
		return 1;
	}
	for (int i = 0; i < num_train; ++i) {
		for (int k = 0; k < 10; ++k) {
			y_train[i][k] = 0.0;
		}
		int lbl = (int)train_labels[i];
		if (lbl >= 0 && lbl < 10) {
			y_train[i][lbl] = 1.0;
		}
	}
	
	printf("Training CNN with 4 threads for 3 epochs...\n");
	int epochs = 3;
	double lr = 0.01;
	fit_cnn(cnn, num_train, 28, 28, 1, (double*)train_images, (double*)y_train, 10, epochs, lr);
	printf("CNN training completed successfully!\n");
    
	// Cleanup
	free(y_train);
	destroyNetwork(cnn);
	
	// Test 2: Quick FC baseline
	printf("\n=== Testing FC baseline for comparison ===\n");
	
	double (*x_train)[IMAGE_SIZE] = malloc(num_train * sizeof(*x_train));
	double (*y_train_fc)[10] = malloc(num_train * sizeof(*y_train_fc));
	if (!x_train || !y_train_fc) {
		free(testImages);
		free(train_images);
		free(test_labels);
		free(train_labels);
		return 1;
	}

	for (int i = 0; i < num_train; ++i) {
		for (int r = 0; r < 28; ++r) {
			for (int c = 0; c < 28; ++c) {
				x_train[i][r * 28 + c] = train_images[i][r][c] / 255.0;
			}
		}
		for (int k = 0; k < 10; ++k) {
			y_train_fc[i][k] = 0.0;
		}
		int lbl = (int)train_labels[i];
		if (lbl >= 0 && lbl < 10) {
			y_train_fc[i][lbl] = 1.0;
		}
	}

	Network *net = initNetwork(cross_entropy_loss, cross_entropy_prime);
	setThreadPoolSize(net, 4);
	Layer *fc1_baseline = initFC(IMAGE_SIZE, 128);
	Layer *act1_baseline = initActivation(relu_activation, relu_p, 128);
	Layer *fc2_baseline = initFC(128, 10);
	addLayer(net, fc1_baseline);
	addLayer(net, act1_baseline);
	addLayer(net, fc2_baseline);
	printf("Training FC network with 4 threads and CROSS-ENTROPY loss\n");
	epochs = 10;
	lr = 0.01;
	fit(net, num_train, IMAGE_SIZE, 10, x_train, y_train_fc, epochs, lr);
	int num_test_eval = 1000;
	if (num_test_eval > number_of_images) {
		num_test_eval = number_of_images;
	}
	double **pred = malloc(num_test_eval * sizeof(double*));
	for (int i = 0; i < num_test_eval; ++i) {
		pred[i] = malloc(10 * sizeof(double));
		
		double *test_flat = malloc(IMAGE_SIZE * sizeof(double));
		for (int r = 0; r < 28; ++r){
			for (int c = 0; c < 28; ++c) {
				test_flat[r * 28 + c] = testImages[i][r][c] / 255.0;
			}
		}
		Layer *curr = net->head;
		double *output = test_flat;
		while (curr) {
			output = curr->forward_prop(curr, output);
			curr = curr->next;
		}
		
		for (int k = 0; k < 10; ++k) {
			pred[i][k] = output[k];
		}
		
		free(test_flat);
	}
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

	for (int i = 0; i < num_test_eval; ++i)
		free(pred[i]);
	free(pred);
	free(x_train);
	free(y_train_fc);
	free(testImages);
	free(train_images);
	free(test_labels);
	free(train_labels);
	destroyNetwork(net);
	return 0;
}






