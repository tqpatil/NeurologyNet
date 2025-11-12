// AI GENERATED CODE - DO NOT EDIT
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
	for (int i = 1; i < array_length; ++i)
		if (logits[i] > maxv) maxv = logits[i];
	for (int i = 0; i < array_length; ++i)
		logits_copy[i] = logits[i] - maxv;

	for (int i = 0; i < array_length; ++i)
		exp_buf[i] = exp(logits_copy[i]);

	double sum = 0.0;
	for (int i = 0; i < array_length; ++i)
		sum += exp_buf[i];

	for (int i = 0; i < array_length; ++i)
		probs[i] = exp_buf[i] / sum;

	for (int i = 0; i < array_length; ++i)
		output[i] = (probs[i] - expected[i]) / (double)array_length;

	free(logits_copy);
	free(exp_buf);
	free(probs);
}

void mean_squared_prime(double* expected, double* result, int array_length, double *output) {
	for (int i = 0; i < array_length; ++i)
		output[i] = (2.0 * (expected[i] - result[i])) / array_length;
}

int main() {
	// Sample output from network (logits for CE, outputs for MSE)
	double logits[10] = {0.1, 0.2, 0.15, 0.05, -0.1, 0.3, 0.2, 0.1, -0.05, 0.0};
	double expected[10] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // label 0
	double grad_ce[10], grad_mse[10];
	
	cross_entropy_prime(expected, logits, 10, grad_ce);
	mean_squared_prime(expected, logits, 10, grad_mse);
	
	printf("Logits: ");
	for (int i = 0; i < 10; ++i) printf("%.3f ", logits[i]);
	printf("\n");
	
	printf("CE gradient: ");
	for (int i = 0; i < 10; ++i) printf("%.6f ", grad_ce[i]);
	printf("\n");
	
	printf("MSE gradient: ");
	for (int i = 0; i < 10; ++i) printf("%.6f ", grad_mse[i]);
	printf("\n");
	
	return 0;
}
