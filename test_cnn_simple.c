// AI GENERATED CODE - DO NOT EDIT
#include "net.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMAGE_SIZE 784

// Simplified CNN test to find the segfault cause
int main(void) {
	srand((unsigned)time(NULL));

	printf("=== Simplified CNN Segfault Test ===\n");
	printf("Testing Conv2D layer initialization and forward pass...\n");

	// Create a simple test image: 28x28, 1 channel
	printf("[1] Allocating 1-channel 28x28 image...\n");
	double ***img = malloc(1 * sizeof(double**));
	if (!img) {
		printf("ERROR: Could not allocate channels array\n");
		return 1;
	}
	
	img[0] = malloc(28 * sizeof(double*));
	if (!img[0]) {
		printf("ERROR: Could not allocate row pointers\n");
		free(img);
		return 1;
	}
	
	for (int r = 0; r < 28; ++r) {
		img[0][r] = malloc(28 * sizeof(double));
		if (!img[0][r]) {
			printf("ERROR: Could not allocate row %d\n", r);
			return 1;
		}
		for (int c = 0; c < 28; ++c) {
			img[0][r][c] = 0.5; // Simple constant value
		}
	}
	printf("[1] ✓ Image allocated successfully\n");

	// Create a Conv2D layer
	printf("[2] Initializing Conv2D layer (8 filters, 3x3 kernel, 1 channel, stride 1, no padding)...\n");
	Layer *conv = initConv2D(8, 3, 3, 1, 1, 0);
	if (!conv) {
		printf("ERROR: Could not allocate Conv2D layer\n");
		return 1;
	}
	printf("[2] ✓ Conv2D layer allocated\n");

	// Create a network
	printf("[3] Creating network...\n");
	Network *net = initNetwork(cross_entropy_loss, cross_entropy_prime);
	if (!net) {
		printf("ERROR: Could not allocate network\n");
		return 1;
	}
	setThreadPoolSize(net, 4);
	addLayer(net, conv);
	printf("[3] ✓ Network created\n");

	// Attempt forward pass
	printf("[4] Attempting Conv2D forward pass...\n");
	printf("[4a] Calling conv->forward_prop...\n");
	fflush(stdout);
	
	double *result = conv->forward_prop(conv, (double*)img);
	
	if (!result) {
		printf("ERROR: Forward pass returned NULL\n");
		return 1;
	}
	printf("[4] ✓ Conv2D forward pass completed successfully\n");
	printf("     Output should be 8 filters @ 26x26 = %d values\n", 8 * 26 * 26);

	// Cleanup
	printf("[5] Cleaning up...\n");
	for (int r = 0; r < 28; ++r)
		free(img[0][r]);
	free(img[0]);
	free(img);
	
	destroyNetwork(net);
	printf("[5] ✓ Cleanup complete\n");

	printf("\n✅ CNN segfault test PASSED - Conv2D layer works!\n");
	return 0;
}
