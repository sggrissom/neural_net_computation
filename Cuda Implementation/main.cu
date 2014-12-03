

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
	Timer timer;

	float data[]={
		0,	0,	0,	0,
		0,	0,	1,	1,
		0,	1,	0,	1,
		0,	1,	1,	0,
		1,	0,	0,	1,
		1,	0,	1,	0,
		1,	1,	0,	0,
		1,	1,	1,	1 };

	int inputSize = 3;
	int dataPoints = 8;
	float beta = 0.3;
	float alpha = 0.1;
	float thresh = 0.1;
	int maxIterations = 100000000;
	
	// Initialize host variables ----------------------------------------------
	float** out;
	float** delta;
	float*** weight;
	int* lsize;
	float*** prevDwt;

	cudaError_t cuda_ret;

	// Constants
	const int numl = 4;
	const int outputSize = 1;

	enum Mode {CPU_NORMAL = 1, GPU_NAIVE, GPU_IMPROVED };
	Mode mode;
	enum DataSet {XOR = 1, MUSHROOM };
	DataSet dataSrc;

	if(argc == 2) {
		mode = (Mode) atoi(argv[1]);
		dataSrc = (DataSet) XOR;
	} else if(argc == 3) {
		mode = (Mode) atoi(argv[1]);
		dataSrc = (DataSet) atoi(argv[2]);
	} else {
		printf("\n    Invalid input parameters."
				"\n"
				"\n    Modes: 1 = CPU normal execution"
				"\n           2 = GPU normal execution"
				"\n           3 = GPU with fancy"
				"\n\n");
		exit(0);
	}

	if (dataSrc == XOR) {
	;	
	}

	printf("\nSetting up the problem..."); fflush(stdout);
	startTime(&timer);

	lsize= (int*) malloc(numl * sizeof(int));

	int i, j, k;

	for(i=0;i<numl;i++){
		if (i == numl-1)
			lsize[i] = outputSize;
		else
			lsize[i] = inputSize-1;
	}

	//	allocate memory for output of each neuron
	out = (float**) malloc(numl * sizeof(float*));

	for( i=0;i<numl;i++){
		out[i]= (float*) malloc(lsize[i] * sizeof(float));
	}

	//	allocate memory for delta
	delta = (float**) malloc(numl * sizeof(float*));

		for(i=1;i<numl;i++){
			delta[i]=(float*) malloc(lsize[i] * sizeof(float));
		}

	//	allocate memory for weights
	weight = (float***) malloc(numl * sizeof(float**));

		for(i=1;i<numl;i++){
			weight[i]=(float**) malloc(lsize[i] * sizeof(float*));
			for(j=0;j<lsize[i];j++){
				weight[i][j]=(float*) malloc(lsize[i] * sizeof(float));
			}
		}

	//	allocate memory for previous weights
	prevDwt = (float***) malloc(numl * sizeof(float**));

		for(i=1;i<numl;i++){
			prevDwt[i]=(float**) malloc(lsize[i] * sizeof(float*));
			for(j=0;j<lsize[i];j++){
				prevDwt[i][j]=(float*) malloc(lsize[i] * sizeof(float));
			}
		}

	//	seed and assign random weights
	srand((unsigned)(time(NULL)));
	for(i=1;i<numl;i++)
		for(j=0;j<lsize[i];j++)
			for(k=0;k<lsize[i-1]+1;k++)
				weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;//32767

	//	initialize previous weights to 0 for first iteration
	for(i=1;i<numl;i++)
		for(j=0;j<lsize[i];j++)
			for(k=0;k<lsize[i-1]+1;k++)
				prevDwt[i][j][k]=(double)0.0;


	stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	printf("allocation complete.\n\n"); 

	// Allocate device variables ----------------------------------------------

	if(mode != CPU_NORMAL) {
		printf("Allocating device variables..."); fflush(stdout);
		startTime(&timer);



		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}

	// Copy host variables to device ------------------------------------------

	if(mode != CPU_NORMAL) {
		printf("Copying data from host to device..."); fflush(stdout);
		startTime(&timer);

		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}

	// Launch kernel ----------------------------------------------------------

	printf("Launching kernel ");

	if(mode == CPU_NORMAL) {
		printf("(CPU normal version)...");fflush(stdout);
		startTime(&timer);

		float mse;

		for (i = 0; i < maxIterations; ++i)
		{
			cpu_bpgt(&data[(i%dataPoints)*(inputSize+outputSize)],data[(i%dataPoints)*(inputSize+outputSize) + inputSize],
					out, delta, weight, numl, lsize, beta, alpha, prevDwt);
		}

		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	} else if(mode == GPU_NAIVE) {
		printf("(GPU naive version)...");fflush(stdout);
		startTime(&timer);
		cuda_ret = cudaDeviceSynchronize();
		if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	} else if(mode == GPU_IMPROVED) {
		printf("(GPU better)...");fflush(stdout);
		startTime(&timer);
		cuda_ret = cudaDeviceSynchronize();
		if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	} else {
		printf("Invalid mode!\n");
		exit(0);
	}


	// Copy device variables from host ----------------------------------------

	if(mode != CPU_NORMAL) {

		printf("Copying data from device to host..."); fflush(stdout);
		startTime(&timer);

		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	}

	// Verify correctness -----------------------------------------------------

	printf("Verifying results..."); fflush(stdout);

	int prediction, actual;
	float guess;
	int correct = 0, incorrect = 0;

	for (i = 0; i < dataPoints; ++i)
	{
		cpu_ffwd(&data[i*(inputSize+outputSize)], out, weight, numl, lsize);
		actual = (int) data[i*(inputSize+outputSize) + inputSize];
		guess = out[numl-1][0]; 
		prediction = (int)(guess + 0.5);
		if (prediction == actual)
		{
			correct++;
		} else {
			incorrect++;
		}
	}

	for ( i = 0 ; i < dataPoints ; ++i )
	{
		cpu_ffwd(&data[i*(inputSize+outputSize)], out, weight, numl, lsize);
		for (j=0; j < inputSize; ++j)
		{
			printf("%f ", data[i*(inputSize+outputSize)+j]);
		}
		printf("Ans: %f Guess: %f\n", data[i*(inputSize+outputSize) + inputSize],  out[numl-1][0]);
	}

	float accuracy = ((float)correct)/((float)(correct+incorrect));
	printf("%.2f%% accurate.\n", accuracy * 100);

	// Free memory ------------------------------------------------------------

	free(lsize);

	for( i=0;i<numl;i++){
		free(out[i]);
	}
	free(out);

	for(i=1;i<numl;i++){
		free(delta[i]);
	}
	free(delta);

	for(i=1;i<numl;i++){
		for(j=0;j<lsize[i];j++){
			free(weight[i][j]);
		}
		free(weight[i]);
	}
	free(weight);

	for(i=1;i<numl;i++){
		for(j=0;j<lsize[i];j++){
			free(prevDwt[i][j]);
		}
		free(prevDwt[i]);
	}
	free(prevDwt);

	return 0;
}

