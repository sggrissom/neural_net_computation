
#include <iostream>
#include <time.h>
#include <stdlib.h>
// NeuralNet.cpp : Defines the entry point for the console application.

#include "kernel.cu"
#include "support.h"

using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
	enum Mode {CPU_NORMAL = 1, GPU_NAIVE, GPU_IMPROVED};
	Mode mode;
	enum DataSet {XOR = 1};
	DataSet dataSet;

	if(argc == 1) {
		mode = CPU_NORMAL;
		dataSet = XOR;
	} else if(argc == 2) {
		mode = (Mode) atoi(argv[1]);
		dataSet = XOR;
	} else if(argc == 3) {
		mode = (Mode) atoi(argv[1]);
		dataSet = (DataSet) atoi(argv[2]);
	}else {
		printf("\n    Invalid input parameters."
				"\n\n");
		exit(0);
	}

	Timer timer;
	cudaError_t cuda_ret;
	printf("\nSetting up the problem..."); fflush(stdout);
	startTime(&timer);

	int inputSize = 3;
	int outputSize = 1;
	int dataPoints = 8;
	int testPoints = 8;

	float *data;
	if (dataSet==XOR) {
		float xorData[]={
			0,	0,	0,	0,
			0,	0,	1,	1,
			0,	1,	0,	1,
			0,	1,	1,	0,
			1,	0,	0,	1,
			1,	0,	1,	0,
			1,	1,	0,	0,
			1,	1,	1,	1 };

		data = xorData;
	}

	int *lsize = new int[4];
	lsize[0] = inputSize;
	lsize[1] = inputSize-1;
	lsize[2] = inputSize-2;
	lsize[3] = outputSize;

	int i;

	float beta = 0.3, alpha = 0.1;
	long num_iter = 2000000;

	float *out;
	float *delta;
	float *weight;
	int numl=4;
	float *prevDwt;

	int numn = 0;
	int rowptr_od[numl+1];
	for(int i=0; i<numl; i++) {
		rowptr_od[i] = numn;
		numn += lsize[i];
	}
	rowptr_od[numl] = numn;

	// Allocate memory for out, delta
	out = new float[numn];
	delta = new float[numn];

	// Allocate memory for weights, prevDwt
	int numw = 0;
	int rowptr_w[numl+1];
	for(int i=0; i<numl-1; i++) {
		rowptr_w[i+1] = numw;
		numw += lsize[i]*lsize[i+1];
	}
	weight = new float[numw];
	prevDwt = new float[numw];

	// Seed and assign random weights; set prevDwt to 0 for first iter
	srand((unsigned)(time(NULL)));
	for(i=1;i<numw;i++) {
		weight[i] = (float)(rand())/(RAND_MAX/2) - 1;//32767
		prevDwt[i] = (float)0.0;
	}

	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	// Allocate device variables ------------------------------------------

	float *data_d;
	float *out_d;
	float *delta_d;
	int *rowptr_od_d;
	float *weight_d;
	float *prevDwt_d;
	int *rowptr_w_d;
	int *lsize_d;

	if(mode != CPU_NORMAL) {
		printf("Allocating device variables...\n"); fflush(stdout);
		startTime(&timer);

		cuda_ret = cudaMalloc((void**)&data_d,
				(inputSize+outputSize)*sizeof(float));
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&out_d, numn*sizeof(float));
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&delta_d, numn*sizeof(float));
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&rowptr_od,
				(numl+1)*sizeof(float));
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&weight_d, numw*sizeof(float));
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&prevDwt_d, numw*sizeof(float));
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&rowptr_w_d,
				(numl+1)*sizeof(float));
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&lsize_d, numl*sizeof(int));
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to allocate device memory");

		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}

	// Copy host varibles to device ---------------------------------------

	if(mode != CPU_NORMAL) {
		printf("Copying data from host to device...\n");fflush(stdout);
		startTime(&timer);

		cuda_ret = cudaMemcpy(data_d, data,
				(inputSize+outputSize)*sizeof(float),
				cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to set device memory");
		cuda_ret = cudaMemcpy(out_d, out, numn*sizeof(float),
				cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to set device memory");
		cuda_ret = cudaMemcpy(delta_d, delta, numn*sizeof(float),
				cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to set device memory");
		cuda_ret = cudaMemcpy(weight_d, weight, numw*sizeof(float),
				cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to set device memory");
		cuda_ret = cudaMemcpy(prevDwt_d, prevDwt, numw*sizeof(float),
				cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to set device memory");
		cuda_ret = cudaMemcpy(rowptr_w_d, rowptr_w, numw*sizeof(float),
				cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to set device memory");
		cuda_ret = cudaMemcpy(lsize_d, lsize, numl*sizeof(int),
				cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess)
			FATAL("Unable to set device memory");

		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}

	// Launch kernel ------------------------------------------------------

	printf("Launching kernel ");

	if(mode == CPU_NORMAL) {
		printf("(CPU version)...");fflush(stdout);
		startTime(&timer);

		printf("training the network...");
		for (i=0; i<num_iter ; i++)
		{
			cpu_bpgt(&data[(i%dataPoints)*(inputSize+outputSize)],
					&data[(i%dataPoints)*(inputSize+outputSize) + inputSize],
					out,delta,rowptr_od,weight,numl,lsize,beta,
					alpha,prevDwt,rowptr_w);
		}

		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	} else if(mode == GPU_NAIVE) {
		printf("(GPU naive version)...");fflush(stdout);
		startTime(&timer);

		printf("training the network...");
		for (i=0; i<num_iter ; i++)
		{
			gpu_naive_bpgt(&data_d[(i%dataPoints)*(inputSize+outputSize)],
					&data_d[(i%dataPoints)*(inputSize+outputSize) + inputSize],
					out_d,delta_d,rowptr_od_d,weight_d,numl,lsize_d,beta,
					alpha,prevDwt_d,rowptr_w_d);
		}

		cuda_ret = cudaDeviceSynchronize();
		if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	} else if(mode == GPU_IMPROVED) {
		printf("(GPU improved version)...");fflush(stdout);
		startTime(&timer);

		printf("training the network...");
		for (i=0; i<num_iter ; i++)
		{
			gpu_improved_bpgt(&data[(i%dataPoints)*(inputSize+outputSize)],
					&data[(i%dataPoints)*(inputSize+outputSize) + inputSize],
					out_d,delta_d,rowptr_od_d,weight_d,numl,lsize_d,beta,
					alpha,prevDwt_d,rowptr_w_d);
		}

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

		cuda_ret = cudaMemcpy(out, out_d, numn * sizeof(float),
				cudaMemcpyDeviceToHost);
		if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	}

	// Verify correctness -----------------------------------------------------



	if ( i == num_iter )
		cout << endl << i << " iterations completed..." << endl;

	cout<< "Now using the trained network to make predctions on test data...." << endl << endl;
	for ( i = 0 ; i < testPoints ; ++i )
	{
		ffwd(&data[i*(inputSize+outputSize)],
				out,weight,numl,lsize);
		for (int j=0; j < inputSize; ++j)
		{
			cout << data[i*(inputSize+outputSize)+j] << " ";
		}
		cout << "Ans:" << data[i*(inputSize+outputSize) + inputSize] <<
		"  Guess:" << out[rowptr_od[numl - 1]] << endl;
	}

	int prediction, actual;
	float guess;
	int correct = 0, incorrect = 0;

	for (int i = 0; i < dataPoints; ++i)
	{
		ffwd(&data[i*(inputSize+outputSize)],
				out,weight,numl,lsize);
		actual = (int) (data[i*(inputSize+outputSize) + inputSize]);
		guess = out[rowptr_od[numl-1]];
		prediction = (int)(guess + 0.5);
		if (prediction == actual)
		{
			correct++;
		} else {
			incorrect++;
		}
	}

	float accuracy = ((float)correct)/((float)(correct+incorrect));

	cout << "Correct = " << correct << endl;
	cout << "Incorrect = " << incorrect << endl;
	cout << "Accuracy = " << accuracy*100 << "%" << endl;


	// free out
	delete[] out;

	// free delta
	delete[] delta;

	// free weight
	delete[] weight;

	// free prevDwt
	delete[] prevDwt;

	// free layer info
	delete[] lsize;

	// Free memory ------------------------------------------------------------

	if(mode != CPU_NORMAL) {
		cudaFree(data_d);
		cudaFree(out_d);
		cudaFree(delta_d);
		cudaFree(rowptr_od_d);
		cudaFree(weight_d);
		cudaFree(prevDwt_d);
		cudaFree(rowptr_w_d);
		cudaFree(lsize_d);
	}

	return 0;
}
