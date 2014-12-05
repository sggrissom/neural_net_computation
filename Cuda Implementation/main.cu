
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
	printf("Let's begin");

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

	double*data;
	double *data_d;

	if(dataSet==XOR) {
		double xorData[]={
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

	int sz[4] = {inputSize,
		inputSize-1,
		inputSize-2,
		outputSize};

	int i;

	double beta = 0.3, alpha = 0.1;
	double *beta_d;
	double *alpha_d;
	long num_iter = 2000000;

	double **out;
	double **out_d;
	double **delta;
	double **delta_d;
	double ***weight;
	double ***weight_d;
	int numl=4;
	int*numl_d;
	int *lsize = new int[numl];
	int *lsize_d;
	for(i=0;i<numl;i++){
		lsize[i]=sz[i];
	}
	double ***prevDwt;
	double ***prevDwt_d;

	out = new double*[numl];

	for( i=0;i<numl;i++){
		out[i]=new double[lsize[i]];
	}

	//	allocate memory for delta
	delta = new double*[numl];

	for(i=1;i<numl;i++){
		delta[i]=new double[lsize[i]];
	}

	//	allocate memory for weights
	weight = new double**[numl];

	for(i=1;i<numl;i++){
		weight[i]=new double*[lsize[i]];
	}
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			weight[i][j]=new double[lsize[i-1]+1];
		}
	}

	//	allocate memory for previous weights
	prevDwt = new double**[numl];

	for(i=1;i<numl;i++){
		prevDwt[i]=new double*[lsize[i]];

	}
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			prevDwt[i][j]=new double[lsize[i-1]+1];
		}
	}

	//	seed and assign random weights
	srand((unsigned)(time(NULL)));
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)
				weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;//32767

	//	initialize previous weights to 0 for first iteration
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)
				prevDwt[i][j][k]=(double)0.0;

	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	// Allocate device variables ----------------------------------------------

	if(mode != CPU_NORMAL) {
		printf("Allocating device variables..."); fflush(stdout);
		startTime(&timer);

		cuda_ret = cudaMalloc((void**)&data_d, (inputSize+outputSize) * sizeof(double));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&out_d, numl * sizeof(double));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&delta_d, numl * sizeof(double));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&weight_d, numl * sizeof(double));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&lsize_d, numl * sizeof(double));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&prevDwt, numl * sizeof(double));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&alpha_d, sizeof(double));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&beta_d, sizeof(double));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cuda_ret = cudaMalloc((void**)&numl_d, sizeof(int));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");



		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}

	// Copy host variables to device ------------------------------------------

	if(mode != CPU_NORMAL) {
		printf("Copying data from host to device..."); fflush(stdout);
		startTime(&timer);

		cuda_ret = cudaMemcpy(data_d, data,
				(inputSize+outputSize) * sizeof(double), cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

		cuda_ret = cudaMemcpy(delta_d, delta,
				numl * sizeof(double), cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) {
			FATAL("Unable to copy memory to the device");
		}
		cuda_ret = cudaMemcpy(weight_d, weight,
				numl * sizeof(double), cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) {
			FATAL("Unable to copy memory to the device");
		}
		cuda_ret = cudaMemcpy(lsize_d, lsize,
				numl * sizeof(double), cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) {
			FATAL("Unable to copy memory to the device");
		}
		

		cuda_ret = cudaMemcpy(alpha_d,(double*)&alpha,
				sizeof(double), cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) {
			FATAL("Unable to copy memory to the device");
		}

		cuda_ret = cudaMemcpy(beta_d, (double*) &beta,
				sizeof(double), cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) {
			FATAL("Unable to copy memory to the device");
		}

		cuda_ret = cudaMemcpy(numl_d, (int*) &numl,
				sizeof(int), cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) {
			FATAL("Unable to copy memory to the device");
		}

		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}

	// Launch kernel ----------------------------------------------------------

	printf("Launching kernel ");

	if(mode == CPU_NORMAL) {
		printf("(CPU version)...");fflush(stdout);
		startTime(&timer);

		printf("training the network...");
		for (i=0; i<num_iter ; i++)
		{
			cpu_bpgt(&data[(i%dataPoints)*(inputSize+outputSize)],
					&data[(i%dataPoints)*(inputSize+outputSize) + inputSize],
					out, delta,weight,&numl,lsize,&beta,&alpha,prevDwt);
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
					out_d, delta_d,weight_d,numl_d,lsize_d,beta_d,alpha_d,prevDwt_d);
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
					out_d, delta_d,weight_d,numl_d,lsize_d,beta_d,alpha_d,prevDwt_d);
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

		cuda_ret = cudaMemcpy(out, out_d, numl * sizeof(double),
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
				out,weight,&numl,lsize);
		for (int j=0; j < inputSize; ++j)
		{
			cout << data[i*(inputSize+outputSize)+j] << " ";
		}
		cout << "Ans:" << data[i*(inputSize+outputSize) + inputSize] <<  "  Guess:" << out[numl-1][0] << endl;
	}

	int prediction, actual;
	double guess;
	int correct = 0, incorrect = 0;

	for (int i = 0; i < dataPoints; ++i)
	{
		ffwd(&data[i*(inputSize+outputSize)],
				out,weight,&numl,lsize);
		actual = (int) (data[i*(inputSize+outputSize) + inputSize]);
		guess = out[numl-1][0];
		prediction = (int)(guess + 0.5);
		if (prediction == actual)
		{
			correct++;
		} else {
			incorrect++;
		}
	}

	double accuracy = ((double)correct)/((double)(correct+incorrect));

	cout << "Correct = " << correct << endl;
	cout << "Incorrect = " << incorrect << endl;
	cout << "Accuracy = " << accuracy*100 << "%" << endl;


	//	free out
	for(i=0;i<numl;i++)
		delete[] out[i];
	delete[] out;

	//	free delta
	for(i=1;i<numl;i++)
		delete[] delta[i];
	delete[] delta;

	//	free weight
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] weight[i][j];
	for(i=1;i<numl;i++)
		delete[] weight[i];
	delete[] weight;

	//	free prevDwt
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] prevDwt[i][j];
	for(i=1;i<numl;i++)
		delete[] prevDwt[i];
	delete[] prevDwt;

	//	free layer info
	delete[] lsize;

	// Free memory ------------------------------------------------------------

	if(mode != CPU_NORMAL) {
		cudaFree(data_d);
		cudaFree(out_d);
		cudaFree(delta_d);
		cudaFree(weight_d);
		cudaFree(lsize_d);
		cudaFree(prevDwt_d);
		cudaFree(alpha_d);
		cudaFree(beta_d);
		cudaFree(numl_d);
	}

	return 0;
}
