
#include <iostream>
#include <time.h>
#include <stdlib.h>

#include "kernel.cu"
#include "support.h"
#include "dataParser.h"

using std::cout;
using std::endl;

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCheckError( const char *file, const int line )
{
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }

  return;
}

int main(int argc, char* argv[])
{

  enum Mode {CPU_NORMAL = 1, GPU_NAIVE, GPU_IMPROVED};
  Mode mode;
  enum DataSet {XOR = 1, MUSHROOM, IMAGE};
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

  double *data;
  int inputSize;
  int outputSize;
  int dataPoints;
  int testPoints = 8;

  printf("lets do this!! %d\n", dataSet);

  switch (dataSet) {
    case 1: {
              inputSize = 3;
              outputSize = 1;
              dataPoints = 8;

              double xorData[]={
                0,    0,  0,  0,
                0,    0,  1,  1,
                0,    1,  0,  1,
                0,    1,  1,  0,
                1,    0,  0,  1,
                1,    0,  1,  0,
                1,    1,  0,  0,
                1,    1,  1,  1 };

              data = xorData;

            } break;
    case 2: {
              inputSize = 22;
              outputSize = 1;
              dataPoints = 8124;

              double mushroomData[dataPoints *  (inputSize + outputSize)];
              getTestData("./mushroom.data", mushroomData,(inputSize + outputSize), 0);
              data = mushroomData;
            } break;
    case 3:
            {
              printf("xxxxx");

              inputSize = 784;
              outputSize = 1;
              dataPoints = 42000;
              printf("\n\n\nx");

              double *imageData = new double[dataPoints *  (inputSize + outputSize)];
              getTestData("./train.csv", imageData, (inputSize + outputSize), 1);
              data = imageData;
            } break;
    default: {
               exit(-5);
             }
  }

  int *lsize = new int[4];
  lsize[0] = inputSize;
  lsize[1] = inputSize-1;
  lsize[2] = inputSize-2;
  lsize[3] = outputSize;

  int i;

  double beta = 0.3, alpha = 0.1;
  long num_iter = 1000;

  double *out;
  double *delta;
  double *weight;
  int numl=4;
  double *prevDwt;

  int numn = 0;
  int rowptr_od[numl];
  for (i=0; i<numl; i++)
  {
    rowptr_od[i] = numn;
    numn += lsize[i];
  }
  rowptr_od[numl] = numn;

  // Allocate memory for out, delta
  out = new double[numn];
  delta = new double[numn];

  // Allocate memory for weights, prevDwt
  int numw = 0;
  int rowptr_w[numl+1];
  rowptr_w[0] = 0;
  for (i=1; i<numl; i++)
  {
    rowptr_w[i] = numw;
    numw += lsize[i]*(lsize[i-1]+1);
  }
  weight = new double[numw];
  prevDwt = new double[numw];

  // Seed and assign random weights; set prevDwt to 0 for first iter
  srand((unsigned)(time(NULL)));
  for (i=1;i<numw;i++)
  {
    weight[i] = (double)(rand())/(RAND_MAX/2) - 1;//32767
    prevDwt[i] = (double)0.0;
  }

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  // Allocate device variables ------------------------------------------

  double *data_d;
  double *out_d;
  double *delta_d;
  int *rowptr_od_d;
  double *weight_d;
  double *prevDwt_d;
  int *rowptr_w_d;
  int *lsize_d;

  if (mode != CPU_NORMAL)
  {
    printf("Allocating device variables...\n"); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&data_d,
        dataPoints*(inputSize+outputSize)*sizeof(double));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&out_d, numn*sizeof(double));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&delta_d, numn*sizeof(double));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&rowptr_od_d,
        (numl+1)*sizeof(int));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&weight_d, numw*sizeof(double));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&prevDwt_d, numw*sizeof(double));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&rowptr_w_d, numw*sizeof(int));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&lsize_d, numl*sizeof(int));
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }

  // Copy host varibles to device ---------------------------------------

  if (mode != CPU_NORMAL)
  {
    printf("Copying data from host to device...\n");fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(data_d, data,
        dataPoints*(inputSize+outputSize)*sizeof(double),
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");
    cuda_ret = cudaMemcpy(out_d, out, numn*sizeof(double),
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");
    cuda_ret = cudaMemcpy(delta_d, delta, numn*sizeof(double),
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");
    cuda_ret = cudaMemcpy(rowptr_od_d, rowptr_od,
        (numl+1)*sizeof(int),cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");
    cuda_ret = cudaMemcpy(weight_d, weight, numw*sizeof(double),
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");
    cuda_ret = cudaMemcpy(prevDwt_d, prevDwt, numw*sizeof(double),
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");
    cuda_ret = cudaMemcpy(rowptr_w_d, rowptr_w, numw*sizeof(int),
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");
    cuda_ret = cudaMemcpy(lsize_d, lsize, numl*sizeof(int),
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }

  // Launch kernel ------------------------------------------------------

  printf("Launching kernel ");

  if (mode == CPU_NORMAL)
  {
    printf("(CPU version)...");fflush(stdout);
    startTime(&timer);

    printf("training the network...");
    for (i=0; i<num_iter ; i++)
    {
      cpu_bpgt(&data[(i%dataPoints)*(inputSize+outputSize)],
          &data[(i%dataPoints)*(inputSize
            + outputSize) + inputSize],out,delta,
          rowptr_od,weight,numl,lsize,beta,
          alpha,prevDwt,rowptr_w);
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  } else if (mode == GPU_NAIVE)
  {
    printf("(GPU naive version)...");fflush(stdout);
    startTime(&timer);

    printf("training the network...");

    gpu_naive_bpgt(data_d,out_d,delta_d,rowptr_od_d,
        weight_d,numl,lsize_d,beta,alpha,prevDwt_d,
        rowptr_w_d,num_iter, (inputSize + outputSize), dataPoints);


    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to launch/execute kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  } else if (mode == GPU_IMPROVED)
  {
    printf("(GPU improved version)...");fflush(stdout);
    startTime(&timer);

    printf("training the network...");

    gpu_naive_bpgt(data_d,out_d,delta_d,rowptr_od_d,
        weight_d,numl,lsize_d,beta,alpha,prevDwt_d,
        rowptr_w_d,num_iter, (inputSize + outputSize), dataPoints);


    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to launch/execute kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  } else
  {
    printf("Invalid mode!\n");
    exit(0);
  }

  // Copy device variables from host ------------------------------------

  if (mode != CPU_NORMAL)
  {
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(out, out_d, numn*sizeof(double),
        cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");
    cuda_ret = cudaMemcpy(weight, weight_d, numw*sizeof(double),
        cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
      FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }

  // Verify correctness -------------------------------------------------

  cout << endl << num_iter << " iterations completed..." << endl;

  cout<< "Now using the trained network to make predctions on test data...." << endl << endl;

  if(!IMAGE) {
    for (i=0; i<testPoints; i++)
    {
      ffwd(&data[i*(inputSize+outputSize)],
          out,weight,numl,lsize, rowptr_od, rowptr_w);
      for (int j=0; j < inputSize; j++)
      {
        cout << data[i*(inputSize+outputSize)+j] << " ";
      }
      cout << "Ans:" << data[i*(inputSize+outputSize) + inputSize] <<
        "  Guess:" << out[rowptr_od[numl - 1]] << endl;
    }
  }

  int prediction, actual;
  double guess;
  int correct = 0, incorrect = 0;

  for (i=0; i<dataPoints || i < 8000; i++)
  {
    ffwd(&data[i*(inputSize+outputSize)],
        out,weight,numl,lsize, rowptr_od, rowptr_w);
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

  double accuracy = ((double)correct)/((double)(correct+incorrect));

  cout << "Correct = " << correct << endl;
  cout << "Incorrect = " << incorrect << endl;
  cout << "Accuracy = " << accuracy*100 << "%" << endl;

  // Free memory --------------------------------------------------------

  delete[] out;
  delete[] delta;
  delete[] weight;
  delete[] prevDwt;
  delete[] lsize;

  if (mode != CPU_NORMAL) {
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

