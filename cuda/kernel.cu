
#include <math.h>

/******************************************************************************
  GPU main computation kernels
 *******************************************************************************/

__global__ void gpu_naive_kernel(double *in,double *tgt,
    double *out,
    double *delta,
    int *rowptr_od,
    double *weight,
    int numl,
    int *lsize,
    double beta,
    double alpha,
    double *prevDwt,
    int *rowptr_w) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ float sum;

  // update output values for each neuron

  // assign content to input layer
  if (idx < lsize[0])
  {
    out[idx]=in[idx];  // output_from_neuron(i,j) Jth neuron in Ith Layer
  }

  __syncthreads();
  sum=0.0;

  int i,k;
  float temp;


  for(i=1;i<numl;i++){				
    if (idx < lsize[i]){
      sum=0.0;
      __syncthreads();
      for(k=0;k<lsize[i-1];k++){		
        temp = (float) ( out[rowptr_od[i-1]+k]* weight[rowptr_w[i]+(idx*(lsize[i-1]+1))+k]);
        atomicAdd(&sum, temp);
      }
      temp = (float) (weight[rowptr_w[i] + (lsize[i-1]+1)*idx+lsize[i-1]]);	
      atomicAdd(&sum, temp);
      __syncthreads();
      out[rowptr_od[i]+idx]=(double)(1/(1+exp(-sum)));
    }
  }

  __syncthreads();

  // find delta for output layer
  if (idx == 0){
    for(i=0;i<lsize[numl-1];i++){
      delta[rowptr_od[numl-1]+i]=out[rowptr_od[numl-1]+i]*(1-out[rowptr_od[numl-1]+i])*(tgt[i]-out[rowptr_od[numl-1]+i]);
    }
  }


  //	find delta for hidden layers
  for(i=numl-2;i>0;i--){
    sum=0.0;
    __syncthreads();
    if (idx<lsize[i]){
      for(k=0;k<lsize[i+1];k++){
        temp = (float) (delta[rowptr_od[i+1]+k]*weight[rowptr_w[i+1]+idx*(lsize[i]+1)+k]);
        atomicAdd(&sum, temp);
      }
      __syncthreads();
      delta[rowptr_od[i]+idx]=out[rowptr_od[i]+idx]*(1-out[rowptr_od[i]+idx])*((double)(sum));
    }
    __syncthreads();
  }

  __syncthreads();

  //	apply momentum ( does nothing if alpha=0 )
  for(i=1;i<numl;i++){
    if(idx<lsize[i]){
      for(k=0;k<lsize[i-1];k++){
        weight[rowptr_w[i]+idx*(lsize[i-1]+1)+k]+=alpha*prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+k];
      }
      weight[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]]+=alpha*prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]];
    }
    __syncthreads();
  }
  __syncthreads();

  //	adjust weights usng steepest descent
  for(i=1;i<numl;i++){
    if(idx<lsize[i]){
      for(k=0;k<lsize[i-1];k++){
        prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+k]=beta*delta[rowptr_od[i]+idx]*out[rowptr_od[i-1]+k];
        weight[rowptr_w[i]+idx*(lsize[i-1]+1)+k]+=prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+k];
      }
      prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]]=beta*delta[rowptr_od[i]+idx];
      weight[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]]+=prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]];
    }
  }
}

__global__ void gpu_improved_kernel(double *in,double *tgt,
    double *out,
    double *delta,
    int *rowptr_od,
    double *weight,
    int numl,
    int *lsize,
    double beta,
    double alpha,
    double *prevDwt,
    int *rowptr_w) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < lsize[0])
  {
  }
}

/******************************************************************************
  Main computation functions
 *******************************************************************************/

void gpu_naive_bpgt(double *in,double *tgt,
    double *out,
    double *delta,
    int *rowptr_od,
    double *weight,
    int numl,
    int *lsize,
    double beta,
    double alpha,
    double *prevDwt,
    int *rowptr_w) {

  const unsigned int numThreadsPerBlock = 512;
  const unsigned int numBlocks = (128 - 1)/numThreadsPerBlock + 1;
  gpu_naive_kernel <<< numBlocks , numThreadsPerBlock >>>
    (in,tgt,out,delta,rowptr_od,weight,numl,lsize,beta,alpha,prevDwt, rowptr_w);
}

void gpu_improved_bpgt(double *in,double *tgt,
    double *out,
    double *delta,
    int *rowptr_od,
    double *weight,
    int numl,
    int *lsize,
    double beta,
    double alpha,
    double *prevDwt,
    int *rowptr_w) {

  const unsigned int numThreadsPerBlock = 512;
  const unsigned int numBlocks = (128 - 1)/numThreadsPerBlock + 1;
  gpu_improved_kernel <<< numBlocks , numThreadsPerBlock >>>
    (in,tgt,out,delta,rowptr_od,weight,numl,lsize,beta,alpha,prevDwt, rowptr_w);

}





void cpu_bpgt(double *in,double *tgt,
    double *out,
    double *delta,
    int *rowptr_od,
    double *weight,
    int numl,
    int *lsize,
    double beta,
    double alpha,
    double *prevDwt,
    int *rowptr_w)
{
  double sum;
  int i,j,k;

  for(i=0;i<lsize[0];i++)
  {
    out[rowptr_od[0]+i]=in[i];  
  }

  for(i=1;i<numl;i++){				
    for(j=0;j<lsize[i];j++){		
      sum=0.0;
      for(k=0;k<lsize[i-1];k++){		
        sum+= out[rowptr_od[i-1]+k]*weight[rowptr_w[i] + (lsize[i-1]+1)*j+k];	
      }
      sum+=weight[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]];		
      out[rowptr_od[i]+j]=(double)(1/(1+exp(-sum)));
    }
  }


  for(i=0;i<lsize[(numl)-1];i++){
    delta[rowptr_od[(numl)-1]+i]=out[rowptr_od[(numl)-1]+i]*
      (1-out[rowptr_od[(numl)-1]+i])*(tgt[i]-out[rowptr_od[(numl)-1]+i]);
  }

  for(i=numl-2;i>0;i--){
    for(j=0;j<lsize[i];j++){
      sum=0.0;
      for(k=0;k<lsize[i+1];k++){
        sum+=delta[rowptr_od[i+1]+k]*weight[rowptr_w[i+1]+(lsize[i]+1)*j+k];
      }
      delta[rowptr_od[i]+j]=out[rowptr_od[i]+j]*(1-out[rowptr_od[i]+j])*sum;
    }
  }

  for(i=1;i<numl;i++){
    for(j=0;j<lsize[i];j++){
      for(k=0;k<lsize[i-1];k++){
        weight[rowptr_w[i] + (lsize[i-1]+1)*j+k]+=(alpha)*prevDwt[rowptr_w[i] + (lsize[i-1]+1)*j+k];
      }
      weight[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]]+=(alpha)*prevDwt[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]];
    }
  }

  for(i=1;i<numl;i++){
    for(j=0;j<lsize[i];j++){
      for(k=0;k<lsize[i-1];k++){
        prevDwt[rowptr_w[i] + (lsize[i-1]+1)*j+k]=(beta)*delta[rowptr_od[i]+j]*out[rowptr_od[i-1]+k];
        weight[rowptr_w[i] + (lsize[i-1]+1)*j+k]+=prevDwt[rowptr_w[i] + (lsize[i-1]+1)*j+k];
      }
      prevDwt[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]]=(beta)*delta[rowptr_od[i]+j];
      weight[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]]+=prevDwt[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]];
    }
  }
}



void ffwd(double *in,
    double *out,
    double *weight,
    int numl,
    int *lsize,
    int *rowptr_od,
    int *rowptr_w)
{
  double sum;
  int i,j,k;

  for(i=0;i<lsize[0];i++)
  {
    out[rowptr_od[0]+i]=in[i];  
  }



  for(i=1;i<numl;i++){				
    for(j=0;j<lsize[i];j++){		
      sum=0.0;
      for(k=0;k<lsize[i-1];k++){		
        sum+= out[rowptr_od[i-1]+k]*weight[rowptr_w[i] + (lsize[i-1]+1)*j+k];	
      }
      sum+=weight[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]];		
      out[rowptr_od[i]+j]=(double)(1/(1+exp(-sum)));
    }
  }
}


