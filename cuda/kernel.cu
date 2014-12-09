
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


	double sum;

	


	
	if (idx < lsize[0])
	{
		out[rowptr_od[0]+idx]=in[idx];  
	}


	__syncthreads();

	if (idx < numl) {
		for(int j=0;j<lsize[idx];j++){		
			sum=0.0;
			for(int k=0;k<lsize[idx-1];k++){		
				sum+= out[rowptr_od[idx-1]+k]*weight[rowptr_w[idx] + lsize[idx]* j + k];	
			}
			sum+=weight[rowptr_w[idx] + lsize[idx]*j+lsize[idx-1]];		
			out[rowptr_od[idx]+j]=(double)(1/(1+exp(-sum)));
		}
	}

	__syncthreads();

	
	if (idx < lsize[numl-1]) {
		delta[rowptr_od[(numl)-1]+idx]=out[rowptr_od[(numl)-1] + idx]*
			(1-out[rowptr_od[(numl)-1] +idx])*
			(tgt[idx]-out[rowptr_od[(numl)-1] + idx]);
	}
	__syncthreads();

	
	if (idx < numl-1 && idx > 0) {
		for(int j=0;j<lsize[idx];j++){
			sum=0.0;
			for(int k=0;k<lsize[idx+1];k++){
				sum+=delta[rowptr_od[idx+1]+k]*weight[rowptr_w[idx+1]+lsize[idx+1]*k+j];
			}
			delta[rowptr_od[idx]+j]=out[rowptr_od[idx]+j]*(1-out[rowptr_od[idx]+j])*sum;
		}
	}
	__syncthreads();

	
	if (idx < numl && idx > 0) {
		for(int j=0;j<lsize[idx];j++){
			for(int k=0;k<lsize[idx-1];k++){
				weight[rowptr_w[idx] + lsize[idx]*j+k]+=(alpha)*prevDwt[rowptr_w[idx] + lsize[idx]*j+k];
			}
			weight[rowptr_w[idx] + lsize[idx]*j+lsize[idx-1]]+=(alpha)*prevDwt[rowptr_w[idx] + lsize[idx]*j+lsize[idx-1]];
		}
	}
	__syncthreads();

	
	if (idx < numl && idx > 0) {
		for(int j=0;j<lsize[idx];j++){
			for(int k=0;k<lsize[idx-1];k++){
				prevDwt[rowptr_w[idx] + lsize[idx]*j+k]=(beta)*delta[rowptr_od[idx]+j]*out[rowptr_od[idx-1]+k];
				weight[rowptr_w[idx] + lsize[idx]*j+k]+=prevDwt[rowptr_w[idx] + lsize[idx]*j+k];
			}
			prevDwt[rowptr_w[idx] + lsize[idx]*j+lsize[idx-1]]=(beta)*delta[rowptr_od[idx]+j];
			weight[rowptr_w[idx] + lsize[idx]*j+lsize[idx-1]]+=prevDwt[rowptr_w[idx] + lsize[idx]*j+lsize[idx-1]];
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
	int i;

	for(i=0;i<lsize[0];i++)
	{
		out[rowptr_od[0]+i]=in[i];  
	}
	
	for(i=1;i<numl;i++){				
		for(int j=0;j<lsize[i];j++){		
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		
				sum+= out[rowptr_od[i-1]+k]*weight[rowptr_w[i] + lsize[i]*j+k];	
			}
			sum+=weight[rowptr_w[i] + lsize[i]*j+lsize[i-1]];		
			out[rowptr_od[i]+j]=(double)(1/(1+exp(-sum)));
		}
	}

	
	for(i=0;i<lsize[(numl)-1];i++){
		delta[rowptr_od[(numl)-1]+i]=out[rowptr_od[(numl)-1]+i]*
			(1-out[rowptr_od[(numl)-1]+i])*(tgt[i]-out[rowptr_od[(numl)-1]+i]);
	}

	for(i=numl-2;i>0;i--){
		for(int j=0;j<lsize[i];j++){
			sum=0.0;
			for(int k=0;k<lsize[i+1];k++){
				sum+=delta[rowptr_od[i+1]+k]*weight[rowptr_w[i+1]+lsize[i+1]*k+j];
			}
			delta[rowptr_od[i]+j]=out[rowptr_od[i]+j]*(1-out[rowptr_od[i]+j])*sum;
		}
	}
	
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				weight[rowptr_w[i] + lsize[i]*j+k]+=(alpha)*prevDwt[rowptr_w[i] + lsize[i]*j+k];
			}
			weight[rowptr_w[i] + lsize[i]*j+lsize[i-1]]+=(alpha)*prevDwt[rowptr_w[i] + lsize[i]*j+lsize[i-1]];
		}
	}
	
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				prevDwt[rowptr_w[i] + lsize[i]*j+k]=(beta)*delta[rowptr_od[i]+j]*out[rowptr_od[i-1]+k];
				weight[rowptr_w[i] + lsize[i]*j+k]+=prevDwt[rowptr_w[i] + lsize[i]*j+k];
			}
			prevDwt[rowptr_w[i] + lsize[i]*j+lsize[i-1]]=(beta)*delta[rowptr_od[i]+j];
			weight[rowptr_w[i] + lsize[i]*j+lsize[i-1]]+=prevDwt[rowptr_w[i] + lsize[i]*j+lsize[i-1]];
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
	int i=0;

	
	for(i=0;i<lsize[0];i++)
	{
		out[rowptr_od[0]+i]=in[i];  
	}

	
	
	for(i=1;i<numl;i++){				
		for(int j=0;j<lsize[i];j++){		
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		
				sum+= out[rowptr_od[i-1]+k]*weight[rowptr_w[i] + lsize[i]*j+k];	
			}
			sum+=weight[rowptr_w[i] + lsize[i]*j+lsize[i-1]];		
			out[rowptr_od[i]+j]=(double)(1/(1+exp(-sum)));
		}
	}
}


