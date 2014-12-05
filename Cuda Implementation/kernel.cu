
#include <math.h>

/******************************************************************************
  GPU main computation kernels
 *******************************************************************************/

__global__ void gpu_naive_kernel(double *in,double *tgt,
		double **out,
		double **delta,
		double ***weight,
		int *numl,
		int *lsize,
		double *beta,
		double *alpha,
		double*** prevDwt) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;


	double sum;

	//	update output values for each neuron


	//	assign content to input layer
	if (idx < lsize[0])
	{
		out[0][idx]=in[idx];  // output_from_neuron(i,j) Jth neuron in Ith Layer
	}


	__syncthreads();

	//	assign output(activation) value 
	//	to each neuron usng sigmoid func
	if (idx < *numl) {
		for(int j=0;j<lsize[idx];j++){		// For each neuron in current layer
			sum=0.0;
			for(int k=0;k<lsize[idx-1];k++){		// For input from each neuron in preceeding layer
				sum+= out[idx-1][k]*weight[idx][j][k];	// Apply weight to inputs and add to sum
			}
			sum+=weight[idx][j][lsize[idx-1]];		// Apply bias
			out[idx][j]=(double)(1/(1+exp(-sum)));
		}
	}

	__syncthreads();

	//	find delta for output layer
	if (idx < lsize[*numl-1]) {
		delta[(*numl)-1][idx]=out[(*numl)-1][idx]*(1-out[(*numl)-1][idx])*(tgt[idx]-out[(*numl)-1][idx]);
	}
	__syncthreads();

	//	find delta for hidden layers	
	if (idx < *numl-1 && idx > 0) {
		for(int j=0;j<lsize[idx];j++){
			sum=0.0;
			for(int k=0;k<lsize[idx+1];k++){
				sum+=delta[idx+1][k]*weight[idx+1][k][j];
			}
			delta[idx][j]=out[idx][j]*(1-out[idx][j])*sum;
		}
	}
	__syncthreads();

	//	apply momentum ( does nothing if alpha=0 )
	if (idx < *numl && idx > 0) {
		for(int j=0;j<lsize[idx];j++){
			for(int k=0;k<lsize[idx-1];k++){
				weight[idx][j][k]+=(*alpha)*prevDwt[idx][j][k];
			}
			weight[idx][j][lsize[idx-1]]+=(*alpha)*prevDwt[idx][j][lsize[idx-1]];
		}
	}
	__syncthreads();

	//	adjust weights usng steepest descent	
	if (idx < *numl && idx > 0) {
		for(int j=0;j<lsize[idx];j++){
			for(int k=0;k<lsize[idx-1];k++){
				prevDwt[idx][j][k]=(*beta)*delta[idx][j]*out[idx-1][k];
				weight[idx][j][k]+=prevDwt[idx][j][k];
			}
			prevDwt[idx][j][lsize[idx-1]]=(*beta)*delta[idx][j];
			weight[idx][j][lsize[idx-1]]+=prevDwt[idx][j][lsize[idx-1]];
		}
	}
}

__global__ void gpu_improved_kernel(double *in,double *tgt,
		double **out,
		double **delta,
		double ***weight,
		int *numl,
		int *lsize,
		double *beta,
		double *alpha,
		double*** prevDwt) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < lsize[0])
	{
	}
}

/******************************************************************************
  Main computation functions
 *******************************************************************************/

void gpu_naive_bpgt(double *in,double *tgt,
		double **out,
		double **delta,
		double ***weight,
		int *numl,
		int *lsize,
		double *beta,
		double *alpha,
		double*** prevDwt) {

	const unsigned int numThreadsPerBlock = 512;
	const unsigned int numBlocks = (128 - 1)/numThreadsPerBlock + 1;
	gpu_naive_kernel <<< numBlocks , numThreadsPerBlock >>>
		(in,tgt,out,delta,weight,numl,lsize,beta,alpha,prevDwt);
}

void gpu_improved_bpgt(double *in,double *tgt,
		double **out,
		double **delta,
		double ***weight,
		int *numl,
		int *lsize,
		double *beta,
		double *alpha,
		double*** prevDwt) {

	const unsigned int numThreadsPerBlock = 512;
	const unsigned int numBlocks = (128 - 1)/numThreadsPerBlock + 1;
	gpu_improved_kernel <<< numBlocks , numThreadsPerBlock >>>
		(in,tgt,out,delta,weight,numl,lsize,beta,alpha,prevDwt);

}



//	backpropogate errors from output
//	layer uptill the first hidden layer
void cpu_bpgt(double *in,double *tgt,
		double **out,
		double **delta,
		double ***weight,
		int *numl,
		int *lsize,
		double *beta,
		double *alpha,
		double*** prevDwt)
{
	double sum;
	int i;

	//	update output values for each neuron


	//	assign content to input layer
	for(i=0;i<lsize[0];i++)
	{
		out[0][i]=in[i];  // output_from_neuron(i,j) Jth neuron in Ith Layer
	}

	//	assign output(activation) value 
	//	to each neuron usng sigmoid func
	for(i=1;i<*numl;i++){				// For each layer
		for(int j=0;j<lsize[i];j++){		// For each neuron in current layer
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		// For input from each neuron in preceeding layer
				sum+= out[i-1][k]*weight[i][j][k];	// Apply weight to inputs and add to sum
			}
			sum+=weight[i][j][lsize[i-1]];		// Apply bias
			out[i][j]=(double)(1/(1+exp(-sum)));
		}
	}


	//	find delta for output layer
	for(i=0;i<lsize[(*numl)-1];i++){
		delta[(*numl)-1][i]=out[(*numl)-1][i]*
			(1-out[(*numl)-1][i])*(tgt[i]-out[(*numl)-1][i]);
	}

	//	find delta for hidden layers	
	for(i=*numl-2;i>0;i--){
		for(int j=0;j<lsize[i];j++){
			sum=0.0;
			for(int k=0;k<lsize[i+1];k++){
				sum+=delta[i+1][k]*weight[i+1][k][j];
			}
			delta[i][j]=out[i][j]*(1-out[i][j])*sum;
		}
	}

	//	apply momentum ( does nothing if alpha=0 )
	for(i=1;i<*numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				weight[i][j][k]+=(*alpha)*prevDwt[i][j][k];
			}
			weight[i][j][lsize[i-1]]+=(*alpha)*prevDwt[i][j][lsize[i-1]];
		}
	}

	//	adjust weights usng steepest descent	
	for(i=1;i<*numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				prevDwt[i][j][k]=(*beta)*delta[i][j]*out[i-1][k];
				weight[i][j][k]+=prevDwt[i][j][k];
			}
			prevDwt[i][j][lsize[i-1]]=(*beta)*delta[i][j];
			weight[i][j][lsize[i-1]]+=prevDwt[i][j][lsize[i-1]];
		}
	}
}


// feed forward one set of input
void ffwd(double *in,
		double **out,
		double ***weight,
		int *numl,
		int *lsize)
{
	double sum;
	int i=0;

	//	assign content to input layer
	for(i=0;i<lsize[0];i++)
	{
		out[0][i]=in[i];  // output_from_neuron(i,j) Jth neuron in Ith Layer
	}

	//	assign output(activation) value 
	//	to each neuron usng sigmoid func
	for(i=1;i<*numl;i++){				// For each layer
		for(int j=0;j<lsize[i];j++){		// For each neuron in current layer
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		// For input from each neuron in preceeding layer
				sum+= out[i-1][k]*weight[i][j][k];	// Apply weight to inputs and add to sum
			}
			sum+=weight[i][j][lsize[i-1]];		// Apply bias
			out[i][j]=(double)(1/(1+exp(-sum)));
		}
	}
}



