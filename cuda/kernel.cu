#include <math.h>

/*****************************************************************************
  GPU main computation kernels
 *****************************************************************************/


__global__ void gpu_datatest_kernel(double *in,double *tgt,
		double *out,
		double *delta,
		int *rowptr_od,
		double *weight,
		int numl,
		int *lsize,
		double beta,
		double alpha,
		double *prevDwt,
		int *rowptr_w,
		int num_iter) {
	const int numn = 7;
	const int numw = 9;

	printf("\nin:\n");
	for(int i=0; i<32; i++)
		printf("%d ", in[i]);

	printf("\n\nout:\n");
	for (int i=0; i<numn; i++)
		printf("%d ", out[i]);

	printf("\n\ndelta:\n");
	for (int i=0; i<numn; i++)
		printf("%d ", delta[i]);

	printf("\n\nrowptr_od:\n");
	for (int i=0; i<numl+1; i++)
		printf("%d ", rowptr_od[i]);

	printf("\n\nweight:\n");
	for (int i=0; i<numw; i++)
		printf("%d ", weight[i]);

	printf("\n\nnuml:\n%d", numl);

	printf("\n\nlsize:\n");
	for (int i=0; i<numw; i++)
		printf("%d ", prevDwt[i]);

	printf("\n\nbeta:\n%d", beta);

	printf("\n\nalpha:\n%d", alpha);

	printf("\n\nprevDwt:\n");
	for (int i=0; i<numw; i++)
		printf("%d ", prevDwt[i]);

	printf("\n\nrowptr_w:\n");
	for (int i=0; i<numl; i++)
		printf("%d ", rowptr_od[i]);

	printf("\n\nnum_iter:\n%d", num_iter);
}

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
		int *rowptr_w,
		int num_iter) {

	for (int iter=0; iter<num_iter; iter++)
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		int sz =0;
		double sum;

		// update output values for each neuron

		// assign content to input layer
		if (idx < lsize[0])
		{
			// output_from_neuron(i,j) Jth neuron in Ith Layer
			out[idx]=in[idx];
		}

		__syncthreads();

		// assign output(activation) value
		// to each neuron usng sigmoid func
		for (int i=1;i<numl;i++)
		{
			sum=0.0;
			if (idx < lsize[i])
			{
				for (int k=0;k<lsize[i-1];k++)
				{
					/* For input from each neuron in
					 * preceeding layer, apply weight to
					 * inputs and add to sum */
					sum+= out[rowptr_od[i-1]+k]* weight[sz+(idx*(lsize[i-1]+1))+k];
				}
				sum+= weight[sz+lsize[i-1]*idx + lsize[i-1]];
				out[rowptr_od[i]+idx]=(double)(1/(1+exp(-sum)));
			}
			sz+= (lsize[i-1]+1)*lsize[i];
			__syncthreads();
		}

		__syncthreads();

		// find delta for output layer
		if (idx == 0)
		{
			for (int i=0;i<lsize[numl-1];i++)
			{
				delta[rowptr_od[numl-1]+i]=out[rowptr_od[numl-1]+i]*(1-out[rowptr_od[numl-1]+i])*(tgt[i]-out[rowptr_od[numl-1]+i]);
			}
		}

		__syncthreads();

		//	find delta for hidden layers
		for (int i=numl-2;i>0;i--)
		{
			sum=0.0;
			if (idx<lsize[i])
			{
				for (int k=0;k<lsize[i+1];k++)
				{
					sum+=delta[rowptr_od[i+1]+k]*weight[rowptr_w[i+1]+k*(lsize[i]+1)+idx];   // look into // rowptr_WD starts from 1
				}
				delta[rowptr_od[i]+idx]=out[rowptr_od[i]+idx]*(1-out[rowptr_od[i]+idx])*sum;
			}
			__syncthreads();
		}

		__syncthreads();

		//	apply momentum ( does nothing if alpha=0 )
		for (int i=1;i<numl;i++)
		{
			if (idx<lsize[i])
			{
				for (int k=0;k<lsize[i-1];k++)
				{
					weight[rowptr_w[i]+idx*(lsize[i-1]+1)+k]+=alpha*prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+k];
				}
				weight[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]]+=alpha*prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]];
			}
			__syncthreads();
		}
		__syncthreads();

		//	adjust weights usng steepest descent
		for (int i=1;i<numl;i++)
		{
			if (idx<lsize[i])
			{
				for (int k=0;k<lsize[i-1];k++)
				{
					prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+k]=beta*delta[rowptr_od[i]+idx]*out[rowptr_od[i-1]+k];
					weight[rowptr_w[i]+idx*(lsize[i-1]+1)+k]+=prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+k];
				}
				prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]]=beta*delta[rowptr_od[i]+idx];
				weight[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]]+=prevDwt[rowptr_w[i]+idx*(lsize[i-1]+1)+lsize[i-1]];
			}
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
		int *rowptr_w,
		int num_iter) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < lsize[0])
	{
	}
}

/*****************************************************************************
  Main computation functions
 *****************************************************************************/

void gpu_datatest(double *in,double *tgt,
		double *out,
		double *delta,
		int *rowptr_od,
		double *weight,
		int numl,
		int *lsize,
		double beta,
		double alpha,
		double *prevDwt,
		int *rowptr_w,
		int num_iter) {

	const unsigned int numThreadsPerBlock = 1;
	const unsigned int numBlocks = 1;
	gpu_datatest_kernel <<< numBlocks , numThreadsPerBlock >>>
		(in,tgt,out,delta,rowptr_od,weight,numl,lsize,beta,alpha,
		prevDwt,rowptr_w,num_iter);
}

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
		int *rowptr_w,
		int num_iter) {

	const unsigned int numThreadsPerBlock = 512;
	const unsigned int numBlocks = (128 - 1)/numThreadsPerBlock + 1;
	gpu_naive_kernel <<< numBlocks , numThreadsPerBlock >>>
		(in,tgt,out,delta,rowptr_od,weight,numl,lsize,beta,alpha,
		prevDwt,rowptr_w,num_iter);
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
		int *rowptr_w,
		int num_iter) {

	const unsigned int numThreadsPerBlock = 512;
	const unsigned int numBlocks = (128 - 1)/numThreadsPerBlock + 1;
	gpu_improved_kernel <<< numBlocks , numThreadsPerBlock >>>
		(in,tgt,out,delta,rowptr_od,weight,numl,lsize,beta,alpha,
		prevDwt,rowptr_w,num_iter);
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

	for (i=0;i<lsize[0];i++)
	{
		out[rowptr_od[0]+i]=in[i];
	}

	for (i=1;i<numl;i++)
	{
		for (int j=0;j<lsize[i];j++)
		{
			sum=0.0;
			for (int k=0;k<lsize[i-1];k++)
			{
				sum+= out[rowptr_od[i-1]+k]*weight[rowptr_w[i] + (lsize[i-1]+1)*j+k];
			}
			sum+=weight[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]];
			out[rowptr_od[i]+j]=(double)(1/(1+exp(-sum)));
		}
	}

	for (i=0;i<lsize[(numl)-1];i++)
	{
		delta[rowptr_od[(numl)-1]+i]=out[rowptr_od[(numl)-1]+i]*
			(1-out[rowptr_od[(numl)-1]+i])*(tgt[i]-out[rowptr_od[(numl)-1]+i]);
	}

	for (i=numl-2;i>0;i--)
	{
		for (int j=0;j<lsize[i];j++)
		{
			sum=0.0;
			for (int k=0;k<lsize[i+1];k++)
			{
				sum+=delta[rowptr_od[i+1]+k]*weight[rowptr_w[i+1]+(lsize[i]+1)*k+j];
			}
			delta[rowptr_od[i]+j]=out[rowptr_od[i]+j]*(1-out[rowptr_od[i]+j])*sum;
		}
	}

	for (i=1;i<numl;i++)
	{
		for (int j=0;j<lsize[i];j++)
		{
			for (int k=0;k<lsize[i-1];k++)
			{
				weight[rowptr_w[i] + (lsize[i-1]+1)*j+k]+=(alpha)*prevDwt[rowptr_w[i] + (lsize[i-1]+1)*j+k];
			}
			weight[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]]+=(alpha)*prevDwt[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]];
		}
	}

	for (i=1;i<numl;i++)
	{
		for (int j=0;j<lsize[i];j++)
		{
			for (int k=0;k<lsize[i-1];k++)
			{
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
	int i=0;

	for (i=0;i<lsize[0];i++)
	{
		out[rowptr_od[0]+i]=in[i];
	}

	for (i=1;i<numl;i++)
	{
		for (int j=0;j<lsize[i];j++)
		{
			sum=0.0;
			for (int k=0;k<lsize[i-1];k++)
			{
				sum+= out[rowptr_od[i-1]+k]*weight[rowptr_w[i]
						+ (lsize[i-1]+1)*j+k];
			}
			sum+=weight[rowptr_w[i] + (lsize[i-1]+1)*j+lsize[i-1]];
			out[rowptr_od[i]+j]=(double)(1/(1+exp(-sum)));
		}
	}
}

