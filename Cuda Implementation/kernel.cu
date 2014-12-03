

/******************************************************************************
  GPU kernels
 *******************************************************************************/

__global__ void gpu_naive_kernel(float** in, int len) {

	// INSERT KERNEL CODE HERE

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < len)
	{

	}
}

/******************************************************************************
  computation functions
 *******************************************************************************/

void cpu_bpgt(float* in,
		float ans,
		float** out,
		float** delta,
		float*** weight,
		int numl,
		int* lsize,
		float beta,
		float alpha,
		float*** prevDwt) {


	float sum;
	int i=0;

	//	update output values for each neuron
	//	assign content to input layer
	for(i=0;i<lsize[0];i++)
	{
		out[0][i]=in[i];  // output_from_neuron(i,j) Jth neuron in Ith Layer
	}

	//	assign output(activation) value 
	//	to each neuron usng sigmoid func
	for(i=1;i<numl;i++){				// For each layer
		for(int j=0;j<lsize[i];j++){		// For each neuron in current layer
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		// For input from each neuron in preceeding layer
				sum+= out[i-1][k]*weight[i][j][k];	// Apply weight to inputs and add to sum
			}
			sum+=weight[i][j][lsize[i-1]];		// Apply bias
			out[i][j]= (1/(1+exp(-in))); 		// Apply sigmoid function
		}
	}



	//	find delta for output layer
	for(i=0;i<lsize[numl-1];i++){
		delta[numl-1][i]=out[numl-1][i]*
			(1-out[numl-1][i])*(ans-out[numl-1][i]);
	}

	//	find delta for hidden layers	
	for(i=numl-2;i>0;i--){
		for(int j=0;j<lsize[i];j++){
			sum=0.0;
			for(int k=0;k<lsize[i+1];k++){
				sum+=delta[i+1][k]*weight[i+1][k][j];
			}
			delta[i][j]=out[i][j]*(1-out[i][j])*sum;
		}
	}

	//	apply momentum ( does nothing if alpha=0 )
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				weight[i][j][k]+=alpha*prevDwt[i][j][k];
			}
			weight[i][j][lsize[i-1]]+=alpha*prevDwt[i][j][lsize[i-1]];
		}
	}

	//	adjust weights usng steepest descent	
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				prevDwt[i][j][k]=beta*delta[i][j]*out[i-1][k];
				weight[i][j][k]+=prevDwt[i][j][k];
			}
			prevDwt[i][j][lsize[i-1]]=beta*delta[i][j];
			weight[i][j][lsize[i-1]]+=prevDwt[i][j][lsize[i-1]];
		}
	}
}


// feed forward one set of input
void cpu_ffwd(float *in
		float** out,
		float*** weight,
		int numl,
		int* lsize
		)
{
	float sum;
	int i=0;

	//	assign content to input layer
	for(i=0;i<lsize[0];i++)
	{
		out[0][i]=in[i];  // output_from_neuron(i,j) Jth neuron in Ith Layer
	}

	//	assign output(activation) value 
	//	to each neuron usng sigmoid func
	for(i=1;i<numl;i++){				// For each layer
		for(int j=0;j<lsize[i];j++){		// For each neuron in current layer
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		// For input from each neuron in preceeding layer
				sum+= out[i-1][k]*weight[i][j][k];	// Apply weight to inputs and add to sum
			}
			sum+=weight[i][j][lsize[i-1]];		// Apply bias
			out[i][j]= (1/(1+exp(-in)));// Apply sigmoid function
		}
	}
}

float mse(float *tgt, float **out, int lsize) const
{
	double mse=0;
	for(int i=0;i<lsize[numl-1];i++){
		mse+=(tgt[i]-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
	}
	return mse/2;
}

