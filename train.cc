#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include "train.h"
#include "CUDAfunc.h"

#define THREUSEMULTIGPU 256

#define DEBUG 0
#define DEBUGGPU 0

#define LZERO (-1.0E10)
static int bunchcount = 1;

/// global varibles
pthread_t thread;
typedef struct _Threadparas
{
	CMyFBLat *FBLatObj;
	int sentnum;
	int *sentlist;
}Threadparas;

void *thread_ReadLat(void *args);
///

GPU_trainBP::GPU_trainBP(Interface *interobj,CMyFBLat *FBLatObj)
{
	size_t free_mem, total_mem;
	int i,j;
	struct WorkPara *Curpara 	= interobj->para;
	GPU_selected							= Curpara->gpu_used;
	max_chunk_frames					= interobj->max_chunk_frames;
	numlayers									= interobj->numlayers;
	bunchsize									= interobj->max_bunch_frames;
	momentum									= Curpara->momentum;
	lrate											= Curpara->lrate;
	weightcost								= Curpara->weightcost;
	myFBLat										= FBLatObj;
	smooth_factor							= interobj->smooth_factor;
	priorscale 								= Curpara->priorscale;
	
	for(i =0; i < numlayers;i++){
		layersizes[i] = Curpara->layersizes[i];
	}
	zero_vec = new float [bunchsize*layersizes[numlayers -1]];
	for(i =0; i< bunchsize*layersizes[numlayers -1];i++){
		zero_vec[i] = LZERO;
	}
	//// set GPU num
	cudaGetDeviceCount(&GPU_N);
	printf("Existed GPU NUMs: %d\n",GPU_N);
	if(GPU_selected == -1)
		printf("GPU NUMs are set to %d\n",GPU_N);
	else{
		GPU_N = 1;
		printf("GPU NUMs are set to %d\n",GPU_N);
		printf("GPU selected is %d\n",GPU_selected);
	}
	
	////Init cublas && streams
	streams = (cudaStream_t*) malloc(MAXGPU * sizeof(cudaStream_t));
	for(i =0;i< GPU_N;i++)
	{
		cudaError_t er;
		if(GPU_N ==1)
		{
			er = cudaSetDevice(GPU_selected);
			if (er!=cudaSuccess)
				printf("cudaSetDevice(%d) failed\n",GPU_selected);
		}
		else
		{
			er = cudaSetDevice(i);
			if (er!=cudaSuccess)
				printf("cudaSetDevice(%d) failed\n",i);
		}

		er =cudaStreamCreate(&(streams[i]));
		if (er!=cudaSuccess)
			printf("cudaStreamCreate(%d) failed\n",i);

		cublasStatus_t eb = cublasCreate(&handles[i]);
		if (eb!=CUBLAS_STATUS_SUCCESS)
			printf("cublasCreate(%d) failed\n",i);
		eb = cublasSetStream(handles[i],streams[i]);
		if (eb!=CUBLAS_STATUS_SUCCESS)
			printf("cublasSetStream(handles[%d],streams[%d]) failed\n",i,i);
	}
	if(GPU_N >1)
	{
		for(i =0;i< GPU_N;i++)
		{
			cudaSetDevice(i);
			for(j =0;j< GPU_N;j++)
			{
				if(j != i)
					cudaDeviceEnablePeerAccess(i, 0);
			}
		}
	}
	
	cudaMemGetInfo(&free_mem, &total_mem);
	total_mem /= 1024*1024;
	free_mem /= 1024*1024;
	printf("Total GPU memory: %d M\n", 	total_mem);
	printf("Free GPU memory:  %d M\n", 	free_mem);
		
	//// Alloc device Memory
	for(j =0;j< GPU_N;j++)
	{
		if(GPU_N > 1)
			cudaSetDevice(j);
		devnew_vf("in", 											max_chunk_frames *layersizes[0], &(dev[j].in));
		devnew_vf("out", 											bunchsize *layersizes[numlayers -1], &(dev[j].out));
		devnew_vi("targ", 										max_chunk_frames, &(dev[j].targ));
		devnew_vf("prior",										layersizes[numlayers -1], &(dev[j].prior));
		devnew_vf("likely", 									bunchsize *layersizes[numlayers -1], &(dev[j].likely));
		//devnew_vf("dedx_fsmooth", 						bunchsize *layersizes[numlayers -1], &(dev[j].dedx_fsmooth));
		for (i = 1; i< numlayers; i++)
		{
			devnew_vf("bias", 	 					layersizes[i], &(dev[j].bias[i]));
			devnew_vf("weights", 					layersizes[i] *layersizes[i-1], &(dev[j].weights[i]));
			devnew_vf("delta_bias", 	 		layersizes[i], &(dev[j].delta_bias[i]));
			devnew_vf("delta_weights", 		layersizes[i] *layersizes[i-1], &(dev[j].delta_weights[i]));
			devnew_vf("layer_y", 					bunchsize *layersizes[i], &(dev[j].layer_y[i]));
			devnew_vf("layer_dydx", 			bunchsize *layersizes[i], &(dev[j].layer_dydx[i]));
			devnew_vf("layer_dedx", 			bunchsize *layersizes[i], &(dev[j].layer_dedx[i]));	
		}
		cudaMemGetInfo(&free_mem, &total_mem);
		total_mem /= 1024*1024;
		free_mem /= 1024*1024;
		printf("Total GPU memory: %d M\n", 	total_mem);
		printf("Free GPU memory:  %d M\n", 	free_mem);
		
		devnew_vf("dedx_num", 		bunchsize *layersizes[numlayers -1], &(dev[j].dedx_num));	
		devnew_vf("dedx_den", 		bunchsize *layersizes[numlayers -1], &(dev[j].dedx_den));	
	}
	
	////copy weights && biases ,prior prob to devices
	for(j =0;j< GPU_N;j++)
	{
		if(GPU_N >1)
			cudaSetDevice(j);
		
		for(i = 1; i< numlayers; i++)
		{
			todev_vf_vf("weights", 	layersizes[i-1] *layersizes[i], Curpara->weights[i], dev[j].weights[i]);
			todev_vf_vf("bias", 		layersizes[i], Curpara->bias[i], dev[j].bias[i]);
		}
		todev_vf_vf("prior", 			layersizes[numlayers -1], Curpara->prior, dev[j].prior);
	}
	if(GPU_N >1){
		cudaDeviceSynchronize();
	}
	
	printf("Created net with %d layers,max bunchsize %d, max chunk frames %d.\n", numlayers, bunchsize, max_chunk_frames);
	cudaMemGetInfo(&free_mem, &total_mem);
	total_mem /= 1024*1024;
	free_mem /= 1024*1024;
	printf("Total GPU memory: %d M\n", 	total_mem);
	printf("Free GPU memory:  %d M\n", 	free_mem);
}

GPU_trainBP::~GPU_trainBP()
{
	int i,j;
	delete []zero_vec;
	////streams & cublas free	
	for(j =0;j< GPU_N;j++)
	{
		if(GPU_N ==1)
		{
			 cudaSetDevice(GPU_selected);
		}
		else
		{
			cudaSetDevice(j);
		}
		
		devfree_vf("in", dev[j].in);
		devfree_vf("out", dev[j].out);
		devfree_vi("targ", dev[j].targ);
		devfree_vf("prior",dev[j].prior);
		devfree_vf("likely", dev[j].likely);
		//devfree_vf("dedx_fsmooth", dev[j].dedx_fsmooth);
		for (i = 1; i< numlayers; i++)
		{
				devfree_vf("weights", dev[j].weights[i]);
				devfree_vf("bias", dev[j].bias[i]);
				devfree_vf("delta_weights", dev[j].delta_weights[i]);
				devfree_vf("delta_bias", dev[j].delta_bias[i]);
				devfree_vf("layer_y", dev[j].layer_y[i]);
				devfree_vf("layer_dedx", dev[j].layer_dedx[i]);
				devfree_vf("layer_dydx", dev[j].layer_dydx[i]);
		}
		cudaFree(dev[j].dedx_num);
		cudaFree(dev[j].dedx_den);
		cublasDestroy(handles[j]);
		cudaStreamDestroy(streams[j]);
	}
}

void GPU_trainBP::train(int n_frames, int n_bunchs, int n_sents, int *samples_in_bunch, int *sent_in_bunch, int *sentlist_in_chunk, float* in, int *targ)
{
	int i;
	int frames_this_bunch;	// Number of frames to handle this bunch
	int sents_this_bunch;
	int n_input = layersizes[0];
	float *realin;
	int *realtarg;
	int *cur_sent_list;

printf("1111\n");

	/// Thread ReadLat
	Threadparas args;
	args.FBLatObj = myFBLat;
	args.sentlist = sentlist_in_chunk;
	args.sentnum = n_sents;
	int status = pthread_create(&thread, NULL, thread_ReadLat, (void *)&args);
	if(status){
		printf("Error: can not create thread <ReadLat>.\n");
		exit(0);
	}

printf("2222\n");
	// First copy data to GPU	
	todev_vf_vf("in", 	n_frames* n_input, in, dev[0].in);
	todev_vi_vi("targ", n_frames, targ, dev[0].targ);
	realin = dev[0].in;
	realtarg = dev[0].targ;
	cur_sent_list = sentlist_in_chunk;

printf("3333\n");

	for (i=0; i< n_bunchs; i++){
		frames_this_bunch = samples_in_bunch[i];
		sents_this_bunch  = sent_in_bunch[i];

		train_bunch_single(frames_this_bunch, sents_this_bunch, realin, cur_sent_list, realtarg);

		realin += n_input * frames_this_bunch;
		realtarg +=  frames_this_bunch;
		cur_sent_list += sents_this_bunch;
	}
}

int GPU_trainBP::CrossValid(int n_frames, float* in, int *targ)
{
	int correct_samples =0;
	int *out = new int [bunchsize];
  int i,j;
  int frames_this_bunch;	// Number of frames to handle this bunch
  int n_input = layersizes[0];
  float *realin;
	
	// First copy data to GPU
	for(i= 0; i< GPU_N;i++)
	{
		if(GPU_N >1)
			cudaSetDevice(i);
    todev_vf_vf("in", n_frames* n_input/GPU_N, in + i* n_frames* n_input/GPU_N, dev[i].in);
  }
  
	///for single gpu
	realin = dev[0].in;
	
	for (i=0; i< n_frames; i+= bunchsize)
  {
    frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;
    if(GPU_N == 1)
			cv_bunch_single(frames_this_bunch, realin, out);
		else{
	//		cv_bunch_multi(frames_this_bunch, realin, out);
		}
			
		//// compute correct_samples
		for(j =0; j< frames_this_bunch;j++)
		{
			if( out[j] == targ[j]){
				correct_samples ++;
			}
		}

    realin += n_input * frames_this_bunch;
    targ += frames_this_bunch;
  }
  
  delete []out;
  return correct_samples;
}

void GPU_trainBP::train_bunch_single(int frames_this_bunch, int sents_this_bunch, const float* in, int* cur_sent_list, const int *targ)
{
		const float one  = 1.0f;
		const float zero = 0.0f;
   	int cur_layer;			// The index of the current layer.
    int prev_layer;			// The index of the previous layer.
    int cur_layer_units;	// The number of units in the current layer.
    int prev_layer_units;	// The number of units in the previous layer.
    int cur_layer_size;		// The size of the current layer.
    
    float* cur_layer_y;				// Output from the current layer
    const float* prev_layer_y;	// Output from the previous non-linearity.
    float* cur_layer_dydx;	// dydx for the current layer.
    float* prev_layer_dedx;	// dedy for the previous layer.
    float* cur_layer_dedx;	// dedx for the current layer.
    float* cur_layer_bias;	// Biases for the current layer.
    float* cur_layer_delta_bias; // Delta biases for the current layer.
    float* cur_layer_delta_weights;
    float* cur_weights;		// Weights inputing to the current layer.
    float cur_lrate =  lrate/frames_this_bunch;
    
    //struct timeval timest,timeen;
    //gettimeofday(&timest,NULL);

	//// Forward
	for (cur_layer=1; cur_layer< numlayers; cur_layer++)
	{
			prev_layer = cur_layer - 1;
			cur_layer_units = layersizes[cur_layer];
			prev_layer_units = layersizes[prev_layer];
			cur_layer_size = cur_layer_units * frames_this_bunch;
			cur_layer_y = dev[0].layer_y[cur_layer];
			if (cur_layer==1)
				prev_layer_y = in;
			else
				prev_layer_y = dev[0].layer_y[prev_layer];
			cur_layer_bias = dev[0].bias[cur_layer];
			cur_weights 	 = dev[0].weights[cur_layer];
#if DEBUGGPU
			cudaError_t err = cudaGetLastError();
			if(err != cudaSuccess){
				printf("before kernel error existed.\n");
			}
#endif
			DevMultiCopy(streams[0],frames_this_bunch, cur_layer_units, cur_layer_bias, cur_layer_y);
			SgemmNN(handles[0],cur_layer_units, prev_layer_units, frames_this_bunch, cur_weights, prev_layer_y, cur_layer_y, one, one); 

			if (cur_layer != numlayers - 1){
				DevSigmoid(streams[0],cur_layer_size, cur_layer_y, cur_layer_y);
			}
			else{
				DevGetLikelyhood(streams[0],frames_this_bunch ,cur_layer_units, cur_layer_y, priorscale, dev[0].prior, dev[0].likely, dev[0].out);
    	}
	}
  		
  	//cudaStreamSynchronize(streams[0]);
  	//gettimeofday(&timeen,NULL);
		//printf("MLP Forward time: %.2f ms\n", 1000.0* (timeen.tv_sec - timest.tv_sec) + (timeen.tv_usec - timest.tv_usec)/1000.0);
		//timest = timeen;
		
    // Backward
    for (cur_layer = numlayers -1; cur_layer >0; cur_layer--)
    {
			prev_layer = cur_layer - 1;
			cur_layer_units = layersizes[cur_layer];
			prev_layer_units = layersizes[prev_layer];
			cur_layer_size = cur_layer_units * frames_this_bunch;
			cur_layer_y = dev[0].layer_y[cur_layer];
			if (cur_layer==1)
			    prev_layer_y = in;
			else
			    prev_layer_y = dev[0].layer_y[prev_layer];
			cur_layer_dydx = dev[0].layer_dydx[cur_layer];
			prev_layer_dedx = dev[0].layer_dedx[prev_layer];
			cur_layer_dedx = dev[0].layer_dedx[cur_layer];
			cur_layer_bias = dev[0].bias[cur_layer];
			cur_layer_delta_bias = dev[0].delta_bias[cur_layer];
			cur_layer_delta_weights = dev[0].delta_weights[cur_layer];
			cur_weights = dev[0].weights[cur_layer];
			
			if (cur_layer != numlayers - 1)
			{
		 	    DevDsigmoid(streams[0], cur_layer_size, cur_layer_y, cur_layer_dydx);
		 	    DevVecMul(streams[0],   cur_layer_size, cur_layer_dydx, cur_layer_dedx, cur_layer_dedx);
			}
			else{
				//gettimeofday(&timest,NULL);
#if DEBUG
/*
	float *tmp = new float[3004];
	cudaMemcpy(tmp, dev[0].likely, sizeof(float)* 3004, cudaMemcpyDeviceToHost);
	for(int k =0;k < 3004 ;k++)
		printf("%f\n",tmp[k]);
	delete []tmp;
	exit(0);
*/
#endif
				cudaMemcpy(dev[0].dedx_num, zero_vec, sizeof(float)* bunchsize* layersizes[numlayers -1],cudaMemcpyHostToDevice);
				cudaMemcpy(dev[0].dedx_den, zero_vec, sizeof(float)* bunchsize* layersizes[numlayers -1],cudaMemcpyHostToDevice);
				cudaStreamSynchronize(streams[0]);
				//計算分子狀態佔有率和分母狀態佔有率
				myFBLat->DoLatFB(sents_this_bunch, cur_sent_list, dev[0].likely, dev[0].dedx_num, dev[0].dedx_den);

				//gettimeofday(&timeen,NULL);
				//printf("Lat FB time: %.2f ms\n", 1000.0* (timeen.tv_sec - timest.tv_sec) + (timeen.tv_usec - timest.tv_usec)/1000.0);
				//DevSubIndex(streams[0], frames_this_bunch, cur_layer_units, dev[0].out, targ, dev[0].dedx_fsmooth);
				//平滑計算：delta J = (1-H) *(delta JCE) + H * (DEN_occ - NUM_occ）
				DevFSmooth(streams[0], frames_this_bunch, cur_layer_units, dev[0].out, targ, dev[0].dedx_num, dev[0].dedx_den, cur_layer_dedx, smooth_factor);
#if 0	
				printf("bunch: %d\n", bunchcount);
				if(bunchcount == 1)
				{
					float *tmp =new float[cur_layer_units*frames_this_bunch];
					fromdev_vf_vf("tmp",cur_layer_units*frames_this_bunch,cur_layer_dedx,tmp);
					for(int l=0;l< cur_layer_units*frames_this_bunch;l++)
						//if(tmp[l] > 1e-8)
							printf("%d: %.10e\n",l, tmp[l]);
					delete []tmp;
					exit(0);
				}
				bunchcount ++;
#endif				
			}
			
			if (cur_layer != 1)
			{
			    SgemmTN(handles[0], prev_layer_units, cur_layer_units, frames_this_bunch, cur_weights, cur_layer_dedx, prev_layer_dedx, zero, one);
			}
			
			// Update weights.
			float lrate_decay = -weightcost *lrate;

			//if(cur_layer == numlayers - 1)
			{
				SgemmNT(handles[0], cur_layer_units, frames_this_bunch, prev_layer_units, cur_layer_dedx, prev_layer_y, cur_layer_delta_weights ,momentum, -cur_lrate);
				DevAccSumrow(streams[0], cur_layer_units, frames_this_bunch, cur_layer_dedx, cur_layer_delta_bias, momentum, -cur_lrate);
				
				cublasSaxpy(handles[0],	cur_layer_units *prev_layer_units, &lrate_decay, cur_weights,	1, cur_layer_delta_weights, 1);
				DevAccSum(streams[0],	cur_layer_units *prev_layer_units, cur_layer_delta_weights,	cur_weights, 1.0);
				
				cublasSaxpy(handles[0],	cur_layer_units, &lrate_decay, cur_layer_bias,	1, cur_layer_delta_bias, 1);
				DevAccSum(streams[0],	cur_layer_units, cur_layer_delta_bias,	cur_layer_bias, 1.0);
			}

/*
		 if(cur_layer ==1){
		float *tmpout = new float[1 *cur_layer_units];
    fromdev_vf_vf("data",1 *cur_layer_units, cur_layer_bias,tmpout);
    for(int tmpj =0 ;tmpj < cur_layer_units ;tmpj ++)
    {
    for(int tmpi =0;tmpi< 1; tmpi++)
    {
    	printf("%f\n",(tmpout[tmpj + tmpi *cur_layer_units]));
    }
    }
    delete [] tmpout;
    exit(0);}
*/
  	}
}

void GPU_trainBP::cv_bunch_single(int n_frames, const float *in, int* out)
{
		const float one  = 1.0f;
		//const float zero = 0.0f;
   	int cur_layer;			// The index of the current layer.
    int prev_layer;			// The index of the previous layer.
    int cur_layer_units;	// The number of units in the current layer.
    int prev_layer_units;	// The number of units in the previous layer.
    int cur_layer_size;		// The size of the current layer.

    float* cur_layer_y;				// Output from the current layer
    const float* prev_layer_y;	// Output from the previous non-linearity.
    float* cur_layer_bias;	// Biases for the current layer.
    float* cur_weights;		// Weights inputing to the current layer.
    
    int *devout;
    devnew_vi("devout", n_frames, &devout);
		//// Forward
    for (cur_layer=1; cur_layer< numlayers; cur_layer++)
		{
				prev_layer = cur_layer - 1;
				cur_layer_units = layersizes[cur_layer];
				prev_layer_units = layersizes[prev_layer];
				cur_layer_size = cur_layer_units * n_frames;
				cur_layer_y = dev[0].layer_y[cur_layer];
				if (cur_layer==1)
					prev_layer_y = in;
				else
					prev_layer_y = dev[0].layer_y[prev_layer];
				cur_layer_bias = dev[0].bias[cur_layer];
				cur_weights 	 = dev[0].weights[cur_layer];
		
				DevMultiCopy(streams[0],n_frames, cur_layer_units, cur_layer_bias, cur_layer_y);
				SgemmNN(handles[0],cur_layer_units, prev_layer_units, n_frames, cur_weights, prev_layer_y, cur_layer_y, one, one); 
    
				if (cur_layer != numlayers - 1){
					DevSigmoid(streams[0],cur_layer_size, cur_layer_y, cur_layer_y);
				}
				else{
					DevSoftmax(streams[0],n_frames, cur_layer_units, cur_layer_y, dev[0].out);
					DevGetMaxIndex(streams[0], cur_layer_units, n_frames, dev[0].out,  devout);
    	  }
		}
		fromdev_vi_vi("devout",n_frames,devout,out);
		devfree_vi("devout",devout);
		
////
//		float *asf = new float[cur_layer_units* n_frames];
//		//fromdev_vf_vf("out", cur_layer_units* n_frames, dev[0].out ,asf);
//		for(int tmp=0;tmp <  n_frames;tmp++)
//   		printf("%d\n",out[tmp]);
//  		delete []asf;
//   		exit(0);

}

void GPU_trainBP::returnWeights(float **weights, float **bias)
{
  int i;
	////copy weights && biases to devices
	if(GPU_N >1)
			cudaSetDevice(0);
	
	for(i = 1; i< numlayers; i++)
	{
		fromdev_vf_vf("weights", layersizes[i-1] *layersizes[i], dev[0].weights[i], weights[i]);
		fromdev_vf_vf("bias", layersizes[i], dev[0].bias[i], bias[i]);
	}
}

///// following are alloc and free functions
void GPU_trainBP::devnew_vf(const char* varname, int n, float **devptr)
{
  cudaError_t cudaStat =  cudaMalloc((void **) devptr, n* sizeof(float));
  if(cudaStat !=cudaSuccess ) 
	{
		printf("%s device momory alloc error\n", varname);
		exit(0);
	}
	float *zero = new float [n];
	for(int i=0;i< n;i++)
		zero[i] = 0.0f;
	cublasStatus_t err = cublasSetVector(n,sizeof(float),zero,1,(*devptr),1);
	if(err != CUBLAS_STATUS_SUCCESS){
		printf("%s cublasSetVector error.\n",varname);
		exit(0);
	}
	delete []zero;
}

void GPU_trainBP::devnew_vd(const char* varname, int n, double **devptr)
{
  cudaError_t cudaStat =  cudaMalloc(devptr, n* sizeof(double));
  if(cudaStat !=cudaSuccess ) 
	{
		printf("%s device momory alloc error\n", varname);
		exit(0);
	}
	double *zero = new double [n];
	for(int i=0;i< n;i++)
		zero[i] = 0.0f;
	cublasStatus_t err = cublasSetVector(n,sizeof(double),zero,1,(*devptr),1);
	if(err != CUBLAS_STATUS_SUCCESS){
		printf("%s cublasSetVector error.\n",varname);
		exit(0);
	}
	delete []zero;
}

void GPU_trainBP::devnew_vi(const char* varname, int n, int **devptr)
{
  cudaError_t cudaStat = cudaMalloc((void **) devptr, n* sizeof(int));
  if(cudaStat !=cudaSuccess ) 
	{
		printf( "%s device momory alloc error\n", varname);
		exit(0);
	}
	int *zero = new int [n];
	for(int i=0;i< n;i++)
		zero[i] = 0;
	cublasStatus_t err = cublasSetVector(n,sizeof(int),zero,1,(*devptr),1);
	if(err != CUBLAS_STATUS_SUCCESS){
		printf("%s cublasSetVector error.\n",varname);
		exit(0);
	}
	delete []zero;
}

void GPU_trainBP::devfree_vf(const char* varname, float* devptr)
{
	cudaFree((void *) devptr);
}

void GPU_trainBP::devfree_vi(const char* varname, int* devptr)
{
  cudaFree((void *) devptr);
}

void GPU_trainBP::todev_vf_vf(const char* varname, int n, const float* from, float* devto)
{
  cublasStatus_t  e = cublasSetVector(n, sizeof(float), from, 1, devto, 1);
  if (e != CUBLAS_STATUS_SUCCESS)
  {
		printf("cuda blas todev_vf_vf error variable %s\n",varname);
		exit(0);
  }
}

void GPU_trainBP::fromdev_vf_vf(const char* varname, int n, const float* devfrom, float* to)
{
  cublasStatus_t e = cublasGetVector(n, sizeof(float), devfrom, 1, to, 1);
  if (e != CUBLAS_STATUS_SUCCESS)
  {
		printf("cuda blas fromdev_vf_vf error variable %s\n",varname);
		exit(0);
  }
}

void GPU_trainBP::todev_vi_vi(const char* varname, int n, const int* from, int* devto)
{
  cublasStatus_t e = cublasSetVector(n, sizeof(int), from, 1, devto, 1);
  if (e != CUBLAS_STATUS_SUCCESS)
  {
		printf("cuda blas todev_vi_vi error variable %s\n", varname);
		exit(0);
  }
}

void GPU_trainBP::fromdev_vi_vi(const char* varname, int n,const int* devfrom, int* to)
{
  cublasStatus_t e = cublasGetVector(n, sizeof(int), devfrom, 1, to, 1);
  if (e != CUBLAS_STATUS_SUCCESS)
  {
		printf("cuda blas fromdev_vi_vi error variable %s\n", varname);
		exit(0);
  }
}

void *thread_ReadLat(void *args)
{
	Threadparas *paras = (Threadparas *) args;
	CMyFBLat *curFBLat = paras->FBLatObj;
	int *cursentlist = paras->sentlist;
	int cursentnum = paras->sentnum;
	
	curFBLat->ReadLatThread(cursentnum , cursentlist);
	return 0;
}