#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "Interface.h"
#include "MyFBLat.h"

#define MAXGPU 		4
#define MAXLAYER	10

struct WorkSpace
{
    float *in; 								/// input data
    float *out;								/// Output data
    int *targ;
    float *likely;						  /// target data
    float *prior;

		float *weights[MAXLAYER];   	/// weights for layers
		float *bias[MAXLAYER];      	/// biases for layers

    float *layer_y[MAXLAYER]; 		/// Output from layer
    float *layer_dydx[MAXLAYER];  /// dy/dx
    float *layer_dedx[MAXLAYER];  /// de/dx
    float *dedx_num;
    float *dedx_den;
    float *dedx_fsmooth;
    float *delta_bias[MAXLAYER]; // Output bias update
    float *delta_weights[MAXLAYER]; // Output bias update
};

class GPU_trainBP
{
public:
		GPU_trainBP(Interface *interobj, CMyFBLat *FBLatObj);
		~GPU_trainBP();
public:
		void forward(int n_frames, const float* in, float *out);
		void forward_bunch(int n_frames, const float* in, float *out);
		void train(int n_frames, int n_bunchs, int n_sents, int *samples_in_bunch, int *sent_in_bunch, int *sentlist_in_chunk, float* in, int *targ);
		int  CrossValid(int n_frames, float* in, int *targ);
		void train_bunch_single(int frames_this_bunch, int sents_this_bunch, const float* in, int* cur_sent_list, const int *targ);
		//void cv_bunch_multi(int n_frames, const float* in, int *targ);
		void cv_bunch_single(int n_frames, const float* in, int *out);
		void returnWeights(float **weights, float **bias);    			/// copy weights and biases from gpu memory to cpu memory 
	 
		int max_chunk_frames;
		int numlayers;
		int layersizes[MAXLAYER];
		int bunchsize;
		float momentum;
		float lrate;
		float weightcost;
private:
		void devnew_vf(const char* varname, int n, float **devptr);
		void devnew_vi(const char* varname, int n, int **devptr);
		void devnew_vd(const char* varname, int n, double **devptr);
		void devfree_vf(const char* varname,  float* devptr);
		void devfree_vi(const char* varname,  int* devptr);
		void todev_vf_vf(const char* varname, int n, const float* from, float* devto);
		void fromdev_vf_vf(const char* varname, int n, const float* devfrom, float* to);
		void todev_vi_vi(const char* varname, int n, const int* from, int* devto);
		void fromdev_vi_vi(const char* varname, int n, const int* devfrom, int* to);
    
		CMyFBLat *myFBLat;
		float priorscale;
		float smooth_factor;
		float *zero_vec;
		
		WorkSpace dev[MAXGPU];  //viaribles for devices
		int GPU_N;							//devices used num
		int GPU_selected;				//devices selected, -1 表示采用所有的GPU
		
		cublasHandle_t handles[MAXGPU];
		cudaStream_t *streams;
		int global_first;
};
