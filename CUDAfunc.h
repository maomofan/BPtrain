#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

static const int CUDA_MAXBLOCKS = 65535;
static const int NTHREADS = 256;

#define LZEROEMIT (-100.0)

__global__ void kernSigmoid(int n, float* in_vec, float* out_vec);
__global__ void kernBinary(int n, float* in_vec, float* rand_vec);
__global__ void kernMultiCopy(int mat_height, int vec_len, float* vec, float* mat);
__global__ void kernSumcol(int rows, int cols, float* in, float* res);
__global__ void kernAccSumcol(int rows, int cols, float* in, float* res, float alpha, float beta);
__global__ void kernAccSumrow(int rows, int cols, float* in, float* res, float alpha, float beta);
__global__ void kernSoftmax(int rows, int cols, float *in_vec, float* out_vec); 
__global__ void kernDsigmoid(int n, float* in_vec, float* out_vec);
__global__ void kernVecMul(int n, float *in_vec1, float *in_vec2, float *res_vec);
__global__ void kernSubIndex(int rows , int cols, float *in_vec1, const int *in_index, float *res_vec);
__global__ void kernAccSum(int n, int size, float* in, float* res, float beta);
__global__ void kernGetMaxIndex(int rows, int cols, float* invec, int* outvec);
__global__ void kernGetLikelyhood(int rows, int cols, float* in_vec,  float priscale,float* pri_vec, float* out_vec, float *out_vec2);
__global__ void kernGetGrad(int n, int size, float *vec1,float * vec2);
__global__ void kernFSmooth(int frames, int layersize, int size, float *vecout, const int *targ, float *vec_num, float *vec_den, float *vec_res, float beta);

inline void SgemmTN(cublasHandle_t handle,int m, int k,
			int n, const float* A, const float* B, float* C, 
			const float alpha, const float beta)
{		
    cublasStatus_t e =cublasSgemm(handle,CUBLAS_OP_T, CUBLAS_OP_N,
		m, n, k, &beta, (float*)A, k, (float*) B, k, &alpha, C, m);
		if(e != CUBLAS_STATUS_SUCCESS)
		{
			printf("...........SgemmTN wrong\n");
		}
		if(e == CUBLAS_STATUS_EXECUTION_FAILED)
		{
			printf("...........1\n");
		}
}

inline void SgemmNN(cublasHandle_t handle,int m, int k,
			int n, const float* A,const float* B, float* C,
			const float alpha, const float beta)
{
    cublasStatus_t e =cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k, &beta, (float*)A, m, (float*) B, k, &alpha, C, m);
		if(e != CUBLAS_STATUS_SUCCESS)
		{
			printf("...........SgemmNN wrong\n");
		}
}

inline void SgemmNT(cublasHandle_t handle,int m, int k,
			int n, const float* A,
			const float* B, float* C, const float alpha, const float beta)
{
    cublasStatus_t e =cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_T,
		m, n, k, &beta, (float*)A, m, (float*) B, n, &alpha, C, m);
		if(e != CUBLAS_STATUS_SUCCESS)
		{
			printf("...........SgemmNT wrong\n");
		}
}

inline void DevSigmoid(cudaStream_t stream, int n, float* in_vec, float* out_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevSigmoid: nblocks too large\n");
    kernSigmoid<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, out_vec);
}

inline void DevDsigmoid(cudaStream_t stream, int n, float* in_vec, float* out_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks> CUDA_MAXBLOCKS)
				printf("DevDsigmoid: nblocks too large\n");
    kernDsigmoid<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, out_vec);
}

inline void DevSoftmax(cudaStream_t stream, int rows, int cols, float* in_vecs, float* out_vecs)
{
    int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevSoftmax: nblocks too large\n");
    kernSoftmax<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in_vecs, out_vecs);
}

inline void DevMultiCopy(cudaStream_t stream,int mat_height, int vec_len,
		  float* vec, float* mat)
{
    int nblocks = (vec_len + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("DevMultiCopy: nblocks too large\n");
    kernMultiCopy<<<nblocks, NTHREADS,0,stream>>>(mat_height, vec_len, vec, mat);
}

inline void DevSumcol(cudaStream_t stream,int rows, int cols, float* in, float* res)
{
    int nblocks = (cols + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("DevSumcol: nblocks too large\n");
    kernSumcol<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in, res);
}

inline void DevAccSumcol(cudaStream_t stream,int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int nblocks = (cols + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("DevSumcol: nblocks too large\n");
    kernAccSumcol<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in, res, alpha, beta);
}

inline void DevAccSumrow(cudaStream_t stream,int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("DevSumrow: nblocks too large\n");
    kernAccSumrow<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in, res, alpha, beta);
}

inline void DevAccSum(cudaStream_t stream, int n, float* in, float* res, float beta)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    int size = 1;
    while (nblocks > CUDA_MAXBLOCKS)
    {
    	//printf("DevAccSum: nblocks %d too large, we cut half.\n", nblocks);
    	nblocks = (nblocks + 1)/2;
    	size *= 2;
    }
		
    kernAccSum<<<nblocks, NTHREADS,0,stream>>>(n, size, in, res,  beta);
}

inline void DevVecMul(cudaStream_t stream, int n, float *in_vec1, float *in_vec2, float *res_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevVecMul: nblocks too large\n");
    kernVecMul<<<nblocks,NTHREADS,0,stream>>>(n, in_vec1, in_vec2, res_vec);
}

inline void DevSubIndex(cudaStream_t stream, int rows , int cols, float *in_vec1, const int *in_index, float *res_vec)
{
	 int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevSubIndex: nblocks too large\n");
    kernSubIndex<<<nblocks,NTHREADS,0,stream>>>( rows, cols, in_vec1, in_index, res_vec);
}

inline void DevGetMaxIndex(cudaStream_t stream, int rows , int cols, float *invec, int *outvec)
{
	 	int nblocks = (cols + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevSubIndex: nblocks too large\n");
    kernGetMaxIndex<<<nblocks,NTHREADS,0,stream>>>( rows, cols, invec, outvec);
}

inline void DevGetLikelyhood(cudaStream_t stream, int rows, int cols, float *in_vec, float priscale, float *pri_vec, float *out_vec, float *out_vec2)
{
	int nblocks = (rows + NTHREADS-1)/NTHREADS;
	if (nblocks > CUDA_MAXBLOCKS)
		printf("DevGetLikelyhood: nblocks too large\n");
	kernGetLikelyhood<<<nblocks,NTHREADS,0,stream>>>(rows, cols, in_vec, priscale, pri_vec, out_vec, out_vec2);
}

inline void DevGetGrad(cudaStream_t stream, int n, float *vec1, float *vec2)
{
		int nblocks = (n + NTHREADS-1)/NTHREADS;
    int size = 1;
    while (nblocks > CUDA_MAXBLOCKS)
    {
    	//printf("DevGetGrad: nblocks %d too large, we cut half.\n", nblocks);
    	nblocks = (nblocks + 1)/2;
    	size *= 2;
    }
		
    kernGetGrad<<<nblocks, NTHREADS,0,stream>>>(n, size, vec1, vec2);
}

inline void DevFSmooth(cudaStream_t stream, int frames, int layersize, float *vecout, const int *targ, float *vec_num, float *vec_den, float *vec_res, float beta)
{
		int nblocks = (frames * layersize + NTHREADS-1)/NTHREADS;
    int size = 1;
    while (nblocks > CUDA_MAXBLOCKS)
    {
    	//printf("DevFSmooth: nblocks %d too large, we cut half.\n", nblocks);
    	nblocks = (nblocks + 1)/2;
    	size *= 2;
    }
		
    kernFSmooth<<<nblocks, NTHREADS,0,stream>>>(frames, layersize, size, vecout, targ, vec_num, vec_den, vec_res, beta);
}