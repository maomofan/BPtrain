#include "CUDAfunc.h"
#include <stdlib.h>

__global__ void kernBinary(int n, float* in_vec, float* rand_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
    {
			if(in_vec[i] > rand_vec[i])
			{
				in_vec[i] = 1.0f;
			}
			else
			{
				in_vec[i] = 0.0f;
			}
		}
}

__global__ void kernSigmoid(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
			out_vec[i] = 1.0f/(1.0f + expf(- in_vec[i]));
}

__global__ void kernDsigmoid(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
    {
			const float y = in_vec[i];
			out_vec[i] = (1.0f - y) * y;
    }
}

__global__ void  kernSoftmax(int rows, int cols, float* in_vec, float* out_vec)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row < rows)
    {
			int i;
			const int index = row * cols;
			const float* invec = &in_vec[index];
		  float* outvec = &out_vec[index];
			const float* inptr;
			float* outptr;
		
			// First find the max of each vector
			float max;
			
			inptr = invec;
			max = *inptr++;
			for (i=cols-1; i!=0; i--)
			{
			    float val;
		
			    val = *inptr++;
			    if (val>max)
				max = val;
			}
			// Now put exp(in-max) in out
			inptr = invec;
			outptr = outvec;
			float sumexp = 0;
			for (i=cols; i!=0; i--)
			{
			    float f, e;
			    
			    f = *inptr++;
			    e = expf(f - max);
			    *outptr++ = e;
			    sumexp += e;
			}
			// Now scale the output
			float scale = 1.0f/sumexp;
			outptr = outvec;
			for (i=cols; i!=0; i--)
			{
			    *outptr = (*outptr) * scale;
			    outptr++;
			}
    }
}

__global__ void kernMultiCopy(int mat_height, int vec_len,
		   float* vec, float* mat)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < vec_len)
    { 
			int j;
			float val = vec[col];
			float* top = &mat[col];
			for (j=mat_height; j!=0; j--)
			{
			    *top = val;
			    top += vec_len;
			}
    }
}

__global__ void kernSumcol(int rows, int cols, float* in, float* res)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < cols)
    {
			int j;
			const float* fromp = &in[col];
			float* top = &res[col];
			
			(*top) = (*fromp);
			fromp +=cols;
			for (j=rows-1; j!=0; j--)
			{
			    (*top) += (*fromp);
			    fromp+=cols;
			}
    }
}

__global__ void kernAccSumcol(int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < cols)
    {
			int j;
			const float* fromp = &in[col];
			float* top = &res[col];
			
			(*top) = (*top) *alpha + beta *(*fromp);
			fromp +=cols;
			for (j=rows-1; j!=0; j--)
			{
			    (*top) += beta *(*fromp);
			    fromp+=cols;
			}
    }
}

__global__ void kernAccSumrow(int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < rows)
    {
			int j;
			const float* fromp = &in[row];
			float* top = &res[row];
			
			(*top) = (*top) *alpha + beta *(*fromp);
			fromp +=rows;
			for (j= cols -1; j!=0; j--)
			{
			    (*top) += beta *(*fromp);
			    fromp += rows;
			}
    }
}

__global__ void kernVecMul(int n, float* in_vec1, float* in_vec2, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			out_vec[i] = in_vec1[i] * in_vec2[i];
}

__global__ void kernSubIndex( int rows , int cols, float *in_vec1, const int *in_index, float *res_vec)
{
	 int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(i < rows)
	 {
	 	 int j;
	 	 for(j =0; j< cols;j++)
		 	 	res_vec[cols *i +j] = in_vec1[cols *i +j];
		 int ind = in_index[i];
		 res_vec[cols *i + ind] = in_vec1[cols *i +ind] - 1.0;
	 }
}

__global__ void kernAccSum(int n, int size, float* in, float* res, float beta)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int pos ,k;
	for(k=0; k < size; k++){
		pos = size *i + k;
		if( pos < n)
		{
			res[pos] = in[pos] + beta *res[pos];
		}
	}
}

__global__ void kernGetMaxIndex(int rows, int cols, float* invec, int* outvec)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < cols)
	{
		float *p = invec + rows * i;
		int maxinx = 0;
		float max = *p;
		for(int j=1;j< rows;j++)
		{
			if(p[j] > max)
			{
				max = p[j];
				maxinx = j;
			}
		}
		outvec[i] = maxinx;
	}
}

__global__ void  kernGetLikelyhood(int rows, int cols, float* in_vec, float priscale, float* pri_vec, float* out_vec,float *out_vec2)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row < rows)
    {
			int i;
			const int index = row * cols;
			const float* invec = &in_vec[index];
		  float* outvec = &out_vec[index];
		  float* outvec2 = &out_vec2[index];
			const float* inptr;
			float* outptr;
		
			// First find the max of each vector
			float max;
			
			inptr = invec;
			max = *inptr++;
			for (i=cols-1; i!=0; i--)
			{
			    float val;
		
			    val = *inptr++;
			    if (val>max)
					max = val;
			}
			// Now put exp(in-max) in out
			inptr = invec;
			outptr = outvec;
			float sumexp = 0;
			for (i=cols; i!=0; i--)
			{
			    float f, e;
			    
			    f = *inptr++;
			    e = expf(f - max);
			    sumexp += e;
			}
			
			// Now get the output
			float scale = 1.0f/sumexp;
			inptr = invec;
			outptr = outvec;
			float *outptr2 = outvec2;
			for (i= 0; i< cols; i++)
			{
				if(pri_vec[i] >= 0)
			    outptr[i] = inptr[i] -max - log(sumexp) - priscale* log(pri_vec[i]);
			  else
			    outptr[i] = inptr[i] -max - log(sumexp) - LZEROEMIT;
			  outptr2[i] = expf(inptr[i] -max) *scale;
			}
    }
}

__global__ void kernGetGrad(int n, int size, float *vec1,float * vec2)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int pos ,k;
	for(k=0; k < size; k++){
		pos = size *i + k;
		if( pos < n)
		{
			vec1[pos] =  expf(vec2[pos]) - expf(vec1[pos]);
		}
	}
}

__global__ void kernFSmooth(int frames, int layersize, int size, float *vecout, const int *targ, float *vec_num, float *vec_den, float *vec_res, float beta)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int pos ,k, stateid, label, frameid;
	for(k=0; k < size; k++){
		pos = size *i + k;
		if( pos < frames * layersize)
		{
			frameid = pos/layersize;
			stateid = pos%layersize;
			if(stateid == targ[frameid])
				label = 1.0;
			else
				label = 0;
			vec_res[pos] = (1.0 - beta) *(vecout[pos] -label) + beta *(expf(vec_den[pos])- expf(vec_num[pos]));
		}
	}
}
