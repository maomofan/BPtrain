#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <stdlib.h>
#include <malloc.h>

#include "MyFBLat.h"
#define MAXLINE 	4096
#define BLOCKSIZE 20
#define MAXTRANS 	10000
#define MAXPHONES	10000
#define MAXARCS	2000000
#define MAXGPUARCS 500000
#define MAXDUR 7

////// global varibles
static const int NTHREADS = 256;
static const int CUDA_MAXBLOCKS = 65535;
//static int flag = 0;
static float time_readlat = 0;
static float time_acoustic = 0;
static float time_latfb = 0;
static float time_occacc = 0;

cudaError_t cudaStat;
__device__ int *devnumstates;
__device__ float *devtrans;
__device__ int   *devstateidx;
__device__ int *devarcsttime;
__device__ int *devarcentime;
__device__ int *devarctransid;
__device__ int *devarcphoneid;
__device__ double *devarcStateBeta;
__device__ double *devarcStateAlpha;
__device__ double *devarcStateOcc;
__device__ int *devarcoffset;
//////

#define DEBUG 0
#define OLDCODE 0
#define DEBUGGPU 0

///////// base functions
void check_malloc(cudaError_t cudaStat, const char *name)
{
	if(cudaStat != cudaSuccess){
		printf("%s cudaMalloc Error\n", name);
	}
}
///////////////

//////////////device functions
__device__ double datomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
		__double_as_longlong(val + __longlong_as_double(assumed)));
	}while (assumed != old);
	return __longlong_as_double(old);
}

__device__ LogDouble KernLAdd(LogDouble x, LogDouble y)
{
	double minLogExp = -log(1.0E10);
	LogDouble temp,diff,z;

	if (x<y) {
		temp = x; x = y; y = temp;
	}
	diff = y-x;
	if (diff< minLogExp) 
		return  (x<LSMALL)?LZERO:x;
	else {
		z = exp(diff);
		return x+log(1.0+z);
	}
}

/*
__device__ double atomicLAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
		__double_as_longlong(KernLAdd(val,__longlong_as_double(assumed))));
	}while (assumed != old);
	return __longlong_as_double(old);
}
*/

__device__ float atomicLAdd(float* address, double val)
{
	unsigned int* address_as_ull = (unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
		__float_as_int(KernLAdd(val,__int_as_float(assumed))));
	}while (assumed != old);
	return __int_as_float(old);
}
///////////////

//////////////kernel functions
#if OLDCODE
__global__ void kernSetTrans(int n, int *array_numstate, float *array_value)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n){
  	devtrans[i].numstate = array_numstate[i];
  	devtrans[i].value = array_value + i *MAXSTATE *MAXSTATE;
  }
}

__global__ void kernSetStateidx(int *h_dstateidx)
{
	devstateidx = h_dstateidx;
}

__global__ void kernSetArcs(int n, int *h_dsttime, int *h_dentime, int *h_dtransid, int *h_dphoneid)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n){
  	devarcs[i].sttime 	= h_dsttime[i];
  	devarcs[i].entime 	= h_dentime[i];
  	devarcs[i].numstate = devtrans[h_dtransid[i]].numstate;
  	devarcs[i].transval = devtrans[h_dtransid[i]].value;
  	devarcs[i].stateindex = devstateidx +h_dphoneid[i] *MAXSTATE;
  }
}

__global__ void kernSetOccs(int count, int offset, double **h_dBeta, double **h_dAlpha, double **h_dOcc)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count){
  	int idx = i +offset;
  	devarcs[idx].StateBeta_t = h_dBeta[i];
  	devarcs[idx].StateAlpha_t = h_dAlpha[i];
  	devarcs[idx].stateOcc_t = h_dOcc[i];
  	devarcs[idx].aclike = LZERO;
  	devarcs[idx].Arcocc = LZERO;
  }
}
#endif

__global__ void kernCalcArcStateOcc(int numarc, int offset, Logfloat *emitprob, int statesize, int fea_offset, double *h_daclike)
{
	int threadid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int arcid = threadid + offset;
	if( threadid < numarc){

		int t_st, t_en, t, dur;
		int i,j;
		int transid, phoneid;
		int numstate, *stateidx;
		Logfloat *trans, *curemitprob;
		LogDouble *beta_t, *beta_t1;
		LogDouble *alpha_t, *alpha_t1;
		LogDouble *occ_t;
		
		curemitprob = emitprob +fea_offset;
		t_st 	= devarcsttime[arcid] +1;
		t_en 	= devarcentime[arcid];
		transid = devarctransid[arcid];
		phoneid = devarcphoneid[arcid];
		trans = devtrans + transid *MAXSTATE *MAXSTATE;
		numstate = devnumstates[transid];
		stateidx = devstateidx + phoneid *MAXSTATE;
		dur = t_en - t_st +1;
		beta_t  = devarcStateBeta + devarcoffset[arcid];
		alpha_t = devarcStateAlpha + devarcoffset[arcid];
		occ_t = devarcStateOcc + devarcoffset[arcid];
		
		for(i =0; i < numstate * dur;i++){
			beta_t[i] = LZERO;
			alpha_t[i] = LZERO;
			occ_t[i] = LZERO;
		}

		if(t_st == t_en){      
			for(i =2;i< numstate; i++){
				beta_t[i] = KernLAdd(beta_t[i],trans[(i-1)*numstate + numstate -1]);
				beta_t[i] += curemitprob[(t_en -1) *statesize+ stateidx[i-1]];
				beta_t[0] = KernLAdd(beta_t[0],beta_t[i] + trans[i-1]);
				alpha_t[i] = KernLAdd(alpha_t[i],trans[i-1]);
				alpha_t[i] += curemitprob[(t_st -1) *statesize+ stateidx[i-1]];
				alpha_t[0] = KernLAdd(alpha_t[0],alpha_t[i] + trans[(i-1)*numstate + numstate -1]);
			}
		}
		else{
			beta_t = devarcStateBeta + devarcoffset[arcid] + (t_en -t_st)*numstate;

			for(i=2;i< numstate; i++){
				beta_t[i] = KernLAdd(beta_t[i],trans[(i-1)*numstate + numstate -1]);
				beta_t[i] += curemitprob[(t_en -1) *statesize+ stateidx[i-1]];
				beta_t[0] = KernLAdd(beta_t[0],beta_t[i]);
			}
			
			for(t =t_en -1; t >= t_st;t--){
				beta_t = devarcStateBeta + devarcoffset[arcid] + (t -t_st)*numstate;
				beta_t1 = devarcStateBeta + devarcoffset[arcid] + (t -t_st +1)*numstate;
				for(i=2;i< numstate; i++){
					for(j =2;j< numstate;j++){
						beta_t[i] = KernLAdd(beta_t[i],beta_t1[j] + trans[(i-1)*numstate +j-1]);
					}
					beta_t[i] += curemitprob[(t -1) *statesize+ stateidx[i-1]];
					if(t == t_st)
						beta_t[0] = KernLAdd(beta_t[0],beta_t[i] + trans[i-1]);
					else
						beta_t[0] = KernLAdd(beta_t[0],beta_t[i] );
				}
			}
			alpha_t = devarcStateAlpha + devarcoffset[arcid];
			for(i=2;i< numstate; i++){
				alpha_t[i] = KernLAdd(alpha_t[i],trans[i-1]);
				alpha_t[i] += curemitprob[(t_st -1) *statesize+ stateidx[i-1]];
				alpha_t[0] = KernLAdd(alpha_t[0],alpha_t[i]);
			}
		
			for(t =t_st +1; t <= t_en;t++){
				alpha_t = devarcStateAlpha + devarcoffset[arcid] + (t -t_st) *numstate;
				alpha_t1 = devarcStateAlpha + devarcoffset[arcid] +(t -t_st -1) *numstate;
				for(i=2;i< numstate; i++){
					for(j =2;j< numstate;j++){
						alpha_t[i] = KernLAdd(alpha_t[i],alpha_t1[j] + trans[(j-1)* numstate +i-1]);
					}
					alpha_t[i] += curemitprob[(t -1) *statesize+ stateidx[i-1]];
					if( t== t_en)
						alpha_t[0] = KernLAdd(alpha_t[0],alpha_t[i] + trans[(i-1)* numstate +numstate -1]);
					else
						alpha_t[0] = KernLAdd(alpha_t[0],alpha_t[i]);
				}
			}
		}
		
		h_daclike[arcid] = *(devarcStateAlpha + devarcoffset[arcid] +(t_en -t_st)*numstate);
		for(t =t_st; t <= t_en;t++){
			alpha_t = devarcStateAlpha + devarcoffset[arcid] + (t -t_st)* numstate;
			beta_t  = devarcStateBeta + devarcoffset[arcid] + (t -t_st)* numstate;
			occ_t = devarcStateOcc + devarcoffset[arcid] + (t -t_st)* numstate;
			for(i=2;i< numstate; i++){
				occ_t[i] = alpha_t[i] + beta_t[i] - curemitprob[stateidx[i-1] +(t -1)*statesize] - h_daclike[arcid];
			}
		}
	}
}

__global__ void kernCalcStateOcc(int numarc, int offset, double *arcoccs, float *output, int statesize, int out_offset)
{
	int threadid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int arcid	= threadid +offset;
	if(threadid < numarc){
		int t,j;
		int t_st,t_en, transid, phoneid, numstate, cur_stateidx;
		double arcocc, Arcstateocc;
		float *curoutput;
		int *stateidx;
		float *address;
		
		curoutput = output +out_offset;
		t_st 	= devarcsttime[arcid] +1;
		t_en 	= devarcentime[arcid];
		transid = devarctransid[arcid];
		phoneid = devarcphoneid[arcid];
		numstate = devnumstates[transid];
		arcocc = arcoccs[arcid];
		stateidx = devstateidx + phoneid *MAXSTATE;
		for(j= 2;j< numstate;j++){
			cur_stateidx = stateidx[j-1];
			for(t =t_st;t<= t_en; t++){
				Arcstateocc = *(devarcStateOcc + devarcoffset[arcid] + (t-t_st)*numstate +j);
				address = &(curoutput[(t -1)*statesize +cur_stateidx]);
				atomicLAdd(address, arcocc+ Arcstateocc);
			}
		}
	}
}

/* 
__global__ void kernCalcStateOcc(int numarc, int offset, double *arcoccs, double *output, int statesize, int out_offset, int sign)
{
	int threadid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int arcid	= threadid +offset;
	if(threadid < numarc){
		int t,j;
		int t_st,t_en, transid, phoneid, numstate, cur_stateidx;
		double arcocc, Arcstateocc;
		double *curoutput;
		int *stateidx;
		float *address;
		
		curoutput = output +out_offset;
		t_st 	= devarcsttime[arcid] +1;
		t_en 	= devarcentime[arcid];
		transid = devarctransid[arcid];
		phoneid = devarcphoneid[arcid];
		numstate = devnumstates[transid];
		arcocc = arcoccs[arcid];
		stateidx = devstateidx + phoneid *MAXSTATE;
		for(j= 2;j< numstate;j++){
			cur_stateidx = stateidx[j-1];
			for(t =t_st;t<= t_en; t++){
				Arcstateocc = *(devarcStateOcc + devarcoffset[arcid] + (t-t_st)*numstate +j);
				address = &(curoutput[(t -1)*statesize +cur_stateidx]);
					datomicAdd(address, sign *exp(arcocc+ Arcstateocc));
			}
		}
	}
}
*/
__global__ void kernCalcStateOccAddNum(int numarc, int offset, double latocc, double *arcoccs, float *output, int statesize, int out_offset)
{
	int threadid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int arcid	= threadid +offset;
	if(threadid < numarc){
		int t,j;
		int t_st,t_en, transid, phoneid, numstate, cur_stateidx;
		double arcocc, Arcstateocc;
		int *stateidx;
		float *address;
		float *curoutput;
		
		curoutput = output +out_offset;
		t_st 	= devarcsttime[arcid] +1;
		t_en 	= devarcentime[arcid];
		transid = devarctransid[arcid];
		phoneid = devarcphoneid[arcid];
		numstate = devnumstates[transid];
		arcocc = arcoccs[arcid];
		stateidx = devstateidx + phoneid *MAXSTATE;
		for(j= 2;j< numstate;j++){
			cur_stateidx = stateidx[j-1];
			for(t =t_st;t<= t_en; t++){
				Arcstateocc = *(devarcStateOcc + devarcoffset[arcid] + (t-t_st)*numstate +j);
				address = &(curoutput[(t -1)*statesize +cur_stateidx]);
					atomicLAdd(address, arcocc + latocc + Arcstateocc);
			}
		}
	}
}
/*
__global__ void kernCalcStateOccAddNum(int numarc, int offset, double latocc, double *arcoccs, double *output, int statesize, int out_offset, int sign)
{
	int threadid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int arcid	= threadid +offset;
	if(threadid < numarc){
		int t,j;
		int t_st,t_en, transid, phoneid, numstate, cur_stateidx;
		double arcocc, Arcstateocc;
		int *stateidx;
		float *address;
		double *curoutput;
		
		curoutput = output +out_offset;
		t_st 	= devarcsttime[arcid] +1;
		t_en 	= devarcentime[arcid];
		transid = devarctransid[arcid];
		phoneid = devarcphoneid[arcid];
		numstate = devnumstates[transid];
		arcocc = arcoccs[arcid];
		stateidx = devstateidx + phoneid *MAXSTATE;
		for(j= 2;j< numstate;j++){
			cur_stateidx = stateidx[j-1];
			for(t =t_st;t<= t_en; t++){
				Arcstateocc = *(devarcStateOcc + devarcoffset[arcid] + (t-t_st)*numstate +j);
				address = &(curoutput[(t -1)*statesize +cur_stateidx]);
					datomicAdd(address, sign *exp(arcocc + latocc + Arcstateocc));
			}
		}
	}
}
*/

__global__ void kernMergeNum(int size, float *outputden, float *outputnum, int statesize, int dur, int offset, double numlatocc)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int pos ,k;
	for(k=0; k < size; k++){
		pos = size *i + k;
		if( pos < statesize *dur)
		{
			outputden[offset +pos] = KernLAdd(outputden[offset +pos], numlatocc + outputnum[offset +pos]);
		}
	}
}

inline void DevMergeNum(cudaStream_t stream, float *outputden, float *outputnum, int statesize, int dur, int offset, double numlatocc)
{
		int nblocks = (statesize * dur + NTHREADS-1)/NTHREADS;
    int size = 1;
    while (nblocks > CUDA_MAXBLOCKS)
    {
    	printf("DevMergeNum: nblocks %d too large, we cut half.\n", nblocks);
    	nblocks = (nblocks + 1)/2;
    	size *= 2;
    }
    kernMergeNum<<<nblocks, NTHREADS,0,stream>>>(size, outputden, outputnum, statesize, dur, offset, numlatocc);
}
//////////////

void CMyFBLat::Initial(char *fn_scp, char *fn_state, char *fn_hmm, char *fn_hmmlist, double ac_scale, double lm_scale, double a_kappa, int state_size, int gpu_used)
{
#if DEBUGGPU
	cudaStat = cudaGetLastError();
	if(err != cudaSuccess){
		printf("Assert No Error at begin.\n");
	}
#endif
			
	cudaError_t er = cudaSetDevice(gpu_used);
	if (er!=cudaSuccess)
		printf("cudaSetDevice(%d) failed\n",gpu_used);
	er = cudaStreamCreate(&stream);
	if (er!=cudaSuccess)
		printf("cudaStreamCreate failed\n");
	
	////initial varibles
	h_offset = new int [MAXARCS];
	h_aclike = new double [MAXARCS];
	h_arcocc = new double [MAXARCS];
	h_arcstateocc = new double [MAXARCS *MAXSTATE* MAXDUR];
	h_arcocc_den = new double [MAXARCS];
	h_arcstateocc_den = new double [MAXARCS *MAXSTATE* MAXDUR];
	cudaStat = cudaMalloc(&h_dsttime, sizeof(int) *MAXARCS);
	check_malloc(cudaStat, "h_dsttime");
	cudaStat = cudaMalloc(&h_dentime, sizeof(int) *MAXARCS);
	check_malloc(cudaStat, "h_dentime");
	cudaStat = cudaMalloc(&h_dtransid, sizeof(int) *MAXARCS);
	check_malloc(cudaStat, "h_dtransid");
	cudaStat = cudaMalloc(&h_dphoneid, sizeof(int) *MAXARCS);
	check_malloc(cudaStat, "h_dphoneid");
	cudaStat = cudaMalloc(&h_doffset,sizeof(int)* MAXARCS);
	check_malloc(cudaStat, "h_doffset");
	cudaStat = cudaMalloc(&h_dBeta,sizeof(double)* MAXGPUARCS *MAXSTATE* MAXDUR);
	check_malloc(cudaStat, "h_dBeta");
	cudaStat = cudaMalloc(&h_dAlpha,sizeof(double)* MAXGPUARCS*MAXSTATE* MAXDUR);
	check_malloc(cudaStat, "h_dAlpha");
	cudaStat = cudaMalloc(&h_dOcc,sizeof(double)* MAXGPUARCS*MAXSTATE* MAXDUR);
	check_malloc(cudaStat, "h_dOcc");
	cudaStat = cudaMalloc(&h_daclike, sizeof(double) *MAXARCS);
	check_malloc(cudaStat, "h_daclike");
	cudaStat = cudaMalloc(&h_darcocc, sizeof(double) *MAXARCS);
	check_malloc(cudaStat, "h_darcocc");
	cudaMemcpyToSymbol(devarcsttime, &h_dsttime, sizeof(h_dsttime));
	cudaMemcpyToSymbol(devarcentime, &h_dentime, sizeof(h_dentime));
	cudaMemcpyToSymbol(devarctransid, &h_dtransid, sizeof(h_dtransid));
	cudaMemcpyToSymbol(devarcphoneid, &h_dphoneid, sizeof(h_dphoneid));
	cudaMemcpyToSymbol(devarcStateBeta, &h_dBeta, sizeof(h_dBeta));
	cudaMemcpyToSymbol(devarcStateAlpha, &h_dAlpha, sizeof(h_dAlpha));
	cudaMemcpyToSymbol(devarcStateOcc, &h_dOcc, sizeof(h_dOcc));
	cudaMemcpyToSymbol(devarcoffset, &h_doffset, sizeof(h_doffset));
	
	LoadLatlist(fn_scp);
	LoadStateindex(fn_state);
	LoadHMM(fn_hmm);
	LoadHmmlist(fn_hmmlist);
	acscale = ac_scale;
	lmscale = lm_scale;
	kappa   = a_kappa;
	statesize = state_size;
	total_Pr_den  = 0;
	total_Pr_num  = 0;
	total_T		= 0;
	
	for(int i=0; i< LATCACHE; i++){
		 numlatstat[i] = 0;
		 denlatstat[i] = 0;
	}
	numlatposforwrite = 0;
	denlatposforwrite = 0;
	numlatposforread = 0;
	denlatposforread = 0;
	cudaStreamSynchronize(stream);
}

CMyFBLat::CMyFBLat()
{
	
}

CMyFBLat::~CMyFBLat()
{
	map<string,Triphone*> ::iterator it1;
	map<string,Transform*> ::iterator it2;
	
	for(it1 = triphonehash.begin();it1 != triphonehash.end(); it1++){
		delete [](it1->second->stateindex);
		delete it1->second;
	}
	for(it2 = transhash.begin();it2 != transhash.end(); it2++){
		delete [](it2->second->value);
		delete it2->second;
	}

	stateindex.clear();
	hmmlisthash.clear();
	triphonehash.clear();
	transhash.clear();
	Lat_den.clear();
	Lat_num.clear();
	
	delete []h_offset;
	delete []h_aclike;
	delete []h_arcocc;
	delete []h_arcocc_den;
	delete []h_arcstateocc;
	delete []h_arcstateocc_den;
	cudaFree(h_dsttime);
	cudaFree(h_dentime);
	cudaFree(h_dtransid);
	cudaFree(h_dphoneid);
	cudaFree(h_doffset);
	cudaFree(h_dBeta);
	cudaFree(h_dAlpha);
	cudaFree(h_dOcc);
	cudaFree(h_daclike);
	cudaFree(h_darcocc);
	cudaStreamDestroy(stream);
}

///Need Unix format
void CMyFBLat::LoadLatlist(char *fn_scp)
{
	FILE *fp;
	char buf[MAXLINE];
	char *p, *pp;
	string name;
	int init_size;
	int count =0;

	init_size = 2;
	Lat_den.resize(init_size);
	Lat_num.resize(init_size);
	fp = fopen(fn_scp,"r");
	if(fp ==NULL){
		printf("Error: Cannot open %s for read.",fn_scp);
		exit(0);
	}
	while(fgets(buf,MAXLINE,fp)){
		buf[strlen(buf) -1] ='\0';
		if(NULL != (p =strstr(buf,"="))){
			pp = p +1;
		}else
			pp =buf;
		p = strstr(pp,"\t");
		*p = '\0';
		if(count == Lat_den.capacity()){
			init_size *= 2;
			Lat_den.resize(init_size);
			Lat_num.resize(init_size);
		}
		Lat_den[count] = p +1;
		Lat_num[count] = pp;
		count ++;
	}

	fclose(fp);
	printf("Load Lattice list over.\n");
	fflush(stdout);
}

////Need Unix format 
void CMyFBLat::LoadHmmlist(char *fn_hmmlist)
{
	FILE *fp;
	char buf[MAXLINE];
	char *p;
	string phyname,logname;

	fp = fopen(fn_hmmlist,"r");
	if(fp ==NULL){
		printf("Error: Cannot open %s for read.",fn_hmmlist);
		exit(0);
	}
	while(fgets(buf,MAXLINE,fp)){
		buf[strlen(buf) -1] ='\0';
		if(NULL != (p =strstr(buf," "))){
			*p = '\0';
			logname = buf;
			phyname = p+1;
			hmmlisthash.insert(pair<string,string>(logname,phyname));
		}else{
			hmmlisthash.insert(pair<string,string>(buf,buf));
		}
	}
	fclose(fp);
	printf("Load Hmmlist over.\n");
	fflush(stdout);
}

///Need Unix format
void CMyFBLat::LoadStateindex(char *fn_state)
{
	FILE *fp;
	char buf[MAXLINE];
	string name;
	int count =0;

	fp = fopen(fn_state,"r");
	if(fp ==NULL){
		printf("Error: Cannot open %s for read.",fn_state);
		exit(0);
	}
	while(fgets(buf,MAXLINE,fp)){
		buf[strlen(buf)-1] ='\0';
		name = buf;
		stateindex.insert(pair<string,int>(name,count));
		count ++;
	}
	fclose(fp);
	printf("Load State index over.\n");
	fflush(stdout);
}

////Need TXT format model 
void CMyFBLat::LoadHMM(char *fn_model)
{
	FILE *fp;
	char buf[MAXLINE],buf2[MAXLINE];
	char *p, *pp;
	int numstate;
	string name;
	int i,j,count,count2;
	Transform *trans;
	Triphone *phone;
	float *val;
  int idx_trans;
  int idx_phones;
  int size;
  int size_stateidx;
  int *array_numstate;
	int *array_stateidx;
	
	fp = fopen(fn_model,"r");
	if(fp ==NULL){
		printf("Error: Cannot open %s for read.",fn_model);
		exit(0);
	}
	idx_trans = 0;
	idx_phones = 0;
  val = new float [MAXTRANS *MAXSTATE *MAXSTATE];
  array_numstate = new int[MAXTRANS];
  array_stateidx = new int[MAXSTATE *MAXPHONES];
  size = MAXTRANS;
  size_stateidx  = MAXPHONES;
  
	while(fgets(buf,MAXLINE,fp)){
		if(NULL != (p =strstr(buf,"~t"))){
			fgets(buf2,MAXLINE,fp);
			if(NULL != (p=strstr(buf2,"TRANSP"))){
				p = strstr(buf2," ");
				numstate = atoi(p+1);
				p = strstr(buf,"\"");
				pp = strstr(p+1,"\"");
				*pp ='\0';
				name= p+1;
				trans = new Transform;
				trans->idx = idx_trans;
				trans->name = name;
				trans->numstate = numstate;
				idx_trans ++;
				if(idx_trans > size){
					float *tmp_val = val;
					int *tmp_array_numstate = array_numstate;
					int tmp_size = size;
					size += MAXTRANS;
					val = new float[size *MAXSTATE *MAXSTATE];
					array_numstate = new int[size];
					memcpy(array_numstate,tmp_array_numstate,sizeof(int)*tmp_size);
					memcpy(val,tmp_val,sizeof(float)*tmp_size*MAXSTATE *MAXSTATE);
					delete []tmp_val;
					delete []tmp_array_numstate;
				}
				array_numstate[idx_trans -1] = numstate;
				float *cur_val = val + (idx_trans -1)* MAXSTATE *MAXSTATE;
				
				for(i =0,j =0 ;i< numstate;i++){
					fgets(buf2,MAXLINE,fp);
					p = buf2 +1;
					while(NULL != (pp =strstr(p," "))){
						*pp = '\0';
						cur_val[j] =  Logf(atof(p));
						j++;
						p = pp+1;
					}
					cur_val[j] = Logf(atof(p));
					j++;
				}
				transhash.insert(pair<string,Transform*>(name,trans));
			}
		}
		else if(NULL != (p =strstr(buf,"~h"))){
			count = 0;
			count2 =0;
			p = strstr(buf,"\"");
			pp = strstr(p+1,"\"");
			*pp ='\0';
			phone = new Triphone;
			phone->idx  = idx_phones;
			phone->name = p+1;
			idx_phones ++;
			if(idx_phones > size_stateidx){
				int *tmp_array_stateidx = array_stateidx;
				int tmp_size = size_stateidx;
				size_stateidx += MAXPHONES;
				array_stateidx = new int[size_stateidx *MAXSTATE];
				memcpy(array_stateidx,tmp_array_stateidx,sizeof(int)*tmp_size *MAXSTATE);
				delete []tmp_array_stateidx;
			}
			int *cur_array_stateidx = array_stateidx + (idx_phones -1)* MAXSTATE;
			
			while(fgets(buf,MAXLINE,fp)){
				if(NULL != (p=strstr(buf,"NUMSTATES"))){
					p = strstr(buf," ");
					phone->numstate = atoi(p+1);
					continue;
				}
				if(NULL != (p=strstr(buf,"~t"))){
					count++;
					p = strstr(buf,"\"");
					pp = strstr(p+1,"\"");
					*pp ='\0';
					name = p+1;
					trans = transhash.find(name)->second;
					phone->trans = trans;
					break;
				}
				if(NULL != (p=strstr(buf,"~s"))){
					count2 ++;
					p = strstr(buf,"\"");
					pp = strstr(p+1,"\"");
					*pp ='\0';
					name = p+1;
					cur_array_stateidx[count2] = stateindex.find(name)->second;
					continue;
				}
			}
			if(count != 1){
				printf("Error: phone %s has no trans.",name.c_str());
				exit(0);
			}
			if(count2 != phone->numstate -2){
				if(phone->name != "sil" && phone->name != "sp"){
					printf("Error: phone %s has no trans.",name.c_str());
					exit(0);
				}else{
					for(i=1;i< phone->numstate -1;i++){
						char curs[10];
						sprintf(curs,"%d",i+1);
						name = phone->name + "_s" + curs;
						cur_array_stateidx[i] = stateindex.find(name)->second;
					}
				}
			}
			triphonehash.insert(pair<string,Triphone*>(phone->name,phone));
		}
	}
	fclose(fp);
	
	////Move trans to device
	int *h_dnumstates;
	float *h_dval;
	cudaStat = cudaMalloc(&h_dnumstates, sizeof(int) *idx_trans);
	check_malloc(cudaStat, "h_dnumstates");
	cudaStat = cudaMalloc(&h_dval, sizeof(float) *idx_trans *MAXSTATE *MAXSTATE);
	check_malloc(cudaStat, "h_dval");
	cudaMemcpy(h_dval, val, sizeof(float) *idx_trans *MAXSTATE *MAXSTATE, cudaMemcpyHostToDevice);
	cudaMemcpy(h_dnumstates, array_numstate, sizeof(int) *idx_trans, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(devtrans, &h_dval, sizeof(h_dval));
	cudaMemcpyToSymbol(devnumstates, &h_dnumstates, sizeof(h_dnumstates));
	
	////Move phones to device
	int *h_dstateidx;
	cudaStat = cudaMalloc(&h_dstateidx, sizeof(int) *idx_phones *MAXSTATE);
	check_malloc(cudaStat, "h_dstateidx");
	cudaMemcpy(h_dstateidx, array_stateidx, sizeof(int) *idx_phones *MAXSTATE, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(devstateidx, &h_dstateidx, sizeof(h_dstateidx));
	
#if DEBUG
/*
	int  cur_idx_phones = 14277;
	int *tmpvalue2 = new int [cur_idx_phones *MAXSTATE];
	int *dtmpvalue2;
	cudaStat = cudaMemcpyFromSymbol(&dtmpvalue2 ,devstateidx, sizeof(dtmpvalue2));
	check_malloc(cudaStat, "dtmpvalue22");
	cudaMemcpy(tmpvalue2, h_dstateidx, sizeof(int) *idx_phones *MAXSTATE, cudaMemcpyDeviceToHost);
	for(int k =0;k < cur_idx_phones; k++){
		for(int m=0; m< MAXSTATE; m++)
			printf("%d\t",tmpvalue2[k *MAXSTATE + m]);
		printf("\n");
	}
	printf("total phones num: %d\n", cur_idx_phones);
	delete []tmpvalue2;
	exit(0);
*/
#endif
	
	delete []val;
	delete []array_numstate;
	delete []array_stateidx;

	printf("Load HMM over.\n");
	fflush(stdout);
}

/*
////Lattice must be DOS format
void CMyFBLat::ReadLat(char *fn_scp, Lattice *lat)
{
	//struct timeval timest,timeen;
	int i;
	FILE *fp_scp;
	char buf[MAXLINE];
	string phonename;
	char *p, *pp;
	
//gettimeofday(&timest,NULL);
	fp_scp = fopen(fn_scp,"r");
	if(fp_scp ==NULL){
		printf("Error: Cannot open %s for read.",fn_scp);
		exit(0);
	}
	while(fgets(buf,MAXLINE,fp_scp)){
		p =strstr(buf,"N=");
		pp =strstr(buf,"L=");
		if( p && pp){
			*(pp -1) = '\0';
			lat->numarc = atoi(pp+2);
			lat->numnode = atoi(p+2);
			
			lat->nodes = new Node [lat->numnode];
			lat->arcs  = new Arc [lat->numarc];
			
			if(MAXARCS < lat->numarc){
				printf("Error: latnum %d > %d.\n",lat->numarc, MAXARCS);
				exit(0);
			}
			int *prearcsize = new int[lat->numnode];
			int *folarcsize = new int[lat->numnode];

			for(i=0;i< lat->numnode;i++){
				lat->nodes[i].prearcnum =0;
				lat->nodes[i].folarcnum =0;
				lat->nodes[i].prearcs = (int *)malloc(BLOCKSIZE*sizeof(int));
				lat->nodes[i].folarcs = (int *)malloc(BLOCKSIZE*sizeof(int));
				prearcsize[i] = BLOCKSIZE;
				folarcsize[i] = BLOCKSIZE;
			}
			
			for(i=0;i< lat->numnode;i++){
				fgets(buf,MAXLINE,fp_scp);
				p = strstr(buf,"I=");
				pp = strstr(buf, " ");
				*pp = '\0';
				lat->nodes[i].idx = atoi(p+2);
				p = strstr(pp+1,"t=");
				pp = strstr(pp+1, "\t");
				*pp = '\0';
				lat->nodes[i].time = (int) (100 *atof(p+2) +0.5);
			}
			for(i=0;i< lat->numarc;i++){
				fgets(buf,MAXLINE,fp_scp);
				p = strstr(buf,"J=");
				pp = strstr(buf, "\t");
				*pp = '\0';
				lat->arcs[i].idx = atoi(p+2);
				p = strstr(pp+1,"S=");
				pp = strstr(pp+1, "\t");
				*pp = '\0';
				lat->arcs[i].stnode = &(lat->nodes[atoi(p+2)]);
				
				Node *curnode = lat->arcs[i].stnode;
				if(curnode->folarcnum == folarcsize[curnode->idx]){
					folarcsize[curnode->idx] *= 2;
					curnode->folarcs = (int*)realloc(curnode->folarcs,sizeof(int)*folarcsize[curnode->idx]);
				}
				curnode->folarcs[curnode->folarcnum++] = i;
				
				p = strstr(pp+1,"E=");
				pp = strstr(pp+1, "\t");
				*pp = '\0';
				lat->arcs[i].ennode = &(lat->nodes[atoi(p+2)]);
				
				arr_sttime[i] = lat->arcs[i].stnode->time;
				arr_entime[i] = lat->arcs[i].ennode->time;
				
				curnode = lat->arcs[i].ennode;
				if(curnode->prearcnum == prearcsize[curnode->idx]){
					prearcsize[curnode->idx] *= 2;
					curnode->prearcs = (int*)realloc(curnode->prearcs,sizeof(int)*prearcsize[curnode->idx]);
				}
				curnode->prearcs[curnode->prearcnum++] = i;
				
				p = strstr(pp+1,"l=");
				pp = strstr(pp+1, "\t");
				*pp = '\0';
				lat->arcs[i].lmlike = atof(p+2);
				p = strstr(pp+1,"d=");
				pp = strstr(pp+1, ",");
				*pp = '\0';
				phonename = p+3;
				phonename = hmmlisthash.find(phonename)->second;
				p = pp+1;
				pp = strstr(pp+1, ":");
				*pp = '\0';
				lat->arcs[i].duration = (int) (100 *atof(p) +0.5);
				lat->arcs[i].phone = triphonehash.find(phonename)->second;

				arr_phoneid[i] = lat->arcs[i].phone->idx;
				arr_transid[i] = lat->arcs[i].phone->trans->idx;	
			}
			free(prearcsize);
			free(folarcsize);
		}
	}
	lat->startarc = new Arc;
	lat->startarc->ennode = lat->nodes;
	fclose(fp_scp);
	
	////copy arcs to device
	int gpuarcnum = lat->numarc;
	cudaMemcpy(h_dsttime, arr_sttime, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
	cudaMemcpy(h_dentime, arr_entime, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
	cudaMemcpy(h_dtransid, arr_transid, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
	cudaMemcpy(h_dphoneid, arr_phoneid, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);

#if DEBUG
	int idx_trans = 45;
	float *tmpvalue = new float [idx_trans *MAXSTATE *MAXSTATE];
	float *dtmpvalue;
	int *tmpnumstate0 = new int [idx_trans];
	int *dtmpnumstate0;
	cudaMemcpyFromSymbol(&dtmpvalue ,devtrans, sizeof(dtmpvalue));
	cudaMemcpyFromSymbol(&dtmpnumstate0 ,devnumstates, sizeof(dtmpnumstate0));
	cudaMemcpy(tmpvalue, dtmpvalue,sizeof(float) *idx_trans *MAXSTATE *MAXSTATE, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpnumstate0, dtmpnumstate0, sizeof(int)*idx_trans, cudaMemcpyDeviceToHost);
	for(int l =0;l < idx_trans; l++){
		printf("numstate: %d\n", tmpnumstate0[l]);
		for(int k =0;k < MAXSTATE; k++){
			for(int m =0; m< MAXSTATE; m++)
				printf("%.6e ",expf(tmpvalue[m + k*MAXSTATE + l*MAXSTATE *MAXSTATE]));
			printf("\n");
		}
	}
	printf("total trans num: %d\n", idx_trans);
	delete []tmpvalue;
	delete []tmpnumstate0;
	exit(0);
#endif

#if DEBUG
	int cur_arcs = 211;
	int *dtmptransid;
	int *dtmpphoneid;
	int *dtmpsttime;
	int *dtmpentime;
	int *tmpsttime   = new int [cur_arcs];
	int *tmpentime   = new int [cur_arcs];
	int *tmptransid = new int [cur_arcs];
	int *tmpphoneid = new int [cur_arcs];
	
	cudaMemcpyFromSymbol(&dtmptransid ,devarctransid, sizeof(dtmptransid));
	cudaMemcpyFromSymbol(&dtmpphoneid ,devarcphoneid, sizeof(dtmpphoneid));
	cudaMemcpyFromSymbol(&dtmpsttime ,devarcsttime, sizeof(dtmpsttime));
	cudaMemcpyFromSymbol(&dtmpentime ,devarcentime, sizeof(dtmpentime));

	cudaMemcpy(tmptransid, dtmptransid, sizeof(int)*cur_arcs , cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpphoneid, dtmpphoneid, sizeof(int)*cur_arcs , cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpsttime, dtmpsttime, sizeof(int)*cur_arcs , cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpentime, dtmpentime,sizeof(int)*cur_arcs , cudaMemcpyDeviceToHost);

	for(int arcid = 0; arcid < 211;arcid++){
		printf("arcid: %d\n", arcid);
		printf("arc sttime: %d\n", tmpsttime[arcid]);
		printf("arc entime: %d\n", tmpentime[arcid]);
		printf("arc transid: %d\n", tmptransid[arcid]);
		printf("arc phoneid: %d\n", tmpphoneid[arcid]);
	}
	delete []tmpphoneid;
	delete []tmptransid;
	delete []tmpsttime;
	delete []tmpentime;
	exit(0);
#endif
}
*/

int CMyFBLat::LatInLat(Lattice *latnum,Lattice* latden)
{
	Arc *arc1 = latnum->startarc;
	Arc *arc2 = latden->startarc;
	
	for(int i =0; i< latnum->numarc; i++){
		(latnum->arcs +i)->latinlatTag =0;
	}
	latnum->startarc->latinlatTag =0;
	
	return LatInLatPath(latnum, latden, arc1, arc2);
}

int CMyFBLat::LatInLatPath(Lattice *latnum, Lattice *latden, Arc *arc1, Arc *arc2)
{
	int i,j;
	if(arc1->ennode->folarcnum == 0 && arc2->ennode->folarcnum ==0)
		return 1;
	if(arc1->latinlatTag == 1)
		return 1;
		
	if(arc1->ennode->folarcnum == 0 || arc2->ennode->folarcnum ==0)
		return 0;
	else{
		for (i =0; i< arc1->ennode->folarcnum;i++){
			int Arcfind = 0;
			Arc *arcnum =  latnum->arcs + arc1->ennode->folarcs[i];
			for(j =0;j< arc2->ennode->folarcnum;j++){
				Arc *arcden = latden->arcs + arc2->ennode->folarcs[j];
				if(arcnum->stnode->time == arcden->stnode->time && arcnum->ennode->time == arcden->ennode->time
					&& arcnum->phone->name == arcden->phone->name){
						if(LatInLatPath(latnum,latden,arcnum,arcden)){
							Arcfind =1;
							arcnum->latinlatTag =1;
						}
				}
			}
			if(Arcfind != 1)
				return 0;
		}
		return 1;
	}
}

void CMyFBLat::InitArc(Lattice *lat, int count)
{
	for(int i =0;i< count;i++){
		Arc *arc = lat->arcs +i;
		arc->aclike = LZERO;
		arc->Arcocc = LZERO;
		arc->Arcalpha = LZERO;
		arc->ArcBeta = LZERO;
	}
}

int CMyFBLat::InitArcgpu(Lattice *lat, int offset)
{
	int dur;
	int count;
	int numstate;
	int i;
	
	count =0;
	sum_occnum = 0;
	for(i =offset ;i < lat->numarc; i++){
		Arc *arc = lat->arcs +i;
		dur = arc->duration;
		numstate = arc->phone->numstate;
		if(i -offset +1 > MAXGPUARCS){
			break;
		}
		if(sum_occnum +numstate* dur> MAXSTATE *MAXDUR *MAXGPUARCS ){
			break;
		}
		count++;
		h_offset[i] = sum_occnum;
		sum_occnum += numstate* dur;
	}
	cudaMemcpy(h_doffset +offset, h_offset +offset, sizeof(int) *count, cudaMemcpyHostToDevice);
	return count;
}

void CMyFBLat::FreeLat(Lattice *lat)
{
	int i,numnode;
	numnode = lat->numnode;

	for(i=0; i< numnode; i++){
		free(lat->nodes[i].prearcs);
		free(lat->nodes[i].folarcs);
	}
	delete [](lat->arr_sttime);
	delete [](lat->arr_entime);
	delete [](lat->arr_transid);
	delete [](lat->arr_phoneid);
	delete [](lat->arcs);
	delete [](lat->nodes);
}

LogDouble CMyFBLat::CalcArcFB(Lattice *lat, Logfloat *emitprob, float *output, int offset)
{
	int arcnum;
	LogDouble pr;
	int n_blocks;
	int cur_count;
	int counts, counts_data;
	int partid;
	
	arcnum = lat->numarc;
	InitArc(lat, arcnum);
	partid = 0;
	counts = 0;
	counts_data = 0;
	while(counts != arcnum){
		cur_count = InitArcgpu(lat, counts);
		part_datacount[partid] = sum_occnum;
		part_arccount[partid] = cur_count;
		part_arcoffset[partid] = counts;
		part_dataoffset[partid] = counts_data;
		partid ++;
		n_blocks = (cur_count + NTHREADS-1)/NTHREADS;
		
		//計算弧的發射概率
		kernCalcArcStateOcc<<<n_blocks, NTHREADS, 0, stream>>>(cur_count, counts, emitprob, statesize, offset, h_daclike);
		cudaMemcpy(h_arcstateocc + counts_data, h_dOcc, sizeof(double)*sum_occnum, cudaMemcpyDeviceToHost);
		counts += cur_count;
		counts_data += sum_occnum;
	}
	part_arcoffset[partid] = counts;
	cudaMemcpy(h_aclike ,h_daclike, sizeof(double) *arcnum, cudaMemcpyDeviceToHost);
#if DEBUG
/*
	int cur_arc_num = 211;
	printf("arcs: %d\n", cur_arc_num);
	for(int k=0;k< cur_arc_num; k++){
			printf("%.6e\n",h_aclike[k]);
	}
	exit(0);
*/
#endif
	pr = CalcArcOcc(lat, h_aclike, h_arcocc);  //計算弧的佔有率
#if DEBUG
/*
	int cur_arc_num = 211;
	printf("arcs: %d\n", cur_arc_num);
	for(int k=0;k< cur_arc_num; k++){
			printf("%.6e\n",lat->arcs[k].Arcocc);
	}
	exit(0);
*/
#endif
	arcnum = lat->numarc;
	cudaMemcpy(h_darcocc ,h_arcocc, sizeof(double) *arcnum, cudaMemcpyHostToDevice);
	counts = 0;
	partid = 0;
	cur_count = 0;
	while(counts + cur_count!= arcnum){
		sum_occnum = part_datacount[partid];
		cur_count = part_arccount[partid];
		counts = part_arcoffset[partid];
		counts_data = part_dataoffset[partid];
		partid ++;
		cudaMemcpy(h_dOcc, h_arcstateocc + counts_data, sizeof(double)*sum_occnum, cudaMemcpyHostToDevice);
		n_blocks = (cur_count + NTHREADS-1)/NTHREADS;

		kernCalcStateOcc<<<n_blocks, NTHREADS, 0, stream>>>(cur_count, counts, h_darcocc, output,statesize, offset);
	}
#if DEBUG
/*
	int tmpsize = statesize* lat->nodes[lat->numnode -1].time;
	double *tmpocc = new double [tmpsize];
	cudaMemcpy(tmpocc, output, sizeof(double) *tmpsize, cudaMemcpyDeviceToHost);
	printf("occs:\n");
	for(int k=0;k< tmpsize; k++){
			printf("%.6e\n",tmpocc[k]);
	}
	delete []tmpocc;
	exit(0);
*/
#endif
	return pr;
}

LogDouble CMyFBLat::CalcArcFBAddNum(Lattice *latden, Lattice *latnum, double pr_num, Logfloat *emitprob, float *outputden, float *outputnum, int offset)
{
	struct timeval timest,timeen;
	float curtime;
	int arcnum;
	LogDouble pr;
	int n_blocks;
	int cur_count;
	int counts, counts_data;
	int partid;
	int dur;
	double denlatocc, numlatocc;
	
		//gettimeofday(&timest,NULL);
	arcnum = latden->numarc;
	InitArc(latden, arcnum);
	partid = 0;
	counts = 0;
	counts_data = 0;
	while(counts != arcnum){
		cur_count = InitArcgpu(latden, counts);
		part_datacount_den[partid] = sum_occnum;
		part_arccount_den[partid] = cur_count;
		part_arcoffset_den[partid] = counts;
		part_dataoffset_den[partid] = counts_data;
		partid ++;
		n_blocks = (cur_count + NTHREADS-1)/NTHREADS;
		kernCalcArcStateOcc<<<n_blocks, NTHREADS, 0, stream>>>(cur_count, counts, emitprob, statesize, offset, h_daclike);
		cudaMemcpy(h_arcstateocc_den + counts_data, h_dOcc, sizeof(double)*sum_occnum, cudaMemcpyDeviceToHost);
		counts += cur_count;
		counts_data += sum_occnum;
	}
	part_arcoffset_den[partid] = counts;
	cudaMemcpy(h_aclike ,h_daclike, sizeof(double) *arcnum, cudaMemcpyDeviceToHost);
		//gettimeofday(&timeen,NULL);
		//curtime = 1000.0* (timeen.tv_sec - timest.tv_sec) + (timeen.tv_usec - timest.tv_usec)/1000.0;
		//time_acoustic += curtime;
		//printf("Acoustic time: %.2f ms\n", curtime);
#if 0
if(flag == 1){
	int cur_arc_num = 31231;
	printf("arcs: %d\n", cur_arc_num);
	for(int k=0;k< cur_arc_num; k++){
			printf("%.6e\n",h_aclike[k]);
	}
	exit(0);
}
#endif
		//gettimeofday(&timest,NULL);
	pr = CalcArcOcc(latden, h_aclike, h_arcocc_den);
	numlatocc = pr_num - LAdd(pr_num,pr);
	denlatocc = pr - LAdd(pr_num,pr);
		//gettimeofday(&timeen,NULL);
		//curtime = 1000.0* (timeen.tv_sec - timest.tv_sec) + (timeen.tv_usec - timest.tv_usec)/1000.0;
		//time_latfb += curtime;
		//printf("ArcOcc calc time: %.2f ms\n", curtime);
#if 0
if(flag == 1){
	int cur_arc_num = latnum->numarc;
	printf("arcs: %d\n", cur_arc_num);
	for(int k=0;k< cur_arc_num; k++){
			printf("%.10e\n",h_arcocc[k]);
	}
	exit(0);
}
#endif
		//gettimeofday(&timest,NULL);
	arcnum = latden->numarc;
	cudaMemcpy(h_darcocc ,h_arcocc_den, sizeof(double) *arcnum, cudaMemcpyHostToDevice);
	counts = 0;
	cur_count =0;
	partid = 0;
	while(counts + cur_count != arcnum){
		sum_occnum = part_datacount_den[partid];
		cur_count = part_arccount_den[partid];
		counts = part_arcoffset_den[partid];
		counts_data = part_dataoffset_den[partid];
		partid ++;
		cudaMemcpy(h_dOcc, h_arcstateocc_den + counts_data, sizeof(double)*sum_occnum, cudaMemcpyHostToDevice);
		n_blocks = (cur_count + NTHREADS-1)/NTHREADS;
		kernCalcStateOccAddNum<<<n_blocks, NTHREADS, 0, stream>>>(cur_count, counts, denlatocc, h_darcocc, outputden,statesize, offset);
	}

	dur = latden->nodes[latden->numnode -1].time;
	DevMergeNum(stream, outputden, outputnum, statesize, dur, offset, numlatocc);
		//cudaStreamSynchronize(stream);
		//gettimeofday(&timeen,NULL);
		//curtime = 1000.0* (timeen.tv_sec - timest.tv_sec) + (timeen.tv_usec - timest.tv_usec)/1000.0;
		//time_occacc += curtime;
		//printf("Occ accum time: %.2f ms\n", curtime);
#if 0
if(flag == 1){
	int tmpsize = statesize* latnum->nodes[latnum->numnode -1].time;
	float *tmpocc = new float [tmpsize];
	cudaStat = cudaMemcpy(tmpocc, outputden, sizeof(float) *tmpsize, cudaMemcpyDeviceToHost);
	if(cudaStat != cudaSuccess){
		printf("fsafsafsa\n");
	}
	printf("tmpsize: %d\n",tmpsize);
	for(int k=0;k< tmpsize; k++){
			printf("%.10e\n",exp(tmpocc[k +offset]));
	}
	delete []tmpocc;
	exit(0);
}
#endif
	return LAdd(pr_num,pr);
}

LogDouble CMyFBLat::CalcArcOcc(Lattice *lat ,LogDouble *aclikes, LogDouble *arcoccs)
{
	int i,j;
	int numarc, numprearc,numfolarc;
	Arc *arc, *prearc, *folarc;
	LogDouble alpha, beta, alphaAll;
	std::vector<double> alpha_smbr(numarc, 0), beta_smbr(numarc, 0);//forward and backward variable for sMBR
	double tot_forward_score = 0;


	numarc = lat->numarc;
	alphaAll = LZERO;

	//alpha_first
	for(i=0;i < numarc;i++){
		arc = lat->arcs +i;
		numprearc = arc->stnode->prearcnum;
		numfolarc = arc->ennode->folarcnum;
		if( numprearc==0)
			alpha = 0;
		else{
			alpha = LZERO;
			for(j =0;j< numprearc;j++){
				prearc = lat->arcs + arc->stnode->prearcs[j];
				alpha = LAdd(alpha, prearc->Arcalpha);
			}
		}
		alpha += (aclikes[i] * acscale + arc->lmlike *lmscale) *kappa;
		arc->Arcalpha = alpha;
		if(numfolarc == 0)
			alphaAll = LAdd(alphaAll, alpha);
	}
	
	//beta_first
	for(i= numarc -1;i >=0;i--){
		arc = lat->arcs +i;
		numprearc = arc->stnode->prearcnum;
		numfolarc = arc->ennode->folarcnum;
		if( numfolarc==0)
			beta = 0;
		else{
			beta = LZERO;
			for(j = numfolarc -1;j >= 0;j--){
				folarc = lat->arcs + arc->ennode->folarcs[j];
				beta = LAdd(beta, folarc->ArcBeta);
			}
		}
		beta += (aclikes[i] * acscale +arc->lmlike *lmscale) *kappa;
		arc->ArcBeta = beta;
		arcoccs[i] = arc->ArcBeta + arc->Arcalpha - (aclikes[i] * acscale +arc->lmlike *lmscale) *kappa -alphaAll;
	}
	return alphaAll;

	//alpha_second
	for(i=0;i < numarc;i++){
		arc = lat->arcs +i;
		numprearc = arc->stnode->prearcnum;
		numfolarc = arc->ennode->folarcnum;

		double frame_acc = 0.0;
		if( numprearc==0)
			alpha = 0;
		else{
			alpha = LZERO;
			for(j =0;j< numprearc;j++){
					prearc = lat->arcs + arc->stnode->prearcs[j];
					alpha = LAdd(alpha, prearc->Arcalpha);
					frame_acc = ( == ) ? 1.0 : 0.0;
				}
			}
		alpha += (aclikes[i] * acscale + arc->lmlike *lmscale) *kappa;
		arc->Arcalpha = alpha;
		if(numfolarc == 0)
			alphaAll = LAdd(alphaAll, alpha);
	}

}

void CMyFBLat::DoLatFB(int sentnum, int *sentindex, Logfloat *emitprob, float *outputnum, float *outputden)
{
	struct timeval timest,timeen;
	float curtime;
	int i, sentidx, offset, dur;
	char latname[MAXLINE];
	Lattice *latnum, *latden;
	LogDouble pr;
	int lat_in_lat;
	
	offset = 0;
	for(i =0; i< sentnum ;i++){
		sentidx = sentindex[i];
		printf("Processing Sent: %d\n",sentidx);
		fflush(stdout);
		
		while(1){
			if(numlatstat[numlatposforread] == 1){
				latnum = &(numlats[numlatposforread]);
				int gpuarcnum = latnum->numarc;
				cudaMemcpy(h_dsttime, latnum->arr_sttime, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
				cudaMemcpy(h_dentime, latnum->arr_entime, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
				cudaMemcpy(h_dtransid, latnum->arr_transid, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
				cudaMemcpy(h_dphoneid, latnum->arr_phoneid, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
				break;
			}
		}
		
		pr = CalcArcFB(latnum, emitprob, outputnum, offset);//計算分子佔有率
		dur = latnum->nodes[latnum->numnode -1].time;
		printf("num:\tT=%d,pr/fr = %.6e\n",dur,pr/dur);
		fflush(stdout);
		total_Pr_num += pr;
		//printf("Load Num Over pos: %d\n", numlatposforread);
		
		while(1){
			if(denlatstat[denlatposforread] == 1){
				latden = &(denlats[denlatposforread]);
				int gpuarcnum = latden->numarc;
				cudaMemcpy(h_dsttime, latden->arr_sttime, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
				cudaMemcpy(h_dentime, latden->arr_entime, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
				cudaMemcpy(h_dtransid, latden->arr_transid, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
				cudaMemcpy(h_dphoneid, latden->arr_phoneid, sizeof(int) *gpuarcnum, cudaMemcpyHostToDevice);
				break;
			}
		}
		
		lat_in_lat = LatInLat(latnum,latden);//判斷分子詞圖是否在分母詞圖中
		//gettimeofday(&timest,NULL);
		if(lat_in_lat){
			pr = CalcArcFB(latden, emitprob, outputden, offset);//若在，采用于分子相同的計算方法
			printf("den:\tT=%d,pr/fr = %.6e\n",dur,pr/dur);
			fflush(stdout);
		}
		else{
			pr = CalcArcFBAddNum(latden, latnum, pr, emitprob, outputden, outputnum, offset); //否則採用詞圖補償再計算
			printf("den[+num]:\tT=%d,pr/fr = %.6e\n",dur,pr/dur);
			fflush(stdout);
		}

	/*	
		if(sentidx == 52243){
			double *tmp = new double [2000*3004];
			cudaStat = cudaMemcpy(tmp, output, sizeof(double) *1000*3004, cudaMemcpyDeviceToHost);
			if(cudaStat != cudaSuccess){
				printf("fsafsafsa\n");
			}
			for(int m =482*3004; m < 482*3004 +135*3004; m++){
				printf("%.6e\n",tmp[m]);
			}
			delete []tmp;
			exit(0);
		}
	*/
		//cudaStreamSynchronize(stream);
		//gettimeofday(&timeen,NULL);
		//printf("CalcFB time: %.2f ms\n", 1000.0* (timeen.tv_sec - timest.tv_sec) + (timeen.tv_usec - timest.tv_usec)/1000.0);

		total_Pr_den += pr;
		total_T += dur;
		offset += statesize* dur;
		//printf("Load Den Over pos: %d\n", denlatposforread);
		
		cudaStreamSynchronize(stream);
		FreeLat(latden);
		FreeLat(latnum);

		denlatstat[denlatposforread] = 0;
		numlatstat[denlatposforread] = 0;
		denlatposforread ++;
		numlatposforread ++;
		if(denlatposforread == LATCACHE){
			denlatposforread =0;
		}
		if(numlatposforread == LATCACHE){
			numlatposforread =0;
		}
	}
}

////Lattice must be DOS format
void CMyFBLat::ReadLatThread(int sentnum, int *sentlist)
{
	int i,j,sentidx;
	FILE *fp_scp;
	char buf[MAXLINE],latname[MAXLINE];
	string phonename;
	char *p, *pp;
	Lattice *lat;

	for(j =0; j< sentnum ;j++){
		while(1){
			if(numlatstat[numlatposforwrite] == 0)
				break;
		}
		lat = &(numlats[numlatposforwrite]);
		sentidx = sentlist[j];
		strcpy(latname,Lat_num[sentidx].c_str());
		fp_scp = fopen(latname,"r");
		if(fp_scp ==NULL){
			printf("Error: Cannot open %s for read.",latname);
			exit(0);
		}
		while(fgets(buf,MAXLINE,fp_scp)){
			p =strstr(buf,"N=");
			pp =strstr(buf,"L=");
			if( p && pp){
				*(pp -1) = '\0';
				lat->numarc = atoi(pp+2);
				lat->numnode = atoi(p+2);
				lat->nodes = new Node [lat->numnode];
				lat->arcs  = new Arc [lat->numarc];
				
				int *prearcsize = new int[lat->numnode];
				int *folarcsize = new int[lat->numnode];
				
				lat->arr_sttime = new int [lat->numarc];
				lat->arr_entime = new int [lat->numarc];
				lat->arr_transid  = new int [lat->numarc];
				lat->arr_phoneid  = new int [lat->numarc];
				
				for(i=0;i< lat->numnode;i++){
					lat->nodes[i].prearcnum =0;
					lat->nodes[i].folarcnum =0;
					lat->nodes[i].prearcs = (int *)malloc(BLOCKSIZE*sizeof(int));
					lat->nodes[i].folarcs = (int *)malloc(BLOCKSIZE*sizeof(int));
					prearcsize[i] = BLOCKSIZE;
					folarcsize[i] = BLOCKSIZE;
				}
				
				for(i=0;i< lat->numnode;i++){
					fgets(buf,MAXLINE,fp_scp);
					p = strstr(buf,"I=");
					pp = strstr(buf, " ");
					*pp = '\0';
					lat->nodes[i].idx = atoi(p+2);
					p = strstr(pp+1,"t=");
					pp = strstr(pp+1, "\t");
					*pp = '\0';
					lat->nodes[i].time = (int) (100 *atof(p+2) +0.5);
				}
				for(i=0;i< lat->numarc;i++){
					fgets(buf,MAXLINE,fp_scp);
					p = strstr(buf,"J=");
					pp = strstr(buf, "\t");
					*pp = '\0';
					lat->arcs[i].idx = atoi(p+2);
					p = strstr(pp+1,"S=");
					pp = strstr(pp+1, "\t");
					*pp = '\0';
					lat->arcs[i].stnode = &(lat->nodes[atoi(p+2)]);
					
					Node *curnode = lat->arcs[i].stnode;
					if(curnode->folarcnum == folarcsize[curnode->idx]){
						folarcsize[curnode->idx] *= 2;
						curnode->folarcs = (int*)realloc(curnode->folarcs,sizeof(int)*folarcsize[curnode->idx]);
					}
					curnode->folarcs[curnode->folarcnum++] = i;
					
					p = strstr(pp+1,"E=");
					pp = strstr(pp+1, "\t");
					*pp = '\0';
					lat->arcs[i].ennode = &(lat->nodes[atoi(p+2)]);
	
					curnode = lat->arcs[i].ennode;
					if(curnode->prearcnum == prearcsize[curnode->idx]){
						prearcsize[curnode->idx] *= 2;
						curnode->prearcs = (int*)realloc(curnode->prearcs,sizeof(int)*prearcsize[curnode->idx]);
					}
					curnode->prearcs[curnode->prearcnum++] = i;
					
					p = strstr(pp+1,"l=");
					pp = strstr(pp+1, "\t");
					*pp = '\0';
					lat->arcs[i].lmlike = atof(p+2);
					p = strstr(pp+1,"d=");
					pp = strstr(pp+1, ",");
					*pp = '\0';
					phonename = p+3;
					phonename = hmmlisthash.find(phonename)->second;
					p = pp+1;
					pp = strstr(pp+1, ":");
					*pp = '\0';
					lat->arcs[i].duration = (int) (100 *atof(p) +0.5);
					lat->arcs[i].phone = triphonehash.find(phonename)->second;
					
					lat->arr_sttime[i] = lat->arcs[i].stnode->time;
					lat->arr_entime[i] = lat->arcs[i].ennode->time;
					lat->arr_phoneid[i] = lat->arcs[i].phone->idx;
					lat->arr_transid[i] = lat->arcs[i].phone->trans->idx;	
				}
				free(prearcsize);
				free(folarcsize);
			}
		}
		lat->startarc = new Arc;
		lat->startarc->ennode = lat->nodes;
		fclose(fp_scp);

		numlatstat[numlatposforwrite] = 1;
		//printf("Read Num Over pos: %d\n", numlatposforwrite);
		numlatposforwrite ++;
		if(numlatposforwrite == LATCACHE){
			numlatposforwrite = 0;
		}
		
		while(1){
			if(denlatstat[denlatposforwrite] == 0)
				break;
		}
		lat = &(denlats[denlatposforwrite]);
		sentidx = sentlist[j];
		strcpy(latname,Lat_den[sentidx].c_str());
		fp_scp = fopen(latname,"r");
		if(fp_scp ==NULL){
			printf("Error: Cannot open %s for read.",latname);
			exit(0);
		}
printf("here!!!\n");
		while(fgets(buf,MAXLINE,fp_scp)){
			p =strstr(buf,"N=");
			pp =strstr(buf,"L=");
			if( p && pp){
				*(pp -1) = '\0';
				lat->numarc = atoi(pp+2);
				lat->numnode = atoi(p+2);
				lat->nodes = new Node [lat->numnode];
				lat->arcs  = new Arc [lat->numarc];
				
				int *prearcsize = new int[lat->numnode];
				int *folarcsize = new int[lat->numnode];
				
				lat->arr_sttime = new int [lat->numarc];
				lat->arr_entime = new int [lat->numarc];
				lat->arr_transid  = new int [lat->numarc];
				lat->arr_phoneid  = new int [lat->numarc];
				
				for(i=0;i< lat->numnode;i++){
					lat->nodes[i].prearcnum =0;
					lat->nodes[i].folarcnum =0;
					lat->nodes[i].prearcs = (int *)malloc(BLOCKSIZE*sizeof(int));
					lat->nodes[i].folarcs = (int *)malloc(BLOCKSIZE*sizeof(int));
					prearcsize[i] = BLOCKSIZE;
					folarcsize[i] = BLOCKSIZE;
				}
				
				for(i=0;i< lat->numnode;i++){
					fgets(buf,MAXLINE,fp_scp);
					p = strstr(buf,"I=");
					pp = strstr(buf, " ");
					*pp = '\0';
					lat->nodes[i].idx = atoi(p+2);
					p = strstr(pp+1,"t=");
					pp = strstr(pp+1, "\t");
					*pp = '\0';
					lat->nodes[i].time = (int) (100 *atof(p+2) +0.5);
				}
				for(i=0;i< lat->numarc;i++){
					fgets(buf,MAXLINE,fp_scp);
					p = strstr(buf,"J=");
					pp = strstr(buf, "\t");
					*pp = '\0';
					lat->arcs[i].idx = atoi(p+2);
					p = strstr(pp+1,"S=");
					pp = strstr(pp+1, "\t");
					*pp = '\0';
					lat->arcs[i].stnode = &(lat->nodes[atoi(p+2)]);
					
					Node *curnode = lat->arcs[i].stnode;
					if(curnode->folarcnum == folarcsize[curnode->idx]){
						folarcsize[curnode->idx] *= 2;
						curnode->folarcs = (int*)realloc(curnode->folarcs,sizeof(int)*folarcsize[curnode->idx]);
					}
					curnode->folarcs[curnode->folarcnum++] = i;
					
					p = strstr(pp+1,"E=");
					pp = strstr(pp+1, "\t");
					*pp = '\0';
					lat->arcs[i].ennode = &(lat->nodes[atoi(p+2)]);
	
					curnode = lat->arcs[i].ennode;
					if(curnode->prearcnum == prearcsize[curnode->idx]){
						prearcsize[curnode->idx] *= 2;
						curnode->prearcs = (int*)realloc(curnode->prearcs,sizeof(int)*prearcsize[curnode->idx]);
					}
					curnode->prearcs[curnode->prearcnum++] = i;
					
					p = strstr(pp+1,"l=");
					pp = strstr(pp+1, "\t");
					*pp = '\0';
					lat->arcs[i].lmlike = atof(p+2);
					p = strstr(pp+1,"d=");
					pp = strstr(pp+1, ",");
					*pp = '\0';
					phonename = p+3;
					phonename = hmmlisthash.find(phonename)->second;
					p = pp+1;
					pp = strstr(pp+1, ":");
					*pp = '\0';
					lat->arcs[i].duration = (int) (100 *atof(p) +0.5);
					lat->arcs[i].phone = triphonehash.find(phonename)->second;
					
					lat->arr_sttime[i] = lat->arcs[i].stnode->time;
					lat->arr_entime[i] = lat->arcs[i].ennode->time;
					lat->arr_phoneid[i] = lat->arcs[i].phone->idx;
					lat->arr_transid[i] = lat->arcs[i].phone->trans->idx;	
				}
				free(prearcsize);
				free(folarcsize);
			}
		}
		lat->startarc = new Arc;
		lat->startarc->ennode = lat->nodes;
		fclose(fp_scp);

		denlatstat[denlatposforwrite] = 1;
		//printf("Read Den Over pos: %d\n", denlatposforwrite);
		denlatposforwrite ++;
		if(denlatposforwrite == LATCACHE){
			denlatposforwrite = 0;
		}
		
		 printf("Lattice read over.\n");
		 fflush(stdout);
	}
#if 0
	int idx_trans = 45;
	float *tmpvalue = new float [idx_trans *MAXSTATE *MAXSTATE];
	float *dtmpvalue;
	int *tmpnumstate0 = new int [idx_trans];
	int *dtmpnumstate0;
	cudaMemcpyFromSymbol(&dtmpvalue ,devtrans, sizeof(dtmpvalue));
	cudaMemcpyFromSymbol(&dtmpnumstate0 ,devnumstates, sizeof(dtmpnumstate0));
	cudaMemcpy(tmpvalue, dtmpvalue,sizeof(float) *idx_trans *MAXSTATE *MAXSTATE, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpnumstate0, dtmpnumstate0, sizeof(int)*idx_trans, cudaMemcpyDeviceToHost);
	for(int l =0;l < idx_trans; l++){
		printf("numstate: %d\n", tmpnumstate0[l]);
		for(int k =0;k < MAXSTATE; k++){
			for(int m =0; m< MAXSTATE; m++)
				printf("%.6e ",expf(tmpvalue[m + k*MAXSTATE + l*MAXSTATE *MAXSTATE]));
			printf("\n");
		}
	}
	printf("total trans num: %d\n", idx_trans);
	delete []tmpvalue;
	delete []tmpnumstate0;
	exit(0);
#endif

#if 0
	int cur_arcs = 211;
	int *dtmptransid;
	int *dtmpphoneid;
	int *dtmpsttime;
	int *dtmpentime;
	int *tmpsttime   = new int [cur_arcs];
	int *tmpentime   = new int [cur_arcs];
	int *tmptransid = new int [cur_arcs];
	int *tmpphoneid = new int [cur_arcs];
	
	cudaMemcpyFromSymbol(&dtmptransid ,devarctransid, sizeof(dtmptransid));
	cudaMemcpyFromSymbol(&dtmpphoneid ,devarcphoneid, sizeof(dtmpphoneid));
	cudaMemcpyFromSymbol(&dtmpsttime ,devarcsttime, sizeof(dtmpsttime));
	cudaMemcpyFromSymbol(&dtmpentime ,devarcentime, sizeof(dtmpentime));

	cudaMemcpy(tmptransid, dtmptransid, sizeof(int)*cur_arcs , cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpphoneid, dtmpphoneid, sizeof(int)*cur_arcs , cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpsttime, dtmpsttime, sizeof(int)*cur_arcs , cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpentime, dtmpentime,sizeof(int)*cur_arcs , cudaMemcpyDeviceToHost);

	for(int arcid = 0; arcid < 211;arcid++){
		printf("arcid: %d\n", arcid);
		printf("arc sttime: %d\n", tmpsttime[arcid]);
		printf("arc entime: %d\n", tmpentime[arcid]);
		printf("arc transid: %d\n", tmptransid[arcid]);
		printf("arc phoneid: %d\n", tmpphoneid[arcid]);
	}
	delete []tmpphoneid;
	delete []tmptransid;
	delete []tmpsttime;
	delete []tmpentime;
	exit(0);
#endif
}