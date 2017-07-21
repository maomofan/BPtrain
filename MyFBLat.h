#include <vector>
#include <map>
#include <string>
using namespace std;

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "MathFunc.h"

#define MAXPARTNUM 1000
#define MAXSTATE 5
#define LATCACHE 10

typedef struct _Transform
{
	string name;
	int idx;
	int numstate;
	float *value;
}Transform;

typedef struct _Triphone
{
	string name;
	int idx;
	int numstate;
	int *stateindex;
	Transform *trans;
}Triphone;

typedef struct _Node
{
	int idx;
	int time;
	int *prearcs;
	int *folarcs;
	int prearcnum;
	int folarcnum;
}Node;

typedef struct _Arc
{
	int idx;
	int duration;
	Node *stnode;
	Node *ennode;
	Triphone *phone;
	Logfloat lmlike;
	LogDouble aclike;
	LogDouble Arcalpha;
	LogDouble ArcBeta;
	LogDouble Arcocc;
	int latinlatTag;
}Arc;

typedef struct _Arcgpu
{
	int sttime;
	int entime;
	int numstate;
	int *stateindex;
	float *transval;
	LogDouble aclike;
	LogDouble *StateAlpha_t;
	LogDouble *StateBeta_t;
	LogDouble *stateOcc_t;
	LogDouble Arcocc;
}Arcgpu;

typedef struct _Lattice
{
	int numnode;
	int numarc;
	int *arr_sttime;
	int *arr_entime;
	int *arr_transid;
	int *arr_phoneid;
	Node *nodes;
	Arc *arcs;
	Arcgpu *arcs_gpu;
	Arc *startarc; ///a virtual arc used in LatinLat
}Lattice;

class CMyFBLat
{
public:
	CMyFBLat();
	~CMyFBLat();
public:
	void Initial(char *fn_scp, char *fn_state, char *fn_hmm, char *fn_hmmlist, double ac_scale, double lm_scale, double a_kappa, int state_size,  int gpu_used);
	void LoadLatlist(char *fn_scp);
	void LoadStateindex(char *fn_state);
	void LoadHMM(char *fn_model);
	void LoadHmmlist(char *fn_hmmlist);
	void ReadLat(char *fn_scp,Lattice *lat);
	int LatInLat(Lattice *latnum,Lattice* latden);
	int LatInLatPath(Lattice *latnum, Lattice *latden, Arc *arc1, Arc *arc2);
	void InitArc(Lattice *lat, int count);
	int InitArcgpu(Lattice *lat, int offset);
	void CalcArcStateOcc(int numarc, Lattice *lat, Logfloat *emitprob);
	LogDouble CalcArcOcc(Lattice *lat, LogDouble *aclikes, LogDouble *arcoccs);
	LogDouble CalcArcFB(Lattice *lat, Logfloat *emitprob, float *output, int offset);
	LogDouble CalcArcFBAddNum(Lattice *latden, Lattice *latnum, double pr_num, Logfloat *emitprob, float *outputden, float *outputnum, int offset);
	void DoLatFB( int sentnum, int *sentindex, Logfloat *emitprob,  float *outputnum, float *outputden);
	void FreeLat(Lattice *lat);
	void ReadLatThread(int sentnum, int *sentlist);
public:
	map<string,Triphone*> triphonehash;
	map<string,Transform*> transhash;
	map<string,string> hmmlisthash;
	map<string,int> stateindex;
	vector<string> Lat_num;
	vector<string> Lat_den;
	double acscale;
	double lmscale;
	double kappa;
	int statesize;
	double  total_Pr_num, total_Pr_den;
	double  total_T;
	
	cudaStream_t stream; 
private:
	int *h_dsttime;
	int *h_dentime;
	int *h_dtransid;
	int *h_dphoneid;
	int *h_offset;
	int *h_doffset;
	double *h_dBeta;
	double *h_dAlpha;
	double *h_dOcc;
	double *h_aclike;
	double *h_daclike;
	double *h_arcocc;
	double *h_darcocc;
	double *h_arcstateocc;
	double *h_arcocc_den;
	double *h_arcstateocc_den;
	int sum_occnum;
	int part_arcoffset[MAXPARTNUM];
	int part_arccount[MAXPARTNUM];
	int part_datacount[MAXPARTNUM];
	int part_dataoffset[MAXPARTNUM];
	int part_arcoffset_den[MAXPARTNUM];
	int part_arccount_den[MAXPARTNUM];
	int part_datacount_den[MAXPARTNUM];
	int part_dataoffset_den[MAXPARTNUM];
	
	Lattice numlats[LATCACHE];
	Lattice denlats[LATCACHE];
	int numlatstat[LATCACHE];
	int denlatstat[LATCACHE];
	int numlatposforwrite;
	int denlatposforwrite;
	int numlatposforread;
	int denlatposforread;
};
