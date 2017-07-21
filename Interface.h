#include <sys/time.h>
#include <math.h>

#define MAXLAYER 10
#define MAXLINE 1024
#define MAXCHUNK 10000
#define MAXBUNCHNUMPERCHUNK 1024

struct WorkPara
{
	char prior_FN[MAXLINE];
	char fea_FN[MAXLINE];
	char fea_normFN[MAXLINE];	
	int fea_dim;
	int fea_context;

	char targ_FN[MAXLINE];
	int targ_offset;
	
	int traincache;        ////frames to memory one time
	int bunchsize;
	int layersizes[MAXLAYER];
	float momentum;
	float weightcost;
	float lrate;
	float priorscale;
	
	char init_weightFN[MAXLINE];
	char out_weightFN[MAXLINE];
	char log_FN[MAXLINE];
	
	char train_sent_range[MAXLINE];
	char cv_sent_range[MAXLINE];
	
	int gpu_used;
	int init_randem_seed;
	float init_randem_weight_min;
	float init_randem_weight_max;
	float init_randem_bias_max;
	float init_randem_bias_min;
	
	float *indata;
	int *targ;
	float *prior;
	float *weights[MAXLAYER -1];
	float *bias[MAXLAYER -1];
};

struct DTPara
{
   char  fn_scp[MAXLINE];
   char  fn_model[MAXLINE];
   char  fn_hmmlist[MAXLINE];
   char  fn_state[MAXLINE];
   int   outputsize;
   double acscale;
   double kappa;
   double lmscale;
};

class Interface
{
public:
		Interface();
		~Interface();
public:
		void Initial(int argc, char **argv);
		void Writeweights();
		int  Readdata();
		void get_pfile_info();
		void get_chunk_info(char *range);
		void get_chunk_info_cv(char *range);
		int Readchunk(int index);
		int Readchunk_cv(int index);
		void GetRandIndex(int *vec, int len);
public:
		struct WorkPara *para;
		struct DTPara *DT_para;
		
		unsigned int total_frames;
		unsigned int total_sents;
		unsigned int total_chunks;
		unsigned int total_samples;
		unsigned int cv_total_chunks;
		unsigned int cv_total_samples;
		
		int *framesBeforeSent;
		int *chunk_frame_st;
		int *chunk_sent_st;
		int *cv_chunk_frame_st;
		int *sent_list;
		int sent_num_in_bunch[MAXBUNCHNUMPERCHUNK];
		int sample_num_in_bunch[MAXBUNCHNUMPERCHUNK];
		int sample_num_in_chunk[MAXCHUNK];
		int bunch_num_in_chunk;
		
		FILE *fp_log;
		int numlayers;
		int realbunchsize;
		int max_chunk_frames;
		int max_bunch_frames;
		int min_bunch_frames;
		float smooth_factor;
private:
		void get_uint(const char* hdr, const char* argname, unsigned int* val);
		void read_tail(FILE *fp, long int file_offset, unsigned int sentnum, int *out);
		
		void GetRandWeight(float *vec, float min, float max, int len);
		
		FILE *fp_data;
		FILE *fp_prior;
		FILE *fp_targ;
		FILE *fp_init_weight;
		FILE *fp_norm;
		FILE *fp_out;
				
		float *mean;
		float *dVar;
		
		int sent_st, sent_en;
		int cv_sent_st, cv_sent_en;
		int cur_chunk_index;
		int frames_read;
};
