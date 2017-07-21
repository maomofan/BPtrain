///////////////////////////
////  jiapan        ///////        
////  iflytek       ///////
////  2012/12/10     ///////
///////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <time.h>

#include "train.h"

void test();
void test2(int argc,char*argv[]);

int main(int argc, char *argv[])
{
	printf("This is verson2.\n");
	struct WorkPara *paras;
	struct DTPara *DT_paras;
	int *chunk_index;
	int cur_chunk_sents;
	int i,j;
	int correct_samples = 0;
	double timenow;
	timenow = time(NULL);
	
	Interface *InterObj = new Interface;
	CMyFBLat *FBObj = new CMyFBLat;

	InterObj->Initial(argc, argv);
	paras = InterObj->para;
	DT_paras = InterObj->DT_para;
	FBObj->Initial(DT_paras->fn_scp, DT_paras->fn_state,DT_paras->fn_model,DT_paras->fn_hmmlist,DT_paras->acscale,DT_paras->lmscale,DT_paras->kappa,DT_paras->outputsize,paras->gpu_used);
	InterObj->get_pfile_info();
	InterObj->get_chunk_info(paras->train_sent_range);
	
	/////////train	
	GPU_trainBP *TrainObj = new GPU_trainBP(InterObj,FBObj);
		
	chunk_index = new int [InterObj->total_chunks];
	for(i=0;i< InterObj->total_chunks;i++){
		chunk_index[i] =i;
	}
	InterObj->GetRandIndex(chunk_index,InterObj->total_chunks);

	for(i=0;i< InterObj->total_chunks; i++){
		cur_chunk_sents = InterObj->Readchunk(chunk_index[i]);
		fprintf(InterObj->fp_log,"Starting chunk %d of %d containing %d valid sentences.\n", i+1 ,InterObj->total_chunks, cur_chunk_sents);
		fflush(InterObj->fp_log);
		TrainObj->train(InterObj->sample_num_in_chunk[chunk_index[i]], InterObj->bunch_num_in_chunk, cur_chunk_sents, InterObj->sample_num_in_bunch, 
			InterObj->sent_num_in_bunch,InterObj->sent_list, paras->indata, paras->targ);
	}
	TrainObj->returnWeights(paras->weights,paras->bias);
	InterObj->Writeweights();
	
	delete [] chunk_index;

	timenow = time(NULL) - timenow;
	fprintf(InterObj->fp_log,"Object function per frame: %e\n", (FBObj->total_Pr_num - FBObj->total_Pr_den)/FBObj->total_T);
	fprintf(InterObj->fp_log,"Total cost time: %.1f s.\n", timenow);
	
	delete TrainObj;
	delete InterObj;
	delete FBObj;
	return 1;
}

/*
void test()  ////test the part of gpu training 
{
///for debug
	int i;
	FILE *fp_init_weight;
	
	float *in;
	int *targ;
	int gpu_used = 0;
	int numlayers =5;
	int layersizes[5] = {473,1024,1024,1024,3969};
	int bunchsize	= 512;
	float momentum = 0;
	float lrate = 0.002;
	float weightcost = 0;
	float *weights[5];
	float *bias[5];
	char init_weightFN[] = "/home/jiapan/new_BP_Code/QN_cmp/test_mlp/mlp.0.wts";
	
	//// Init weights
	for(i =1; i< numlayers; i++)
	{
		int size	= layersizes[i] *layersizes[i-1];
		weights[i] = new float [size];
		bias[i] = new float [layersizes[i]];
	  
		memset(weights[i],0,size *sizeof(float));
		memset(bias[i],0,layersizes[i] *sizeof(float));
	}
	
	if(NULL ==(fp_init_weight = fopen(init_weightFN, "rb")))
	{
		printf("can not open initial weights file: %s\n", init_weightFN);
		exit(0);
	}
	else
	{
		int stat[10];
		char head[256];
	
		for(i =1; i< numlayers; i++)
		{
			fread(stat,sizeof(int),5,fp_init_weight);
			fread(head,sizeof(char),stat[4],fp_init_weight);
			
			if(stat[1] != layersizes[i] || stat[2] != layersizes[i -1])
			{
				printf("init weights node nums do not match\n");
				exit(0);
			}
			fread(weights[i],sizeof(float),layersizes[i -1] *layersizes[i],fp_init_weight);
			
			fread(stat,sizeof(int),5,fp_init_weight);
			fread(head,sizeof(char),stat[4],fp_init_weight);
			
			if(stat[2] != layersizes[i] || stat[1] != 1)
			{
				printf("init bias node nums do not match\n");
				exit(0);
			}
			fread(bias[i],sizeof(float),layersizes[i],fp_init_weight);
		}
		fclose(fp_init_weight);
	}

	GPU_trainBP *TrainObj = new GPU_trainBP(gpu_used, numlayers, layersizes, bunchsize, momentum, lrate, 
				weightcost, weights, bias);
				
		///for debug
    float *tmpin = new float[512 *473];
    char *tmpname = "/home/jiapan/new_BP_Code/QN_cmp/testin.txt";
    FILE *fp_tmp = fopen(tmpname,"rt");
    
    for(int tmpi =0;tmpi< 512 *473; tmpi++)
    {
    	fscanf(fp_tmp,"%f\n",&(tmpin[tmpi]));
    }
   	fclose(fp_tmp);
       
    int *tmptarg = new int[512];
    tmpname = "/home/jiapan/new_BP_Code/QN_cmp/testtarg.txt";
    fp_tmp = fopen(tmpname,"rt");
  
    for(int tmpi =0;tmpi< 512; tmpi++)
    {
    	fscanf(fp_tmp,"%d\n",&(tmptarg[tmpi]));
    }
    fclose(fp_tmp);
    
	TrainObj->train(512, tmpin ,tmptarg); 
	
	delete [] tmpin;
  delete [] tmptarg;
	delete TrainObj;
}

void test2(int argc,char*argv[]) ////test the part of Reading data 
{
///for debug
//gpu_used=0 numlayers=5 layersizes=473,1024,1024,1024,3969 bunchsize=512 momentum=0.0002 weightcost=0.0001 lrate=0.002 initwts_file= norm_file=/home/jiapan/Tandem_train/80H_Chinese/lib/fea_tr.norm fea_file=/home/jiapan/Tandem_train/80H_Chinese/lib/fea_tr.pfile targ_file=/home/jiapan/Tandem_train/80H_Chinese/lib/lab_state.pfile outwts_file=/home/jiapan/mlp.test.wts log_file=/home/jiapan/mlp.test.log train_sent_range=1-10000 cv_sent_range=10001-10002 fea_dim=43 fea_context=11 traincache=200 init_randem_seed=6346 targ_offset=5 DT_config_file=/home/jiapan/DNN_DT/debug_resource/mpe.t50.cfg lattice_list=/home/jiapan/DNN_DT/debug_resource/1.scp HMMmodel=/home/jiapan/DNN_DT/debug_resource/model/MODELS HMMlist=/home/jiapan/DNN_DT/debug_resource/model/hmmlist NumLatDir=/home/jiapan/DNN_DT/debug_resource/num DenLatDir=/home/jiapan/DNN_DT/debug_resource/den state_map_file=/home/jiapan/DNN_DT/debug_resource/model/state.map
	float *outprob = new float[100000*3969];
	float *stateprob = new float[100000*3969];
	
	for(int k=0;k< 100000*3969;k++){
		outprob[k] = 0;
		stateprob[k] = 0.1; 
	}
	int sent_list[] = {2,4};
	
	Interface *testObj = new Interface;
	testObj->Initial(argc, argv);
	testObj->get_pfile_info();
	testObj->get_chunk_info(testObj->para->train_sent_range);
	testObj->Readchunk(2);
	
	struct DTPara *DT_paras = testObj->DT_para;
	MyInitialise(DT_paras->fn_config, DT_paras->fn_scp, DT_paras->fn_model, DT_paras->fn_hmmlist, DT_paras->dn_num,  DT_paras->dn_den, 
				DT_paras->fn_map, DT_paras->outputsize);
	DoStatCompute(2,sent_list,stateprob,outprob);
	
	for(int k=0;k< 3969000;k++)
		printf("%d\n",outprob[k]);
	delete []outprob;
	delete []stateprob;
	delete testObj;
}
*/