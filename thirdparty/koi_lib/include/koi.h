struct ChunkDesc {
  unsigned int startRowIn;
  unsigned int numRowsIn;
  unsigned int startRowOut;
  int padding_;
};

int host_run_lstm_reverse96(void * chunks,
			    void * outvW,
			    void * lW,
			    void * b,
			    void * out,
			    int num_chunks);


int host_run_lstm_fwd96(void * chunks,
			void * outvW,
			void * lW,
			void * b,
			void * out,
			int num_chunks);


int host_run_lstm_reverse_quantized96(void * chunks,
				    void * outvW,
				    void * lW,
				    void * b,
				    void * quantization_scale,
				    void * out,
				    int num_chunks);


int host_run_lstm_fwd_quantized96(void * chunks,
				void * outvW,
				void * lW,
				void * b,
				void * quantization_scale,
				void * out,
				int num_chunks);

int host_run_lstm_reverse_quantized128(void * chunks,
				    void * outvW,
				    void * lW,
				    void * b,
				    void * quantization_scale,
				    void * out,
				    int num_chunks);


int host_run_lstm_fwd_quantized128(void * chunks,
				void * outvW,
				void * lW,
				void * b,
				void * quantization_scale,
				void * out,
				int num_chunks);


int host_back_guide_step(void * chunks,
			 void * chunk_results,
			 int num_chunks,
			 void * post,
			 int post_stride,
			 void * aux_buffer,
			 void* path,
			 void* moves,
			 void* weights,
			 void* sequence,
			 void* q_string,
			 float qscale,
			 float qshift,
			 int beam_width,
			 float beam_cut,
			 float fixed_stay_score);

int host_beam_search_step(void * chunks,
			 void * chunk_results,
			 int num_chunks,
			 void * post,
			 int post_stride,
			 void * aux_buffer,
			 void* path,
			 void* moves,
			 void* weights,
			 void* sequence,
			 void* q_string,
			 float qscale,
			 float qshift,
			 int beam_width,
			 float beam_cut,
			 float fixed_stay_score);

int host_compute_posts_step(void * chunks,
			    void * chunk_results,
			    int num_chunks,
			    void * post,
			    int post_stride,
			    void * aux_buffer,
			    void* path,
			    void* moves,
			    void* weights,
			    void* sequence,
			    void* q_string,
			    float qscale,
			    float qshift,
			    int beam_width,
			    float beam_cut,
			    float fixed_stay_score);

int host_run_decode(void * chunks,
		    void * chunk_results,
		    int num_chunks,
		    void * post,
		    int post_stride,
		    void * aux_buffer,
		    void* path,
		    void* moves,
		    void* weights,
		    void* sequence,
		    void* q_string,
		    float qscale,
		    float qshift,
		    int beam_width,
		    float beam_cut,
		    float fixed_stay_score,
		    int move_pad);

int fwd_bwd_logspace_host(int T,
			  int N,
			  int L,
			  void * alpha,
			  void * beta_T,
			  void * beta_stay,
			  void * beta_move,
			  void * stay_scores,
			  void * move_scores);

int fwd_bwd_logspace_loop_host(int T,
			       int N,
			       int L,
			       void * alpha,
			       void * beta_T,
			       void * beta_stay,
			       void * beta_move,
			       void * stay_scores,
			       void * move_scores);

int logZ_fwd_host_log(int T,
		      int N,
		      int C,
		      int k,
		      float * logZ,
		      float * Ms_grad,
		      float * Ms,
		      float * v0,
		      float * vT,
		      int * idx);


int logZ_fwd_host_max(int T,
		      int N,
		      int C,
		      int k,
		      float * logZ,
		      float * Ms_grad,
		      float * Ms,
		      float * v0,
		      float * vT,
		      int * idx);


int bwd_scores_host_sparse_log(float * betas,
			       float * Ms,
			       float * vT,
			       int * idx_T,
			       int T,
			       int N,
			       int C);


int bwd_scores_host_sparse_max(float * betas,
			       float * Ms,
			       float * vT,
			       int * idx_T,
			       int T,
			       int N,
			       int C);


int fwd_scores_host_sparse_log(float * alphas,
			       float * Ms,
			       float * v0,
			       int * idx,
			       int T,
			       int N,
			       int C);


int fwd_scores_host_sparse_max(float * alphas,
			       float * Ms,
			       float * v0,
			       int * idx,
			       int T,
			       int N,
			       int C);
