struct ChunkDesc {
    unsigned int startRowIn;
    unsigned int numRowsIn;
    unsigned int startRowOut;
    int padding_;
};

enum KoiTypeId { KOI_I8 = 0, KOI_F16 };

int host_run_lstm_reverse96(void *chunks,
                            void *outvW,
                            void *lW,
                            void *b,
                            void *out,
                            int num_chunks);

int host_run_lstm_fwd96(void *chunks, void *outvW, void *lW, void *b, void *out, int num_chunks);

int host_run_lstm_reverse_quantized96(void *chunks,
                                      void *outvW,
                                      void *lW,
                                      void *b,
                                      void *quantization_scale,
                                      void *out,
                                      int num_chunks);

int host_run_lstm_fwd_quantized96(void *chunks,
                                  void *outvW,
                                  void *lW,
                                  void *b,
                                  void *quantization_scale,
                                  void *out,
                                  int num_chunks);

int host_run_lstm_reverse_quantized128(void *chunks,
                                       void *outvW,
                                       void *lW,
                                       void *b,
                                       void *quantization_scale,
                                       void *out,
                                       int num_chunks);

int host_run_lstm_fwd_quantized128(void *chunks,
                                   void *outvW,
                                   void *lW,
                                   void *b,
                                   void *quantization_scale,
                                   void *out,
                                   int num_chunks);

void host_lstm_step_f16(void *stream,
                        int batch_size,
                        int layer_size,
                        void *bias,
                        void *gate_buf,
                        void *state_buf,
                        void *out);

void host_cutlass_lstm(void *stream,
                       enum KoiTypeId type,
                       int layer_idx,
                       int batch_size,
                       int layer_size,
                       int chunk_size,
                       int direction,
                       int inout_stride,
                       void *inout,
                       void *weights,
                       void *bias,
                       void *scale,
                       void *lstm_state,
                       void *workspace_4KiB,
                       int interleave,
                       int brf_state);

void host_convert(void *stream,
                  void *in,
                  int in_stride0,
                  int in_stride1,
                  int in_stride2,
                  enum KoiTypeId in_type,
                  void *out,
                  int out_stride0,
                  int out_stride1,
                  int out_stride2,
                  enum KoiTypeId out_type,
                  int size0,
                  int size1,
                  int size2);

void host_convert_inplace(void *stream,
                          void *in_out,
                          enum KoiTypeId in_type,
                          enum KoiTypeId out_type,
                          int larger_stride0,
                          int smaller_offset1,
                          int size0,
                          int size1);

void host_bias_tanh_scale_f16(void *stream,
                              int num_rows,
                              int layer_size,
                              float scale_factor,
                              void *in_out,
                              void *bias);

void host_bias_swish_f16_clamp(void *stream,
                               int size0,
                               int size1,
                               int stride0,
                               void *in_out,
                               void *bias,
                               float max_value);

void host_bias_swish_f16_i8(void *stream,
                            int size0,
                            int size1,
                            int in_stride0,
                            int out_stride0,
                            void *in,
                            void *out,
                            void *bias,
                            float scale,
                            float zero_offset);

void host_bias_swish_f16_i8_inplace(void *stream,
                                    int size0,
                                    int size1,
                                    int out_offset,
                                    void *in_out,
                                    void *bias,
                                    float scale,
                                    float zero_offset);

void host_window_ntcw_f16(void *stream,
                          int N_in_stride,
                          int T_in_stride,
                          int C_in_stride,
                          int N,
                          int T_in,
                          int C,
                          int W,
                          int conv_stride,
                          int N_out_stride,
                          int T_out_stride,
                          int C_out_stride,
                          int W_out_stride,
                          void *in_buf,
                          void *out_buf);

int host_convolution_swish_f16(void *stream,
                               int N,
                               int C_in,
                               int C_out,
                               int T_in,
                               int window,
                               int stride,
                               int padding,
                               void *in_buf,
                               void *out_buf,
                               void *weights,
                               void *bias,
                               float clamp_max);

void host_transpose_f16(void *stream,
                        void *in,
                        int size0,
                        int size1,
                        int size2,
                        int in_stride0,
                        int in_stride1,
                        int in_stride2,
                        int out_stride0,
                        int out_stride1,
                        int out_stride2,
                        void *out);

int host_back_guide_step(void *chunks,
                         void *chunk_results,
                         int num_chunks,
                         void *post,
                         int post_stride,
                         void *aux_buffer,
                         void *path,
                         void *moves,
                         void *weights,
                         void *sequence,
                         void *q_string,
                         float qscale,
                         float qshift,
                         int beam_width,
                         float beam_cut,
                         float fixed_stay_score);

int host_beam_search_step(void *chunks,
                          void *chunk_results,
                          int num_chunks,
                          void *post,
                          int post_stride,
                          void *aux_buffer,
                          void *path,
                          void *moves,
                          void *weights,
                          void *sequence,
                          void *q_string,
                          float qscale,
                          float qshift,
                          int beam_width,
                          float beam_cut,
                          float fixed_stay_score);

int host_compute_posts_step(void *chunks,
                            void *chunk_results,
                            int num_chunks,
                            void *post,
                            int post_stride,
                            void *aux_buffer,
                            void *path,
                            void *moves,
                            void *weights,
                            void *sequence,
                            void *q_string,
                            float qscale,
                            float qshift,
                            int beam_width,
                            float beam_cut,
                            float fixed_stay_score);

int host_run_decode(void *chunks,
                    void *chunk_results,
                    int num_chunks,
                    void *post,
                    int post_stride,
                    void *aux_buffer,
                    void *path,
                    void *moves,
                    void *weights,
                    void *sequence,
                    void *q_string,
                    float qscale,
                    float qshift,
                    int beam_width,
                    float beam_cut,
                    float fixed_stay_score,
                    int move_pad);

int fwd_bwd_logspace_host(int T,
                          int N,
                          int L,
                          void *alpha,
                          void *beta_T,
                          void *beta_stay,
                          void *beta_move,
                          void *stay_scores,
                          void *move_scores);

int fwd_bwd_logspace_loop_host(int T,
                               int N,
                               int L,
                               void *alpha,
                               void *beta_T,
                               void *beta_stay,
                               void *beta_move,
                               void *stay_scores,
                               void *move_scores);

int logZ_fwd_host_log_NZ5(int T,
                          int N,
                          int C,
                          int k,
                          float *logZ,
                          float *Ms_grad,
                          float *Ms,
                          float *v0,
                          float *vT,
                          int *idx);

int logZ_fwd_host_log_NZ3(int T,
                          int N,
                          int C,
                          int k,
                          float *logZ,
                          float *Ms_grad,
                          float *Ms,
                          float *v0,
                          float *vT,
                          int *idx);

int logZ_fwd_host_max(int T,
                      int N,
                      int C,
                      int k,
                      float *logZ,
                      float *Ms_grad,
                      float *Ms,
                      float *v0,
                      float *vT,
                      int *idx);

int bwd_scores_host_sparse_log_NZ5(float *betas,
                                   float *Ms,
                                   float *vT,
                                   int *idx_T,
                                   int T,
                                   int N,
                                   int C);

int bwd_scores_host_sparse_log_NZ3(float *betas,
                                   float *Ms,
                                   float *vT,
                                   int *idx_T,
                                   int T,
                                   int N,
                                   int C);

int bwd_scores_host_sparse_max(float *betas, float *Ms, float *vT, int *idx_T, int T, int N, int C);

int fwd_scores_host_sparse_log_NZ5(float *alphas,
                                   float *Ms,
                                   float *v0,
                                   int *idx,
                                   int T,
                                   int N,
                                   int C);

int fwd_scores_host_sparse_log_NZ3(float *alphas,
                                   float *Ms,
                                   float *v0,
                                   int *idx,
                                   int T,
                                   int N,
                                   int C);

int fwd_scores_host_sparse_max(float *alphas, float *Ms, float *v0, int *idx, int T, int N, int C);
