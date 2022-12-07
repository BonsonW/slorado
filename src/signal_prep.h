#include <slow5/slow5.h>
#include <torch/torch.h>

slow5_rec_t *read_file_to_record(char *file_path);
int trim(torch::Tensor signal, int window_size=40, float threshold_factor=2.4, int min_elements=3);