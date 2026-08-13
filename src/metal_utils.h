/* Metal helpers for the Apple Silicon (MPS) build.
 *
 * openfish's Metal decoder (openfish_decode_gpu) reads the scores from an MTLBuffer, not a raw
 * pointer like the CUDA/HIP path. slorado's scores live in a torch MPS tensor, so on Metal we copy
 * them to host and hand them here to be wrapped in a shared MTLBuffer that openfish can decode.
 * Kept in slorado (not openfish) so the openfish submodule stays unchanged.
 */
#ifndef METAL_UTILS_H
#define METAL_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

// Copy a host float16 [N,T,C] scores buffer into a freshly allocated shared MTLBuffer (on the system
// default device, the same one openfish uses). Returns a +1-retained handle (bridged to void*) to
// pass as openfish_decode_gpu()'s scores_NTC; release it with slorado_metal_free_scores.
void *slorado_metal_upload_scores_f16(int n_timesteps, int batch_size, int n_channels, const void *scores_f16_NTC);

void slorado_metal_free_scores(void *handle);

// Host-visible base address of an MTLBuffer (an MPS tensor's storage().data() bit-cast to void*).
// On Apple Silicon MPS allocates shared-storage buffers, so this is the same unified memory the
// GPU just wrote -- letting the CPU beam read the scores in place with no copy. Returns NULL if
// the buffer has no CPU-accessible contents (private storage), so callers can fall back to a copy.
void *slorado_metal_buffer_contents(const void *mtl_buffer_handle);

// Byte length of an MTLBuffer, so callers can bounds-check a zero-copy read.
size_t slorado_metal_buffer_length(const void *mtl_buffer_handle);

#ifdef __cplusplus
}
#endif

#endif // METAL_UTILS_H
