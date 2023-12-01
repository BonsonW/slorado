/* The MIT License
   Copyright (C) 2012 Zilong Tan (eric.zltan@gmail.com)
   Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without
   restriction, including without limitation the rights to use, copy,
   modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/*  Modifications licensed by MIT licence.
 *  Copyright (C) 2020 Oxford Nanopore Technologies
 */

/*  Modifications licensed by MIT licence.
 *  Copyright (C) 2023 Bonson Wong (bonson.ym@gmail.com)
 */

#ifndef FASTHASH_H
#define FASTHASH_H

#include <stdint.h>
#include <stdio.h>

#define NUM_STATES 64

// Compression function for Merkle-Damgard construction.
// This function is generated using the framework provided.
#define FASTHASH_MIX(h) ({					\
			(h) ^= (h) >> 23;				\
			(h) *= 0x2127599bf4325c37ULL;	\
			(h) ^= (h) >> 47; })

// Precompute hashes for kmers used in beam-search
static inline int kmerhash_init(void) {
    extern uint64_t kmerhash_lookup[NUM_STATES];
    for (uint64_t i = 0; i < NUM_STATES; i++) {	
       kmerhash_lookup[i] = FASTHASH_MIX(i);
    }
    return 1;
}

/**
 * fasthash64 - 64-bit implementation of fasthash
 * @buf:  data buffer
 * @len:  data size
 * @seed: the seed
 */
static inline uint64_t fasthash64(const void *buf, size_t len, uint64_t seed) {
	const uint64_t    m = 0x880355f21e6d1965ULL;
	const uint64_t *pos = (const uint64_t *)buf;
	const uint64_t *end = pos + (len / 8);
	const unsigned char *pos2;
	uint64_t h = seed ^ (len * m);
	uint64_t v;

	while (pos != end) {
		v  = *pos++;
		h ^= FASTHASH_MIX(v);
		h *= m;
	}

	pos2 = (const unsigned char*)pos;
	v = 0;

	switch (len & 7) {
	case 7: v ^= (uint64_t)pos2[6] << 48;
	case 6: v ^= (uint64_t)pos2[5] << 40;
	case 5: v ^= (uint64_t)pos2[4] << 32;
	case 4: v ^= (uint64_t)pos2[3] << 24;
	case 3: v ^= (uint64_t)pos2[2] << 16;
	case 2: v ^= (uint64_t)pos2[1] << 8;
	case 1: v ^= (uint64_t)pos2[0];
		h ^= FASTHASH_MIX(v);
		h *= m;
	}

	return FASTHASH_MIX(h);
} 

/**
 * fasthash32 - 32-bit implementation of fasthash
 * @buf:  data buffer
 * @len:  data size
 * @seed: the seed
 */
static inline uint32_t fasthash32(const void *buf, size_t len, uint32_t seed) {
	// the following trick converts the 64-bit hashcode to Fermat
	// residue, which shall retain information from both the higher
	// and lower parts of hashcode.
        uint64_t h = fasthash64(buf, len, seed);
	return h - (h >> 32);
}

/**  Chain a new value to hash
 *
 *   `fasthash64` specialised to case of calculating the new hash
 *   from the previous data when a new value is appended.
 *
 *   Note:
 *       It is assumed that the value added is always 64bits wide,
 *   unlike `fasthash64` which combines blocks of bytes into 64bits
 *   values.
 *
 *   Args:
 *       hash (uint64_t): Hash of previous data
 *       val  (uint64_t): Value to chain to hash
 *
 *   Returns:
 *       uint64_t: New hash with value added
 **/
static inline uint64_t chainfasthash64(uint64_t hash, uint64_t val) {
    const uint64_t m = 0x880355f21e6d1965ULL;
    extern uint64_t kmerhash_lookup[NUM_STATES];    
    hash ^= kmerhash_lookup[val];
    // hash ^= mix(val);
    hash *= m;
    return FASTHASH_MIX(hash);
}

#endif
