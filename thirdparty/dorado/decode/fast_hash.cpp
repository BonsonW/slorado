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
 *  Copyright (C) 2023 Bonson Wong
 */


#include "fast_hash.h"

// Compression function for Merkle-Damgard construction.
// This function is generated using the framework provided.
static inline uint64_t mix(uint64_t h) {
    h ^= h >> 23;
    h *= 0x2127599bf4325c37ULL;
    h ^= h >> 47;
    return h;
}

int kmerhash_init(void)
{
    extern uint64_t kmerhash_lookup[NUM_STATES];
	for (uint64_t i = 0; i < NUM_STATES; i++) {
	    kmerhash_lookup[i] = mix(i);
	}
	return 1;
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
uint64_t chainfasthash64(uint64_t hash, uint64_t val) {
    const uint64_t m = 0x880355f21e6d1965ULL;
    extern uint64_t kmerhash_lookup[NUM_STATES];

    hash ^= kmerhash_lookup[val];
    // hash ^= mix(val);
    hash *= m;
    return mix(hash);
}