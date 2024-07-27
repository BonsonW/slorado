/**
 * @file thread.c
 * @brief multi-thread implementation skeleton
 * @author Hasindu Gamaarachchi (hasindu@garvan.org.au)

MIT License

Copyright (c) 2019,2021  Hasindu Gamaarachchi (hasindu@unsw.edu.au)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/

#include <pthread.h>
#include "slorado.h"
#include "error.h"
#include "misc.h"

static inline int32_t steal_work(pthread_arg_t* all_args, int32_t n_threads) {
	int32_t i, c_i = -1;
	int32_t k;
	for (i = 0; i < n_threads; ++i){
        pthread_arg_t args = all_args[i];
        // fprintf(stderr, "endi : %d, starti : %d\n", args.endi,args.starti);
		if (args.endi-args.starti > STEAL_THRESH) {
            // fprintf(stderr, "gap : %d\n", args.endi-args.starti);
            c_i = i;
            break;
        }
    }
    if (c_i < 0) {
        return -1;
    }
	k = __sync_fetch_and_add(&(all_args[c_i].starti), 1);
    // fprintf(stderr,"k : %d, end %d, start %d\n", k, all_args[c_i].endi, all_args[c_i].starti);
	return k >= all_args[c_i].endi ? -1 : k;
}

void* pthread_single(void* voidargs) {
    int32_t i;
    pthread_arg_t* args = (pthread_arg_t*)voidargs;
    db_t* db = args->db;
    core_t* core = args->core;

#ifndef WORK_STEAL
    for (i = args->starti; i < args->endi; i++) {
        args->func(core,db,i);
    }
#else
    pthread_arg_t* all_args = (pthread_arg_t*)(args->all_pthread_args);
    //adapted from kthread.c in minimap2
    for (;;) {
		i = __sync_fetch_and_add(&args->starti, 1);
		if (i >= args->endi) {
            break;
        }
		args->func(core,db,i);
	}
	while ((i = steal_work(all_args,core->opt.num_thread)) >= 0) {
		args->func(core,db,i);
    }
#endif

    // fprintf(stderr, "thread %d done\n", (myargs->position) / THREADS);
    pthread_exit(0);
}

void pthread_db(core_t* core, db_t* db, void (*func)(core_t*, db_t*, int)){
    // create threads
    pthread_t tids[core->opt.num_thread];
    pthread_arg_t pt_args[core->opt.num_thread];
    int32_t t, ret;
    int32_t i = 0;
    int32_t num_thread = core->opt.num_thread;
    int32_t step = (db->n_rec + num_thread - 1) / num_thread;
    // todo: check for higher num of threads than the data
    // current works but many threads are created despite

    // set the data structures
    for (t = 0; t < num_thread; t++) {
        pt_args[t].core = core;
        pt_args[t].db = db;
        pt_args[t].starti = i;
        i += step;
        if (i > db->n_rec) {
            pt_args[t].endi = db->n_rec;
        } else {
            pt_args[t].endi = i;
        }
        pt_args[t].func=func;
    #ifdef WORK_STEAL
        pt_args[t].all_pthread_args =  (void*)pt_args;
    #endif
        // fprintf(stderr, "t%d: %d-%d\n", t, pt_args[t].starti, pt_args[t].endi);
    }

    // create threads
    for (t = 0; t < core->opt.num_thread; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single,
                                (void*)(&pt_args[t]));
        NEG_CHK(ret);
    }

    // pthread joining
    for (t = 0; t < core->opt.num_thread; t++) {
        int ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }
}

/* process all reads in the given batch db */
void work_db(core_t* core, db_t* db, void (*func)(core_t*, db_t*, int)) {
    if (core->opt.num_thread == 1) {
        int32_t i = 0;
        for (i = 0; i < db->n_rec; i++) {
            func(core, db, i);
        }
    } else {
        pthread_db(core,db,func);
    }
}
