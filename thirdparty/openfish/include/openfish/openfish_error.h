/* @file error.h
**
** error checking macros/functions and error messages

MIT License

Copyright (c) 2018  Hasindu Gamaarachchi (hasindu@unsw.edu.au)
Copyright (c) 2018  Thomas Daniell

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

#ifndef OPENFISH_ERROR_H
#define OPENFISH_ERROR_H

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define openfish_log_level get_openfish_log_level()

// the level of verbosity in the log printed to the standard error
enum openfish_log_level_opt {
    OPENFISH_LOG_OFF,      // nothing at all
    OPENFISH_LOG_ERR,      // error messages
    OPENFISH_LOG_WARN,     // warning and error messages
    OPENFISH_LOG_INFO,     // information, warning and error messages
    OPENFISH_LOG_VERB,     // verbose, information, warning and error messages
    OPENFISH_LOG_DBUG,     // debugging, verbose, information, warning and error messages
    OPENFISH_LOG_TRAC      // tracing, debugging, verbose, information, warning and error messages
};

enum openfish_log_level_opt get_openfish_log_level();
void set_openfish_log_level(enum openfish_log_level_opt level);

#define OPENFISH_DEBUG_PREFIX "[OPENFISH_DEBUG] %s: " /* TODO function before debug */
#define OPENFISH_VERBOSE_PREFIX "[OPENFISH_INFO] %s: "
#define OPENFISH_INFO_PREFIX "[%s::OPENFISH_INFO]\033[1;34m "
#define OPENFISH_WARNING_PREFIX "[%s::OPENFISH_WARNING]\033[1;33m "
#define OPENFISH_ERROR_PREFIX "[%s::OPENFISH_ERROR]\033[1;31m "
#define OPENFISH_NO_COLOUR "\033[0m"

#define OPENFISH_LOG_TRACE(msg, ...) { \
    if (openfish_log_level >= OPENFISH_LOG_TRAC) { \
        fprintf(stderr, OPENFISH_DEBUG_PREFIX msg \
                " At %s:%d\n", \
                __func__, __VA_ARGS__, __FILE__, __LINE__ - 1); \
    } \
}

#define OPENFISH_LOG_DEBUG(msg, ...) { \
    if (openfish_log_level >= OPENFISH_LOG_DBUG) { \
        fprintf(stderr, OPENFISH_DEBUG_PREFIX msg \
                " At %s:%d\n", \
                __func__, __VA_ARGS__, __FILE__, __LINE__ - 1); \
    } \
}

#define OPENFISH_VERBOSE(msg, ...) { \
    if (openfish_log_level >= OPENFISH_LOG_VERB) { \
        fprintf(stderr, OPENFISH_VERBOSE_PREFIX msg "\n", __func__, __VA_ARGS__); \
    } \
}

#define OPENFISH_INFO(msg, ...) { \
    if (openfish_log_level >= OPENFISH_LOG_INFO) { \
        fprintf(stderr, OPENFISH_INFO_PREFIX msg OPENFISH_NO_COLOUR "\n", __func__, __VA_ARGS__); \
    } \
}

#define OPENFISH_WARNING(msg, ...) { \
    if (openfish_log_level >= OPENFISH_LOG_WARN) { \
        fprintf(stderr, OPENFISH_WARNING_PREFIX msg OPENFISH_NO_COLOUR \
                " At %s:%d\n", \
                __func__, __VA_ARGS__, __FILE__, __LINE__ - 1); \
    } \
}

#define OPENFISH_ERROR(msg, ...) { \
    if (openfish_log_level >= OPENFISH_LOG_ERR) { \
        fprintf(stderr, OPENFISH_ERROR_PREFIX msg OPENFISH_NO_COLOUR \
                " At %s:%d\n", \
                __func__, __VA_ARGS__, __FILE__, __LINE__ - 1); \
    } \
}

#ifdef __cplusplus
}
#endif

#endif
