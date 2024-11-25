#include "toml.h"
#include "error.h"
#include "model_config.h"

void check_toml_table(toml_table_t *table) {
    if (!table) {
        ERROR("%s", "missing table in toml, please make sure your model version is supported");
        exit(EXIT_FAILURE);
    }
}

void check_toml_array(toml_array_t *arr) {
    if (!arr) {
        ERROR("%s", "missing array in toml, please make sure your model version is supported");
        exit(EXIT_FAILURE);
    }
}

void check_toml_datum(toml_datum_t datum) {
    if (!datum.ok) {
        ERROR("%s", "error reading field in toml, please make sure your model version is supported");
        exit(EXIT_FAILURE);
    }
}

CRFModelConfig load_crf_model_config(char *path) {
    FILE* fp;
    char errbuf[200];

    char *cpath = (char *)malloc(strlen(path) + 100);
    MALLOC_CHK(cpath);
    sprintf(cpath, "%s/config.toml", path);

    fp = fopen(cpath, "r");
    if (!fp) {
        ERROR("cannot open toml - %s: %s", cpath, strerror(errno));
        exit(EXIT_FAILURE);
    }

    toml_table_t *config_toml = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);
    check_toml_table(config_toml);

    CRFModelConfig config;
    config.qscale = 1.0f;
    config.qbias = 0.0f;

    if (toml_key_exists(config_toml, "qscore")) {
        toml_table_t *qscore = toml_table_in(config_toml, "qscore");
        check_toml_table(qscore);
        toml_datum_t qbias = toml_double_in(qscore, "bias");
        check_toml_datum(qbias);
        toml_datum_t qscale = toml_double_in(qscore, "scale");
        check_toml_datum(qscale);

        config.qbias = (float)qbias.u.d;
        config.qscale = (float)qscale.u.d;
    } else {
        // no qscore calibration found
    }

    config.conv = 4;
    config.insize = 0;
    config.stride = 1;
    config.bias = true;
    config.clamp = false;
    config.decomposition = false;

    // The encoder scale only appears in pre-v4 models.  In v4 models
    // the value of 1 is used.
    config.scale = 1.0f;

    toml_table_t *input = toml_table_in(config_toml, "input");
    check_toml_table(input);
    toml_datum_t num_features = toml_int_in(input, "features");
    check_toml_datum(num_features);
    config.num_features = num_features.u.i;

    toml_table_t *encoder = toml_table_in(config_toml, "encoder");
    check_toml_table(encoder);
    if (toml_key_exists(encoder, "type")) {
        // v4-type model
        toml_array_t *sublayers = toml_array_in(encoder, "sublayers");
        check_toml_array(sublayers);

        for (int i = 0; ; i++) {
            toml_table_t *segment = toml_table_at(sublayers, i);
            if (!segment) break;

            toml_datum_t type_dt = toml_string_in(segment, "type");
            check_toml_datum(type_dt);
            char *type = type_dt.u.s;

            if (strcmp(type, "convolution") == 0) {
                // Overall stride is the product of all conv layers' strides.
                toml_datum_t stride = toml_int_in(segment, "stride");
                check_toml_datum(stride);
                config.stride *= stride.u.i;
            } else if (strcmp(type, "lstm") == 0) {
                toml_datum_t insize = toml_int_in(segment, "insize");
                check_toml_datum(insize);
                config.insize = insize.u.i;
            } else if (strcmp(type, "linear") == 0) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                if (toml_key_exists(segment, "out_features")) {
                    toml_datum_t out_features = toml_int_in(segment, "out_features");
                    check_toml_datum(out_features);
                    config.out_features = out_features.u.i;
                    config.decomposition = true;
                } else {
                    config.decomposition = false;
                }
            } else if (strcmp(type, "clamp") == 0) {
                config.clamp = true;
            } else if (strcmp(type, "linearcrfencoder") == 0) {
                toml_datum_t blank_score = toml_double_in(segment, "blank_score");
                check_toml_datum(blank_score);
                config.blank_score = (float)blank_score.u.d;
            }

            free(type);
        }

        config.conv = 16;
        config.bias = config.insize > 128;
    } else {
        // pre-v4 model
        toml_datum_t stride = toml_int_in(encoder, "stride");
        check_toml_datum(stride);
        config.stride = stride.u.i;

        toml_datum_t features = toml_int_in(encoder, "features");
        check_toml_datum(features);
        config.insize = features.u.i;

        toml_datum_t blank_score = toml_double_in(encoder, "blank_score");
        check_toml_datum(blank_score);
        config.blank_score = (float)blank_score.u.d;

        toml_datum_t scale = toml_double_in(encoder, "scale");
        check_toml_datum(scale);
        config.scale = (float)scale.u.d;

        if (toml_key_exists(encoder, "first_conv_size")) {
            toml_datum_t conv = toml_int_in(encoder, "first_conv_size");
            check_toml_datum(conv);
            config.conv = conv.u.i;
        }
    }

    toml_table_t *global_norm = toml_table_in(config_toml, "global_norm");
    check_toml_table(global_norm);

    // Note that in v4 files state_len appears twice: under global_norm and under
    // linearcrfencoder.  We are ignoring the latter.
    toml_datum_t state_len = toml_int_in(global_norm, "state_len");
    check_toml_datum(state_len);
    config.state_len = state_len.u.i;

    // CUDA and CPU paths do not output explicit stay scores from the NN.
    config.outsize = pow(4, config.state_len) * 4;

    toml_free(config_toml);

    free(cpath);

    return config;
}