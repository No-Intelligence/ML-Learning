#ifndef MLLIB_H
#define MLLIB_H

typedef struct {
    float *pre_activation;
    float *activation;
} layer_t;

typedef struct {
    float *weight;
    float *bias;
} parameter_t;

typedef struct {
    layer_t *layers;
    parameter_t *parameters;
} neural_network_t;

#endif