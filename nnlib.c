#include "nnlib.h"
#include <stdlib.h>

void add_fc_layer (neural_network_t *nn, int in_size, int out_size) {
    nn->n_layers++;

    realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    calloc(nn->layers[nn->n_layers - 1].output, out_size * sizeof(float));
}