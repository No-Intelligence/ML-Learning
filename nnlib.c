#include "nnlib.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#define PI 3.14159265f

neural_network_t* alloc_neural_network (void) {
    neural_network_t *nn = calloc(1, sizeof(neural_network_t));
    nn->layers = NULL;
    return nn;
}

void add_fc_layer (neural_network_t *nn, int in_size, int out_size) {
    nn->n_layers++;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = LAYER_FC;
    nn->layers[nn->n_layers - 1].output_size = out_size;
    nn->layers[nn->n_layers - 1].delta = calloc(in_size, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(out_size, sizeof(float));

    nn->layers[nn->n_layers - 1].data.fc.in_size = in_size;
    nn->layers[nn->n_layers - 1].data.fc.out_size = out_size;
    nn->layers[nn->n_layers - 1].data.fc.weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.bias = calloc(out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.m_weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.v_weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.m_bias = calloc(out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.v_bias = calloc(out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.grad_weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.total_grad_weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.grad_bias = calloc(out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.total_grad_bias = calloc(out_size, sizeof(float));
}

void add_conv_layer (neural_network_t *nn, int in_height, int in_width, int in_channel, int filter_height, int filter_width, int n_filters, int filter_stride, int n_padding) {
    nn->n_layers++;
    int out_height = (in_height - filter_height + 2*n_padding) / filter_stride + 1;
    int out_width  = (in_width  - filter_width  + 2*n_padding) / filter_stride + 1;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = LAYER_CONV;
    nn->layers[nn->n_layers - 1].output_size = n_filters * out_height * out_width;
    nn->layers[nn->n_layers - 1].delta = calloc(n_filters * in_height * in_width, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(n_filters * out_height * out_width, sizeof(float));

    nn->layers[nn->n_layers - 1].data.conv.in_height = in_height;
    nn->layers[nn->n_layers - 1].data.conv.in_width = in_width;
    nn->layers[nn->n_layers - 1].data.conv.in_channel = in_channel;
    nn->layers[nn->n_layers - 1].data.conv.filter_height = filter_height;
    nn->layers[nn->n_layers - 1].data.conv.filter_width = filter_width;
    nn->layers[nn->n_layers - 1].data.conv.n_filters = n_filters;
    nn->layers[nn->n_layers - 1].data.conv.filter_stride = filter_stride;
    nn->layers[nn->n_layers - 1].data.conv.n_padding = n_padding;
    nn->layers[nn->n_layers - 1].data.conv.filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.bias = calloc(n_filters, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.m_filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.v_filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.grad_filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.total_grad_filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.m_bias = calloc(n_filters, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.v_bias = calloc(n_filters, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.grad_bias = calloc(n_filters, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.total_grad_bias = calloc(n_filters, sizeof(float));
}

void add_pool_layer (neural_network_t *nn, int in_height, int in_width, int in_channel, int kernel_height, int kernel_width) {
    nn->n_layers++;
    int out_height = in_height / kernel_height;
    int out_width = in_width / kernel_width;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = LAYER_POOL;
    nn->layers[nn->n_layers - 1].output_size = in_channel * out_height * out_width;
    nn->layers[nn->n_layers - 1].delta = calloc(in_channel * in_height * in_width, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(in_channel * out_height * out_width, sizeof(float));

    nn->layers[nn->n_layers - 1].data.pool.in_height = in_height;
    nn->layers[nn->n_layers - 1].data.pool.in_width = in_width;
    nn->layers[nn->n_layers - 1].data.pool.in_channel = in_channel;
    nn->layers[nn->n_layers - 1].data.pool.kernel_height = kernel_height;
    nn->layers[nn->n_layers - 1].data.pool.kernel_width = kernel_width;
    nn->layers[nn->n_layers - 1].data.pool.mask = calloc(in_channel * in_height * in_width, sizeof(uint8_t));
}

void add_activation_layer (neural_network_t *nn, layer_type_t activation) {
    nn->n_layers++;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = activation;
    nn->layers[nn->n_layers - 1].output_size = nn->layers[nn->n_layers - 2].output_size;
    nn->layers[nn->n_layers - 1].delta = calloc(nn->layers[nn->n_layers - 2].output_size, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(nn->layers[nn->n_layers - 2].output_size, sizeof(float));
}

void add_flatten_layer (neural_network_t *nn) {
    nn->n_layers++;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = LAYER_FLATTEN;
    nn->layers[nn->n_layers - 1].output_size = nn->layers[nn->n_layers - 2].output_size;
    nn->layers[nn->n_layers - 1].delta = calloc(nn->layers[nn->n_layers - 2].output_size, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(nn->layers[nn->n_layers - 2].output_size, sizeof(float));
}

void free_neural_network (neural_network_t *nn) {
    for (int i = (nn->n_layers - 1); i >= 0 ; i--)
    {
        switch (nn->layers[i].type)
        {
        case LAYER_FC:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            free(nn->layers[i].data.fc.bias);
            free(nn->layers[i].data.fc.grad_bias);
            free(nn->layers[i].data.fc.grad_weight);
            free(nn->layers[i].data.fc.m_bias);
            free(nn->layers[i].data.fc.m_weight);
            free(nn->layers[i].data.fc.total_grad_bias);
            free(nn->layers[i].data.fc.total_grad_weight);
            free(nn->layers[i].data.fc.v_bias);
            free(nn->layers[i].data.fc.v_weight);
            free(nn->layers[i].data.fc.weight);
            break;

        case LAYER_CONV:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            free(nn->layers[i].data.conv.filter);
            free(nn->layers[i].data.conv.bias);
            free(nn->layers[i].data.conv.m_filter);
            free(nn->layers[i].data.conv.v_filter);
            free(nn->layers[i].data.conv.grad_filter);
            free(nn->layers[i].data.conv.total_grad_filter);
            free(nn->layers[i].data.conv.m_bias);
            free(nn->layers[i].data.conv.v_bias);
            free(nn->layers[i].data.conv.grad_bias);
            free(nn->layers[i].data.conv.total_grad_bias);
            break;

        case LAYER_POOL:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            free(nn->layers[i].data.pool.mask);
            break;

        case LAYER_RELU:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;

        case LAYER_LEAKY_RELU:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;

        case LAYER_GELU:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;

        case LAYER_SOFTMAX:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;

        case LAYER_FLATTEN:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;
        }
    }
    free(nn->layers);
    free(nn);
}

void matrix_arr_mul (float * restrict output_arr, const float * restrict input_arr, const float * restrict matrix, int n_of_output_arr, int n_of_input_arr) {
    memset(output_arr, 0, n_of_output_arr * sizeof(float));
    
    for (int i = 0; i < n_of_output_arr; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < n_of_input_arr; j++)
        {
            sum += matrix[n_of_input_arr * i + j] * input_arr[j];
        }
        output_arr[i] = sum;
    }

}

void add_array (float *operated_arr, float *input_arr, int n_of_arr) {
    for (size_t i = 0; i < n_of_arr; i++)
    {
        operated_arr[i] += input_arr[i];
    }
    
}

void relu (float *input_arr, float *output_arr, int n_of_arr) {
    for (int i = 0; i < n_of_arr; i++)
    {
        output_arr[i] = input_arr[i] * (input_arr[i] > 0 ? 1.0f : 0.0f);
    }

}

void leaky_relu (float *input_arr, float *output_arr, int n_of_arr) {
    for (int i = 0; i < n_of_arr; i++)
    {
        output_arr[i] = input_arr[i] * (input_arr[i] > 0 ? 1.0f : 0.01f);
    }

}

void gelu (float *input_arr, float *output_arr, int n_of_arr) {
    for (int i = 0; i < n_of_arr; i++)
    {
        output_arr[i] = 0.5 * input_arr[i] * (1 + tanhf(sqrtf(2 / PI) * (input_arr[i] + 0.044715 * input_arr[i] * input_arr[i] * input_arr[i])));
    }
    
}

float extract_max (float *input_array, int n_of_input_arr) {
    float max = input_array[0];
    for (int i = 0; i < n_of_input_arr; i++)
    {
        if (max < input_array[i])
        {
            max = input_array[i];
        }
        
    }
    
    return max;
}

void softmax (float *input_arr, float *output_arr, int n_of_arr) {
    float max = 0.0, sum = 0.0;
    float *tmp = malloc(n_of_arr * sizeof(float));
    max = extract_max(input_arr, n_of_arr);
    for (int i = 0; i < n_of_arr; i++)
    {
        tmp[i] = expf(input_arr[i] - max);
        sum += tmp[i];
    }
    for (int j = 0; j < n_of_arr; j++)
    {
        output_arr[j] = tmp[j] / sum;
    }
    free(tmp);

}

void forward_convolution (const float * restrict input, const float * restrict filter, float * restrict output, int n_input_height, int n_input_width, int n_input_channel, int filter_height, int filter_width, int n_filters, int stride, const float * restrict bias) {
    int n_output_height = (n_input_height - filter_height) / stride + 1;
    int n_output_width = (n_input_width - filter_width) / stride + 1;

    memset(output, 0, n_filters * n_output_height * n_output_width * sizeof(float));

    for (int n = 0; n < n_filters; n++)
    {
        for (int c = 0; c < n_input_channel; c++)
        {
            for (int fh = 0; fh < filter_height; fh++)
            {
                for (int fw = 0; fw < filter_width; fw++)
                {
                    float w = filter[n * n_input_channel * filter_height * filter_width + c * filter_height * filter_width + fh * filter_width + fw];
                    for (int oh = 0; oh < n_output_height; oh++)
                    {
                        int ih = oh * stride + fh;
                        int out_off = n * n_output_height * n_output_width + oh * n_output_width;
                        int in_off = c * n_input_height * n_input_width + ih * n_input_width + fw;
                        for (int ow = 0; ow < n_output_width; ow++)
                        {
                            output[out_off + ow] += w * input[in_off + ow * stride];
                        }
                    }
                }
            }
        }
    }
    for (int n = 0; n < n_filters; n++)
    {
        float b = bias[n];
        int out_size = n_output_height * n_output_width;
        int out_off = n * out_size;
        for (int p = 0; p < out_size; p++)
            output[out_off + p] += b;
    }
}

void forward_maxpool(float *input, float *output, int n_channels, int in_height, int in_width, int kernel_height, int kernel_width, uint8_t *mask) {
    //standby
    int out_h = in_height / kernel_height;
    int out_w = in_width / kernel_width;
    memset(mask, 0, n_channels * in_height * in_width * sizeof(uint8_t));

    for (size_t c = 0; c < n_channels; c++)
    {
        for (size_t oh = 0; oh < out_h; oh++)
        {
            for (size_t ow = 0; ow < out_w; ow++)
            {
                float max = -FLT_MAX;
                int max_indics = c * in_height * in_width + (oh*kernel_height)*in_width + (ow*kernel_width);
                for (size_t kh = 0; kh < kernel_height; kh++)
                {
                    for (size_t kw = 0; kw < kernel_width; kw++)
                    {
                        float value = input[c * in_height * in_width + (oh*kernel_height+kh)*in_width + (ow*kernel_width+kw)];
                        if (value > max)
                        {
                            max = value;
                            max_indics = c * in_height * in_width + (oh*kernel_height+kh)*in_width + (ow*kernel_width+kw);
                        }
                        
                    }
                    
                }
                output[c * out_h * out_w + oh * out_w + ow] = max;
                mask[max_indics] = 1;
            }
            
        }
        
    }
    
}

void forward_pass (neural_network_t *nn, float *input) {
    float *current_input = input;
    for (size_t i = 0; i < nn->n_layers; i++)
    {
        switch (nn->layers[i].type)
        {
        case LAYER_FC:
            matrix_arr_mul(nn->layers[i].output, current_input, nn->layers[i].data.fc.weight, nn->layers[i].data.fc.out_size, nn->layers[i].data.fc.in_size);
            add_array(nn->layers[i].output, nn->layers[i].data.fc.bias, nn->layers[i].data.fc.out_size);
            break;

        case LAYER_CONV:
            forward_convolution(current_input, nn->layers[i].data.conv.filter, nn->layers[i].output, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.in_channel, nn->layers[i].data.conv.filter_height, nn->layers[i].data.conv.filter_width, nn->layers[i].data.conv.n_filters, nn->layers[i].data.conv.filter_stride, nn->layers[i].data.conv.bias);
            break;

        case LAYER_POOL:
            forward_maxpool(current_input, nn->layers[i].output, nn->layers[i].data.pool.in_channel, nn->layers[i].data.pool.in_height, nn->layers[i].data.pool.in_width, nn->layers[i].data.pool.kernel_height, nn->layers[i].data.pool.kernel_width, nn->layers[i].data.pool.mask);
            break;

        case LAYER_RELU:
            relu(current_input, nn->layers[i].output, nn->layers[i].output_size);
            break;

        case LAYER_LEAKY_RELU:
            leaky_relu(current_input, nn->layers[i].output, nn->layers[i].output_size);
            break;

        case LAYER_GELU:
            gelu(current_input, nn->layers[i].output, nn->layers[i].output_size);
            break;

        case LAYER_SOFTMAX:
            softmax(current_input, nn->layers[i].output, nn->layers[i].output_size);
            break;

        case LAYER_FLATTEN:
            memcpy(nn->layers[i].output, current_input, nn->layers[i].output_size * sizeof(float));
            break;
        }
        current_input = nn->layers[i].output;
    }
    
}

void compute_output_softmax_delta (float *output_delta, float *output_layer_activation, float *answer_arr, int n_of_arr) {
    for (size_t i = 0; i < n_of_arr; i++)
    {
        output_delta[i] = output_layer_activation[i] - answer_arr[i];
    }
    
}

void compute_backward_fc (float * restrict output_delta, const float * restrict current_delta, const float * restrict weight, int n_output_delta, int n_current_delta) {
    memset(output_delta, 0, n_output_delta * sizeof(float));
    for (int j = 0; j < n_current_delta; j++)
    {
        float cd = current_delta[j];
        int w_base = j * n_output_delta;
        for (int i = 0; i < n_output_delta; i++)
        {
            output_delta[i] += cd * weight[w_base + i];
        }
    }
}

void compute_weight_grad (const float * restrict z_delta, const float * restrict previous_activation_arr, float * restrict output_arr, int n_of_output, int n_of_input) {
    for (int i = 0; i < n_of_output; i++)
    {
        float zd = z_delta[i];
        int row = i * n_of_input;
        for (int j = 0; j < n_of_input; j++)
        {
            output_arr[row + j] = zd * previous_activation_arr[j];
        }
    }
}

void compute_bias_grad (float *output_bias_grad, float *delta, int n_of_arr) {
    memcpy(output_bias_grad, delta, n_of_arr * sizeof(float));
}

void compute_backward_maxpool (float *computed_delta, float *current_delta, uint8_t *mask, int n_channels, int in_h, int in_w) {
    memset(computed_delta, 0, n_channels * in_h * in_w * sizeof(float));

    int index = 0;
    for (size_t c = 0; c < n_channels; c++)
    {
        for (size_t h = 0; h < in_h; h++)
        {
            for (size_t w = 0; w < in_w; w++)
            {
                if (mask[in_h * in_w * c + in_w * h + w] == 1)
                {
                    computed_delta[in_h * in_w * c + in_w * h + w] = current_delta[index];
                    index++;
                }
                
            }
            
        }
        
    }
    
}

void compute_backward_conv (float * restrict computed_delta, float * restrict grad_filter, float * restrict grad_bias, float *activation, const float * restrict current_delta, const float * restrict filter, const float * restrict input, int n_input_height, int n_input_width, int filter_height, int filter_width, int n_filters, int in_channel, int in_h, int in_w, int stride) {
    int n_output_height = (n_input_height - filter_height) / stride + 1;
    int n_output_width = (n_input_width - filter_width) / stride + 1;

    memset(computed_delta, 0, in_channel * n_input_height * n_input_width * sizeof(float));
    memset(grad_filter, 0, n_filters * in_channel * filter_height * filter_width * sizeof(float));

    // grad_filter: accumulate over (oh, ow) with ow innermost for contiguous output access
    for (int n = 0; n < n_filters; n++)
    {
        for (int c = 0; c < in_channel; c++)
        {
            for (int fh = 0; fh < filter_height; fh++)
            {
                for (int fw = 0; fw < filter_width; fw++)
                {
                    float sum = 0.0f;
                    int gf_idx = n * in_channel * filter_height * filter_width + c * filter_height * filter_width + fh * filter_width + fw;
                    for (int oh = 0; oh < n_output_height; oh++)
                    {
                        int ih = oh * stride + fh;
                        int cd_base = n * n_output_height * n_output_width + oh * n_output_width;
                        int in_base = c * n_input_height * n_input_width + ih * n_input_width + fw;
                        for (int ow = 0; ow < n_output_width; ow++)
                        {
                            sum += current_delta[cd_base + ow] * input[in_base + ow * stride];
                        }
                    }
                    grad_filter[gf_idx] = sum;
                }
            }
        }
    }

    // computed_delta: propagate error to input, with ow innermost
    for (int n = 0; n < n_filters; n++)
    {
        for (int c = 0; c < in_channel; c++)
        {
            for (int fh = 0; fh < filter_height; fh++)
            {
                for (int oh = 0; oh < n_output_height; oh++)
                {
                    int ih = oh * stride + fh;
                    for (int fw = 0; fw < filter_width; fw++)
                    {
                        float w = filter[n * in_channel * filter_height * filter_width + c * filter_height * filter_width + fh * filter_width + fw];
                        int cd_base = n * n_output_height * n_output_width + oh * n_output_width;
                        int out_base = c * n_input_height * n_input_width + ih * n_input_width + fw;
                        for (int ow = 0; ow < n_output_width; ow++)
                        {
                            computed_delta[out_base + ow * stride] += current_delta[cd_base + ow] * w;
                        }
                    }
                }
            }
        }
    }

    // grad_bias: sum current_delta over spatial positions
    for (int n = 0; n < n_filters; n++)
    {
        float sum = 0.0f;
        int cd_base = n * n_output_height * n_output_width;
        for (int oh = 0; oh < n_output_height; oh++)
        {
            for (int ow = 0; ow < n_output_width; ow++)
            {
                sum += current_delta[cd_base + oh * n_output_width + ow];
            }
        }
        grad_bias[n] = sum;
    }
}

void backward_pass (neural_network_t *nn, float *input, float *answer) {
    float *current_delta;
    switch (nn->layers[nn->n_layers - 1].type)
    {
    case LAYER_SOFTMAX:
        compute_output_softmax_delta(nn->layers[nn->n_layers - 1].delta, nn->layers[nn->n_layers - 1].output, answer, nn->layers[nn->n_layers - 1].output_size);
        break;
    }
    current_delta = nn->layers[nn->n_layers - 1].delta;

    for (int i = (nn->n_layers - 2); i >= 0; i--)
    {
        switch (nn->layers[i].type)
        {
        case LAYER_FC:
            if (i == 0)
            {
                compute_weight_grad(current_delta, input, nn->layers[i].data.fc.grad_weight, nn->layers[i].output_size, nn->layers[i].data.fc.in_size);
            }
            else
            {
                compute_weight_grad(current_delta, nn->layers[i - 1].output, nn->layers[i].data.fc.grad_weight, nn->layers[i].output_size, nn->layers[i - 1].output_size);
            }
            compute_bias_grad(nn->layers[i].data.fc.grad_bias, current_delta, nn->layers[i].output_size);
            if (i > 0)
            {
                compute_backward_fc(nn->layers[i].delta, current_delta, nn->layers[i].data.fc.weight, nn->layers[i].data.fc.in_size, nn->layers[i].data.fc.out_size);
            }
            for (size_t j = 0; j < nn->layers[i].data.fc.in_size * nn->layers[i].data.fc.out_size; j++)
            {
                nn->layers[i].data.fc.total_grad_weight[j] += nn->layers[i].data.fc.grad_weight[j];
            }
            for (size_t j = 0; j < nn->layers[i].data.fc.out_size; j++)
            {
                nn->layers[i].data.fc.total_grad_bias[j] += nn->layers[i].data.fc.grad_bias[j];
            }
            break;

        case LAYER_CONV:
            if (i == 0)
            {
                compute_backward_conv(nn->layers[i].delta, nn->layers[i].data.conv.grad_filter, nn->layers[i].data.conv.grad_bias, nn->layers[i].output, current_delta, nn->layers[i].data.conv.filter, input, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.filter_height, nn->layers[i].data.conv.filter_width, nn->layers[i].data.conv.n_filters, nn->layers[i].data.conv.in_channel, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.filter_stride);
            }
            else
            {
                compute_backward_conv(nn->layers[i].delta, nn->layers[i].data.conv.grad_filter, nn->layers[i].data.conv.grad_bias, nn->layers[i].output, current_delta, nn->layers[i].data.conv.filter, nn->layers[i - 1].output, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.filter_height, nn->layers[i].data.conv.filter_width, nn->layers[i].data.conv.n_filters, nn->layers[i].data.conv.in_channel, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.filter_stride);
            }
            for (size_t j = 0; j < nn->layers[i].data.conv.filter_height * nn->layers[i].data.conv.filter_width * nn->layers[i].data.conv.n_filters * nn->layers[i].data.conv.in_channel; j++)
            {
                nn->layers[i].data.conv.total_grad_filter[j] += nn->layers[i].data.conv.grad_filter[j];
            }
            for (size_t j = 0; j < nn->layers[i].data.conv.n_filters; j++)
            {
                nn->layers[i].data.conv.total_grad_bias[j] += nn->layers[i].data.conv.grad_bias[j];
            }
            break;

        case LAYER_POOL:
            compute_backward_maxpool(nn->layers[i].delta, current_delta, nn->layers[i].data.pool.mask, nn->layers[i].data.pool.in_channel, nn->layers[i].data.pool.in_height, nn->layers[i].data.pool.in_width);
            break;

        case LAYER_RELU:
            for (size_t j = 0; j < nn->layers[i].output_size; j++)
            {
                nn->layers[i].delta[j] = current_delta[j] * (nn->layers[i - 1].output[j] > 0);
            }
            break;

        case LAYER_LEAKY_RELU:
            for (size_t j = 0; j < nn->layers[i].output_size; j++)
            {
                nn->layers[i].delta[j] = current_delta[j] * (nn->layers[i - 1].output[j] > 0) + 0.01 * current_delta[j] * (nn->layers[i - 1].output[j] <= 0);
            }
            break;

        case LAYER_GELU:
            for (size_t j = 0; j < nn->layers[i].output_size; j++)
            {
                nn->layers[i].delta[j] = current_delta[j] * (0.5 * (1 + tanhf(sqrtf(2 / PI) * (nn->layers[i - 1].output[j] + 0.044715 * nn->layers[i - 1].output[j] * nn->layers[i - 1].output[j] * nn->layers[i - 1].output[j]))) + 0.5 * nn->layers[i - 1].output[j] / coshf(sqrtf(2 / PI) * (nn->layers[i - 1].output[j] + 0.044715 * nn->layers[i - 1].output[j] * nn->layers[i - 1].output[j] * nn->layers[i - 1].output[j])) / coshf(sqrtf(2 / PI) * (nn->layers[i - 1].output[j] + 0.044715 * nn->layers[i - 1].output[j] * nn->layers[i - 1].output[j] * nn->layers[i - 1].output[j])) * sqrtf(2 / PI) * (1 + 0.134145 * nn->layers[i - 1].output[j] * nn->layers[i - 1].output[j]));
            }
            break;

        case LAYER_SOFTMAX:
            break;

        case LAYER_FLATTEN:
            memcpy(nn->layers[i].delta, current_delta, nn->layers[i].output_size * sizeof(float));
            break;
        }
        current_delta = nn->layers[i].delta;
    }
    
}

void parameter_initialize (neural_network_t *nn) {
    for (int i = 0; i < nn->n_layers; i++) {
        switch (nn->layers[i].type) {
        case LAYER_FC:{
            float std = sqrtf(2.0f / (float)nn->layers[i].data.fc.in_size);
            for (int j = 0; j < nn->layers[i].data.fc.in_size * nn->layers[i].data.fc.out_size; j++) {
                nn->layers[i].data.fc.weight[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                nn->layers[i].data.fc.weight[j] *= std * sqrtf(3.0f);
            }
            break;
        }
        case LAYER_CONV:{
            float limit = sqrtf(2.0f / (nn->layers[i].data.conv.in_channel * nn->layers[i].data.conv.filter_height * nn->layers[i].data.conv.filter_width));
            for (int j = 0; j < nn->layers[i].data.conv.in_channel * nn->layers[i].data.conv.n_filters * nn->layers[i].data.conv.filter_height * nn->layers[i].data.conv.filter_width; j++) {
                nn->layers[i].data.conv.filter[j] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
            }
            break;
        }
        default:
            break;
        }
    }
}

void update_param_adam (neural_network_t *nn, float lr, float weight_decay, float beta1, float beta2, float eps, int t, int batch_size) {
    for (size_t layer = 0; layer < nn->n_layers - 1; layer++)
    {
        switch (nn->layers[layer].type)
        {
        case LAYER_FC:{
            float bc = lr * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));

            for (size_t j = 0; j < nn->layers[layer].data.fc.in_size * nn->layers[layer].data.fc.out_size; j++)
            {
                float g = (nn->layers[layer].data.fc.total_grad_weight[j] / batch_size);
                nn->layers[layer].data.fc.m_weight[j] = beta1 * nn->layers[layer].data.fc.m_weight[j] + (1 - beta1) * g;
                nn->layers[layer].data.fc.v_weight[j] = beta2 * nn->layers[layer].data.fc.v_weight[j] + (1 - beta2) * g * g;
                nn->layers[layer].data.fc.weight[j] -= bc * nn->layers[layer].data.fc.m_weight[j] / (sqrtf(nn->layers[layer].data.fc.v_weight[j]) + eps) + lr * weight_decay * nn->layers[layer].data.fc.weight[j];
            }
            memset(nn->layers[layer].data.fc.total_grad_weight, 0, nn->layers[layer].data.fc.in_size * nn->layers[layer].data.fc.out_size * sizeof(float));
            for (size_t j = 0; j < nn->layers[layer].data.fc.out_size; j++)
            {
                float g = (nn->layers[layer].data.fc.total_grad_bias[j] / batch_size);
                nn->layers[layer].data.fc.m_bias[j] = beta1 * nn->layers[layer].data.fc.m_bias[j] + (1 - beta1) * g;
                nn->layers[layer].data.fc.v_bias[j] = beta2 * nn->layers[layer].data.fc.v_bias[j] + (1 - beta2) * g * g;
                nn->layers[layer].data.fc.bias[j] -= bc * nn->layers[layer].data.fc.m_bias[j] / (sqrtf(nn->layers[layer].data.fc.v_bias[j]) + eps);
            }
            memset(nn->layers[layer].data.fc.total_grad_bias, 0, nn->layers[layer].data.fc.out_size * sizeof(float));
            break;
        }

        case LAYER_CONV:{
            float bc = lr * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));

            for (size_t i = 0; i < nn->layers[layer].data.conv.n_filters * nn->layers[layer].data.conv.in_channel * nn->layers[layer].data.conv.filter_height * nn->layers[layer].data.conv.filter_width; i++)
            {
                float g = (nn->layers[layer].data.conv.grad_filter[i] / batch_size);
                nn->layers[layer].data.conv.m_filter[i] = beta1 * nn->layers[layer].data.conv.m_filter[i] + (1 - beta1) * g;
                nn->layers[layer].data.conv.v_filter[i] = beta2 * nn->layers[layer].data.conv.v_filter[i] + (1 - beta2) * g * g;
                nn->layers[layer].data.conv.filter[i] -= bc * nn->layers[layer].data.conv.m_filter[i] / (sqrtf(nn->layers[layer].data.conv.v_filter[i]) + eps) + lr * weight_decay * nn->layers[layer].data.conv.filter[i];
            }
            for (size_t i = 0; i < nn->layers[layer].data.conv.n_filters; i++)
            {
                float g = (nn->layers[layer].data.conv.grad_bias[i] / batch_size);
                nn->layers[layer].data.conv.m_bias[i] = beta1 * nn->layers[layer].data.conv.m_bias[i] + (1 - beta1) * g;
                nn->layers[layer].data.conv.v_bias[i] = beta2 * nn->layers[layer].data.conv.v_bias[i] + (1 - beta2) * g * g;
                nn->layers[layer].data.conv.bias[i] -= bc * nn->layers[layer].data.conv.m_bias[i] / (sqrtf(nn->layers[layer].data.conv.v_bias[i]) + eps);
            }
            break;
        }
        
        
        default:
            break;
        }
    }
    
}

void flush_grad (neural_network_t *nn) {
    for (int i = 0; i < nn->n_layers; i++)
    {
        switch (nn->layers[i].type)
        {
        case LAYER_FC:
            memset(nn->layers[i].data.fc.total_grad_weight, 0, nn->layers[i].data.fc.in_size * nn->layers[i].data.fc.out_size * sizeof(float));
            memset(nn->layers[i].data.fc.total_grad_bias, 0, nn->layers[i].data.fc.out_size * sizeof(float));
            break;

        case LAYER_CONV:
            memset(nn->layers[i].data.conv.total_grad_filter, 0, nn->layers[i].data.conv.filter_height * nn->layers[i].data.conv.filter_width * nn->layers[i].data.conv.in_channel * nn->layers[i].data.conv.n_filters * sizeof(float));
            memset(nn->layers[i].data.conv.total_grad_bias, 0, nn->layers[i].data.conv.n_filters *  sizeof(float));
            break;
        
        default:
            break;
        }
    }
    
}