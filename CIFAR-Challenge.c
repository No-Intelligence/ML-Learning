#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "nnlib.h"
#include <pthread.h>

float learning_rate = 0.001f;
float regularization_rate = 0.0005f;

int CIFAR_10_loader (char *filename, float *image_buffer, uint8_t *answer_buffer) {
    FILE *fp;
    fp = fopen(filename, "rb");
    if (fp == NULL) return -1;
    fseek(fp, 0, SEEK_SET);
    for (size_t i = 0; i < 10000; i++)
    {
        answer_buffer[i] = fgetc(fp);
        for (size_t j = 0; j < 32 * 32 * 3; j++)
        {
            image_buffer[i * 32*32*3 + j] = fgetc(fp)/255.0f;
        }
        
    }
    fclose(fp);
    return 0;
}

int CIFAR_10_batch_loader (float *image_buffer, uint8_t *answer_buffer) {
    FILE *fp;
    for (int loop = 0; loop < 5; loop++)
    {
        if (loop == 0) fp = fopen("data_batch_1.bin", "rb");
        else if (loop == 1) fp = fopen("data_batch_2.bin", "rb");
        else if (loop == 2) fp = fopen("data_batch_3.bin", "rb");
        else if (loop == 3) fp = fopen("data_batch_4.bin", "rb");
        else if (loop == 4) fp = fopen("data_batch_5.bin", "rb");
        if (fp == NULL) {
            printf("no file\n");
            return -1;
        }
        fseek(fp, 0, SEEK_SET);


        for (int i = 0; i < 10000; i++)
        {
            answer_buffer[i + loop*10000] = fgetc(fp);
            for (int j = 0; j < 32*32*3; j++)
            {
                image_buffer[i*32*32*3 + j + loop*32*32*3*10000] = fgetc(fp)/255.0f;
            }
        
        }

    }
    fclose(fp);
    return 0;
}

int find_max_index (float *array, int length) {
    int max_index = 0;
    float f = array[0];
    for (size_t i = 0; i < length; i++)
    {
        if (f <= array[i])
        {
            f = array[i];
            max_index = i;
        }
    }
    return max_index;
}

void show_progress(int current, int total, int width) {
    int percent = (current * 100) / total;
    int filled = (current * width) / total;
    
    printf("\r[");
    for (int i = 0; i < width; i++) {
        putchar(i < filled ? '=' : ' ');
    }
    printf("] %3d%%", percent);
    fflush(stdout);
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    neural_network_t *nn = alloc_neural_network();
    add_conv_layer(nn, 32, 32, 3, 3, 3, 32, 1, 0);
    add_activation_layer(nn, LAYER_GELU);
    add_pool_layer(nn, 30, 30, 32, 3, 3);
    add_flatten_layer(nn);
    add_fc_layer(nn, 10*10*32, 128);
    add_activation_layer(nn, LAYER_GELU);
    add_fc_layer(nn, 128, 10);
    add_activation_layer(nn, LAYER_SOFTMAX);
    
    parameter_initialize(nn);

    float *input_buffer = calloc(32 * 32 * 3 * 50000, sizeof(float));
    uint8_t *answer_label_buffer = calloc(50000, sizeof(uint8_t));
    float *input_one_image = calloc(32 * 32 * 3, sizeof(float));
    float answer_one_label[10];
    float output[10];
    int hit = 0;
    int t = 0;
    float loss;
    float *testbatch_image_buffer = calloc(32 * 32 * 3 * 10000, sizeof(float));
    uint8_t *testbatch_labed_buffer = calloc(10000, sizeof(float));

    //log file
    FILE *log;
    log = fopen("log.csv", "w");
    fprintf(log, "log file,train loss,test loss\n");

    if (CIFAR_10_batch_loader(input_buffer, answer_label_buffer) ==-1) return -1;
    if (CIFAR_10_loader("test_batch.bin", testbatch_image_buffer, testbatch_labed_buffer) == -1) return -1;
    printf("data loaded\n");

    for (int epoch = 0; epoch < 2; epoch++)
    {
        printf("training start\n");
        fprintf(log, "epoch:%d,", epoch + 1);
        loss = 0.0f;
        for (int i = 0; i < 50000; i++)
        {
            memcpy(input_one_image, &input_buffer[32*32*3 * i], 32*32*3 * sizeof(float));
            for (int j = 0; j < 10; j++)
            {
                answer_one_label[j] = 0.0 + (answer_label_buffer[i] == j);
            }
            loss += forward_pass(nn, input_one_image, answer_one_label);
            backward_pass(nn, input_one_image, answer_one_label);
            if ((i%50) == 49)
            {
                t++;
                update_param_adam(nn, learning_rate, regularization_rate, 0.9f, 0.999f, 1e-7, t, 50);
                show_progress(i+1, 50000, 20);
            }
            
        }
        loss /= 50000.0f;
        printf("\ntraining finished. loss:%f\n", loss);
        fprintf(log, "%f,", loss);
        
        hit = 0;
        printf("test start\n");
        loss = 0.0f;
        for (int i = 0; i < 10000; i++)
        {
            memcpy(input_one_image, &testbatch_image_buffer[32*32*3 * i], 32*32*3 * sizeof(float));
            for (int j = 0; j < 10; j++)
            {
                answer_one_label[j] = 0.0 + (testbatch_labed_buffer[i] == j);
            }
            loss += forward_pass(nn, input_one_image, answer_one_label);
            if (testbatch_labed_buffer[i] == find_max_index(nn->layers[nn->n_layers - 1].output, 10))
            {
                hit++;
            }
            if ((i%50) == 49)
            {
                show_progress(i+1, 10000, 20);
            }
        }
        loss /= 10000.0f;
        printf("\ntest finished. loss:%f\n", loss);
        fprintf(log, "%f\n", loss);
        printf("%f%%\n", ((float)hit / 10000.0f) * 100.0f);
        flush_grad(nn);
    }

    fclose(log);
    
    free(input_buffer);
    free(answer_label_buffer);
    free(input_one_image);
    free(testbatch_image_buffer);
    free(testbatch_labed_buffer);
    free_neural_network(nn);
    return 0;
}