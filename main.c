#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct
{
    int block_size;
    int vocab_size;

    int dim;
    int ff_dim;

    int layers_num;

    int num_heads;
    int head_dim; // todo: remove

} TransformerConfig;

typedef struct
{
    float *wte; // (vocab_size, dim)
    float *wpe; // (block_size, dim)

    float *w_attn_qkv; // (layer_idxs, 3 * num_heads * head_dim, dim)
    float *b_attn_qkv; // (layer_idxs, 3 * num_heads * head_dim)

    float *w_attn_out; // (layer_idxs, dim, dim)
    float *b_attn_out; // (layer_idxs, dim)

    float *w_dense_0; // (layer_idxs, ff_dim, dim)
    float *b_dense_0; // (layer_idxs, ff_dim)

    float *w_dense_1; // (layer_idxs, dim, ff_dim)
    float *b_dense_1; // (layer_idxs, dim)

    float *ln_gamma; // (layer_idxs, 2, dim)
    float *ln_beta;  // (layer_idxs, 2, dim)

    float *w_out; // (dim, vocab_size)
} TransformerWeights;

typedef struct
{
    float *x; // (dim,)

    float *q; // (dim,)

    float *k_cache; // (layer_idxs, block_size, dim)
    float *v_cache; // (layer_idxs, block_size, dim)

    float *residual; // (dim,)
    float *dense_h;  // (ff_dim,)

    float *logits; // (vocab_size,)
} TransformerState;

typedef struct
{
    TransformerConfig config;

    TransformerWeights weights;
    TransformerState state;
} Transformer;

typedef enum
{
    WEIGHT_INIT_RANDOM,
    WEIGHT_INIT_CHECKPOINT
} WeightInit;

void load_weights(TransformerConfig *config, TransformerWeights *weights, WeightInit weight_init)
{
    weights->wte = calloc((config->vocab_size) * (config->dim), sizeof(float));
    weights->wpe = calloc((config->block_size) * (config->dim), sizeof(float));

    weights->w_attn_qkv = calloc((config->layers_num) * (3 * config->num_heads * config->head_dim) * (config->dim), sizeof(float));
    weights->b_attn_qkv = calloc((config->layers_num) * (3 * config->num_heads * config->head_dim), sizeof(float));

    weights->w_attn_out = calloc((config->layers_num) * (config->dim) * (config->dim), sizeof(float));
    weights->b_attn_out = calloc((config->layers_num) * (config->dim), sizeof(float));

    weights->w_dense_0 = calloc((config->layers_num) * (config->ff_dim) * (config->dim), sizeof(float));
    weights->b_dense_0 = calloc((config->layers_num) * (config->ff_dim), sizeof(float));

    weights->w_dense_1 = calloc((config->layers_num) * (config->dim) * (config->ff_dim), sizeof(float));
    weights->b_dense_1 = calloc((config->layers_num) * (config->dim), sizeof(float));

    weights->ln_gamma = calloc((config->layers_num) * (2) * (config->dim), sizeof(float));
    weights->ln_beta = calloc((config->layers_num) * (2) * (config->dim), sizeof(float));

    weights->w_out = calloc((config->dim) * (config->vocab_size), sizeof(float));

    if (weights->wte == NULL || weights->wpe == NULL ||
        weights->w_attn_qkv == NULL || weights->w_attn_out == NULL ||
        weights->w_dense_0 == NULL || weights->b_dense_0 == NULL ||
        weights->w_dense_1 == NULL || weights->b_dense_1 == NULL ||
        weights->ln_gamma == NULL || weights->ln_beta == NULL ||
        weights->w_out == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    switch (weight_init)
    {
    case WEIGHT_INIT_RANDOM:
    {
        srand(0);

#define FILL_RANDOM_FLOATS(vector, dim) \
    for (int i = 0; i < dim; i++)       \
        vector[i] = (2.0f * ((float)rand() / RAND_MAX)) - 1.0f;

        FILL_RANDOM_FLOATS(weights->wte, (config->vocab_size) * (config->dim));
        FILL_RANDOM_FLOATS(weights->wpe, (config->block_size) * (config->dim));

        FILL_RANDOM_FLOATS(weights->w_attn_qkv, (config->layers_num) * (3 * config->num_heads * config->head_dim) * (config->head_dim));
        FILL_RANDOM_FLOATS(weights->b_attn_qkv, (config->layers_num) * (3 * config->num_heads * config->head_dim));

        FILL_RANDOM_FLOATS(weights->w_attn_out, (config->layers_num) * (config->dim) * (config->dim));
        FILL_RANDOM_FLOATS(weights->b_attn_out, (config->layers_num) * (config->dim));

        FILL_RANDOM_FLOATS(weights->w_dense_0, (config->layers_num) * (config->ff_dim) * (config->dim));
        FILL_RANDOM_FLOATS(weights->b_dense_0, (config->layers_num) * (config->ff_dim));

        FILL_RANDOM_FLOATS(weights->w_dense_1, (config->layers_num) * (config->dim) * (config->ff_dim));
        FILL_RANDOM_FLOATS(weights->b_dense_1, (config->layers_num) * (config->dim));

        FILL_RANDOM_FLOATS(weights->ln_gamma, (config->layers_num) * (2) * (config->dim));
        FILL_RANDOM_FLOATS(weights->ln_beta, (config->layers_num) * (2) * (config->dim));

        FILL_RANDOM_FLOATS(weights->w_out, (config->dim) * (config->vocab_size));

#undef FILL_RANDOM_FLOATS

        break;
    }
    case WEIGHT_INIT_CHECKPOINT:
    {
        FILE *file = fopen("weights.bin", "rb");

        if (file == NULL)
        {
            fprintf(stderr, "Error opening file\n");
            exit(EXIT_FAILURE);
        }

        int wte_size = (config->vocab_size) * (config->dim);
        int wpe_size = (config->block_size) * (config->dim);

        int w_attn_qkv_size = (3 * config->num_heads * config->head_dim) * (config->num_heads * config->head_dim);
        int b_attn_qkv_size = (3 * config->num_heads * config->head_dim);

        int w_attn_out_size = (config->dim) * (config->dim);
        int b_attn_out_size = (config->dim);

        int w_dense_0_size = (config->ff_dim) * (config->dim);
        int b_dense_0_size = (config->ff_dim);

        int w_dense_1_size = (config->dim) * (config->ff_dim);
        int b_dense_1_size = (config->dim);

        int w_out_size = (config->dim) * (config->vocab_size);

        fread(weights->wte, sizeof(float), wte_size, file);
        fread(weights->wpe, sizeof(float), wpe_size, file);

        for (int layer_idx = 0; layer_idx < config->layers_num; layer_idx++)
        {
            fread(weights->w_attn_qkv + (layer_idx * w_attn_qkv_size), sizeof(float), w_attn_qkv_size, file);
            fread(weights->b_attn_qkv + (layer_idx * b_attn_qkv_size), sizeof(float), b_attn_qkv_size, file);

            fread(weights->w_attn_out + (layer_idx * w_attn_out_size), sizeof(float), w_attn_out_size, file);
            fread(weights->b_attn_out + (layer_idx * b_attn_out_size), sizeof(float), b_attn_out_size, file);

            fread(weights->w_dense_0 + (layer_idx * w_dense_0_size), sizeof(float), w_dense_0_size, file);
            fread(weights->b_dense_0 + (layer_idx * b_dense_0_size), sizeof(float), b_dense_0_size, file);

            fread(weights->w_dense_1 + (layer_idx * w_dense_1_size), sizeof(float), w_dense_1_size, file);
            fread(weights->b_dense_1 + (layer_idx * b_dense_1_size), sizeof(float), b_dense_1_size, file);

            fread(weights->ln_gamma + (layer_idx * 2 * config->dim), sizeof(float), 2 * config->dim, file);
            fread(weights->ln_beta + (layer_idx * 2 * config->dim), sizeof(float), 2 * config->dim, file);
        }

        fread(weights->w_out, sizeof(float), w_out_size, file);

        fclose(file);

        break;
    }
    default:
        fprintf(stderr, "WeightInit %d is not yet supported\n", weight_init);
        exit(EXIT_FAILURE);
    }
}

void load_state(TransformerConfig *config, TransformerState *state)
{
    state->x = calloc((config->dim), sizeof(float));

    state->q = calloc((config->dim), sizeof(float));

    state->k_cache = calloc((config->layers_num) * (config->block_size) * (config->dim), sizeof(float));
    state->v_cache = calloc((config->layers_num) * (config->block_size) * (config->dim), sizeof(float));

    state->residual = calloc((config->dim), sizeof(float));
    state->dense_h = calloc((config->ff_dim), sizeof(float));

    state->logits = calloc((config->vocab_size), sizeof(float));

    if (state->x == NULL || state->q == NULL ||
        state->k_cache == NULL || state->v_cache == NULL ||
        state->residual == NULL || state->dense_h == NULL ||
        state->logits == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

void unload_weights(TransformerWeights *weights)
{
    free(weights->wte);
    free(weights->wpe);

    free(weights->w_attn_qkv);
    free(weights->b_attn_qkv);

    free(weights->w_attn_out);
    free(weights->b_attn_out);

    free(weights->w_dense_0);
    free(weights->b_dense_0);

    free(weights->w_dense_1);
    free(weights->b_dense_1);

    free(weights->ln_gamma);
    free(weights->ln_beta);

    free(weights->w_out);
}

void unload_state(TransformerState *state)
{
    free(state->x);

    free(state->q);

    free(state->k_cache);
    free(state->v_cache);

    free(state->residual);
    free(state->dense_h);
    free(state->logits);
}

void print_vector(float *vector, int dim)
{
    int buffer_size = dim * 8 + 3;
    char buffer[buffer_size];

    int offset = 0;

    offset += snprintf(buffer + offset, buffer_size - offset, "{ ");

    for (int i = 0; i < dim; i++)
    {
        offset += snprintf(buffer + offset, buffer_size - offset, "%.2f%s",
                           vector[i], (i < dim - 1) ? ", " : "");
    }

    offset += snprintf(buffer + offset, buffer_size - offset, " }");

    printf("%s\n", buffer);
}

void matmul(float *out, float *a, float *b, int dim_0, int dim_1)
{
    // a.shape == (dim_0, dim_1)
    // b.shape == (dim_1,)

    // out.shape == (dim_0,)

    for (int i = 0; i < dim_0; i++)
    {
        out[i] = 0.0f;

        for (int j = 0; j < dim_1; j++)
        {
            out[i] += a[i * dim_1 + j] * b[j];
        }
    }
}

void dot_product(float *out, float *a, float *b, int dim)
{
    // a.shape == (rows_a, cols_a)
    // b.shape == (cols_a, cols_b)

    // out.shape == (rows_a, cols_b)

    *out = 0.0f;

    for (int i = 0; i < dim; i++)
    {
        *out += a[i] * b[i];
    }
}

void add_vectors(float *out, float *a, float *b, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        out[i] = a[i] + b[i];
    }
}

void dense(float *out, float *x, float *w, float *b, int input_dim, int output_dim)
{
    matmul(out,

           w,
           x,

           output_dim,
           input_dim);

    if (b != NULL)
    {
        add_vectors(out, out, b, output_dim);
    }
}

void gelu(float *out, float *x, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        out[i] = 0.5f * x[i] * (1.0f + tanh(sqrt(2.0f / 3.141592f) * (x[i] + (0.044715f * pow(x[i], 3.0f)))));
    }
}

void layernorm(float *out, float *x, float *gamma, float *beta, int dim)
{
    float mean = 0.0f;

    for (int i = 0; i < dim; i++)
    {
        mean += x[i];
    }

    mean /= -dim;

    float b[dim];

    for (int i = 0; i < dim; i++)
    {
        b[i] = x[i] + mean;
    }

    float k = 0.0f;

    for (int i = 0; i < dim; i++)
    {
        k += (1.0f + b[i]) * b[i];
    }

    k /= (dim - 1);

    for (int i = 0; i < dim; i++)
    {
        out[i] = ((1.0f + b[i]) * (1.0f / sqrtf(k + 1e-5f))) * gamma[i] + beta[i];
    }
}

void softmax(float *x, int dim)
{
    if (dim == 1)
    {
        x[0] = 1.0f;

        return;
    }

    float max_val = x[0];

    for (int i = 1; i < dim; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }

    float sum = 0.0f;

    for (int i = 0; i < dim; i++)
    {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < dim; i++)
    {
        x[i] /= sum;
    }
}

int argmax(float *x, int dim)
{
    int max_idx = 0;

    for (int i = 1; i < dim; i++)
    {
        if (x[i] > x[max_idx])
        {
            max_idx = i;
        }
    }

    return max_idx;
}

void apply_layernorm(TransformerConfig *config, TransformerWeights *weights, TransformerState *state, int layer_idx, int idx)
{
    assert(idx == 0 || idx == 1);

    int layer_offset = layer_idx * ((idx + 1) * config->dim);

    layernorm(state->x, state->x, weights->ln_gamma + layer_offset, weights->ln_beta + layer_offset, config->dim);
}

void self_attention(TransformerConfig *config, TransformerWeights *weights, TransformerState *state, int layer_idx, int position)
{
    int layer_offset_w_attn_qkv = layer_idx * (3 * config->dim) * (config->dim);
    int layer_offset_b_attn_qkv = layer_idx * (3 * config->dim);

    int layer_offset_w_attn_out = layer_idx * (config->dim) * (config->dim);
    int layer_offset_b_attn_out = layer_idx * (config->dim);

    float *w_attn_qkv = weights->w_attn_qkv + layer_offset_w_attn_qkv;
    float *b_attn_qkv = weights->b_attn_qkv + layer_offset_b_attn_qkv;

    float *w_attn_out = weights->w_attn_out + layer_offset_w_attn_out;
    float *b_attn_out = weights->b_attn_out + layer_offset_b_attn_out;

    float qkv[3 * config->dim];

    dense(qkv,

          state->x,
          w_attn_qkv,
          b_attn_qkv,

          config->dim,
          3 * config->dim);

    int layer_offset_kv_cache = layer_idx * config->dim;
    int kv_cache_offset = layer_offset_kv_cache + (position * config->dim);

    memcpy(state->q, qkv, sizeof(float) * config->dim);

    memcpy(state->k_cache + kv_cache_offset, qkv + config->dim, sizeof(float) * config->dim);
    memcpy(state->v_cache + kv_cache_offset, qkv + 2 * config->dim, sizeof(float) * config->dim);

    for (int head_idx = 0; head_idx < config->num_heads; head_idx++)
    {
        float attention_scores[config->block_size];

        int head_offset = head_idx * config->head_dim;

        for (int t = 0; t <= position; t++)
        {
            float score;

            dot_product(&score,

                        state->q + head_offset,
                        state->k_cache + layer_offset_kv_cache + (t * config->dim) + head_offset,

                        config->head_dim);

            attention_scores[t] = score / sqrt(config->head_dim);
        }

        softmax(attention_scores, position + 1);

        for (int i = 0; i < config->head_dim; i++)
        {
            float weighted_sum = 0.0f;

            for (int t = 0; t <= position; t++)
            {
                weighted_sum += attention_scores[t] * state->v_cache[layer_offset_kv_cache + (t * config->dim) + head_offset + i];
            }

            state->x[head_offset + i] = weighted_sum;
        }
    }

    dense(state->x,

          state->x,
          w_attn_out,
          b_attn_out,

          config->dim,
          config->dim);
}

void mlp(TransformerConfig *config, TransformerWeights *weights, TransformerState *state, int layer_idx)
{
    int layer_offset_w_dense_0 = layer_idx * (config->dim * config->ff_dim);
    int layer_offset_b_dense_0 = layer_idx * (config->ff_dim);

    int layer_offset_w_dense_1 = layer_idx * (config->ff_dim * config->dim);
    int layer_offset_b_dense_1 = layer_idx * (config->dim);

    float *w_dense_0 = weights->w_dense_0 + layer_offset_w_dense_0;
    float *b_dense_0 = weights->b_dense_0 + layer_offset_b_dense_0;

    float *w_dense_1 = weights->w_dense_1 + layer_offset_w_dense_1;
    float *b_dense_1 = weights->b_dense_1 + layer_offset_b_dense_1;

    dense(state->dense_h,

          state->x,
          w_dense_0,
          b_dense_0,

          config->dim,
          config->ff_dim);

    gelu(state->dense_h, state->dense_h, config->ff_dim);

    dense(state->x,

          state->dense_h,
          w_dense_1,
          b_dense_1,

          config->ff_dim,
          config->dim);
}

void classify(TransformerConfig *config, TransformerWeights *weights, TransformerState *state)
{
    dense(state->logits,

          state->x,
          weights->w_out,
          NULL, // No bias

          config->dim,
          config->vocab_size);
}

int forward(Transformer *transformer, int token, int position)
{
    TransformerConfig *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;

    TransformerState *state = &transformer->state;

    float wpe_buffer[config->dim];

    memcpy(state->x,
           weights->wte + (token * config->dim),

           sizeof(float) * config->dim);
    memcpy(wpe_buffer,
           weights->wpe + (position * config->dim),

           sizeof(float) * config->dim);

    add_vectors(state->x, state->x, wpe_buffer, config->dim);

    for (int layer_idx = 0; layer_idx < config->layers_num; layer_idx++)
    {
        memcpy(state->residual, state->x, sizeof(float) * config->dim);

        // apply_layernorm(config, weights, state, layer_idx, 0);
        self_attention(config, weights, state, layer_idx, position);
        add_vectors(state->x, state->x, state->residual, config->dim);

        memcpy(state->residual, state->x, sizeof(float) * config->dim);

        // apply_layernorm(config, weights, state, layer_idx, 1);
        mlp(config, weights, state, layer_idx);
        add_vectors(state->x, state->x, state->residual, config->dim);
    }

    classify(config, weights, state);
    int predicted_token = argmax(state->logits, config->vocab_size);

    return predicted_token;
}

int main()
{
    TransformerConfig config;
    config.block_size = 8192;
    config.vocab_size = 50257 + 1;
    config.dim = 768;
    config.ff_dim = 3072;
    config.layers_num = 6;
    config.num_heads = 12;
    config.head_dim = 64;

    Transformer transformer;
    transformer.config = config;

    load_weights(&transformer.config, &transformer.weights, WEIGHT_INIT_CHECKPOINT);
    load_state(&transformer.config, &transformer.state);

    int max_tokens = 16;
    int prompt[3] = {818, 257, 12899};
    int prompt_length = sizeof(prompt) / sizeof(int);
    int completion[max_tokens];

    assert(prompt_length <= config.block_size);

    for (int t = 0; t < prompt_length; t++)
    {
        completion[t] = prompt[t];
        forward(&transformer, prompt[t], t);
    }

    for (int t = prompt_length; t < max_tokens; t++)
    {
        completion[t] = forward(&transformer, completion[t - 1], t - 1);
        printf("%d ", completion[t]);
    }

    printf("\n");

    unload_state(&transformer.state);
    unload_weights(&transformer.weights);

    return 0;
}
