/*
 * policy_core.h - hardware-free part of the bare-metal policy: observation
 * normalisation, the SAC actor MLP and the action sampler. No BSP calls, so
 * it is also compiled on the host by test_policy_core.py and checked against
 * the PyTorch actor.
 */
#ifndef POLICY_CORE_H
#define POLICY_CORE_H

#include <stdint.h>

typedef struct {
    int n_in, h1, h2, n_out;
} net_dims_t;

/* Words a weight image with these dims occupies:
 * W1[h1][n_in] b1[h1] W2[h2][h1] b2[h2] W3[n_out][h2] b3[n_out] (float32). */
uint32_t net_words(const net_dims_t *d);

/* SB3 SAC actor: latent_pi = Linear-ReLU-Linear-ReLU, then the [mu; log_std]
 * head. out[0] = mu, out[1] = log_std (unclamped). h1buf/h2buf hold h1/h2 floats. */
void mlp_forward(const float *w, const net_dims_t *d, const float *x,
                 float *h1buf, float *h2buf, float *out);

/* pendulum_env._wrap_pi: ((x + pi) mod 2pi) - pi, result in [-pi, pi). */
float wrap_pi(float x);

/* pendulum_env._apply_norm for one feature (NORM_* in rpu_shared.h). */
float apply_norm(int kind, float p1, float p2, float x);

/* xorshift128+ style generator, 32-bit outputs; normal draws via Box-Muller. */
typedef struct {
    uint32_t s[4];
    int has_spare;
    float spare;
} rng_t;

void  rng_seed(rng_t *r, uint32_t seed);
uint32_t rng_u32(rng_t *r);
float rng_uniform(rng_t *r);          /* (0, 1) */
float rng_normal(rng_t *r);           /* N(0, 1) */

/* SquashedDiagGaussian action (SB3 SAC): log_std clamped to [ls_min, ls_max];
 * explore: tanh(mu + exp(log_std) * eps), else tanh(mu).
 * Writes the clamped log_std and the eps used (0 when deterministic). */
float sac_action(float mu, float log_std, float ls_min, float ls_max, int explore,
                 rng_t *r, float *eps_out, float *ls_out);

#endif /* POLICY_CORE_H */
