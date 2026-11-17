/* policy_core.c - see policy_core.h. Pure C99 + libm, float32 throughout. */
#include "policy_core.h"
#include "rpu_shared.h"

#include <math.h>

#define PI_F      3.14159265358979323846f
#define TWO_PI_F  6.28318530717958647692f

uint32_t net_words(const net_dims_t *d)
{
    return (uint32_t)(d->h1 * d->n_in + d->h1
                      + d->h2 * d->h1 + d->h2
                      + d->n_out * d->h2 + d->n_out);
}

/* y[r] = act(b[r] + W[r][:] . x), rows of W contiguous (torch layout). */
static void dense(const float *W, const float *b, const float *x, float *y,
                  int n_out, int n_in, int relu)
{
    for (int r = 0; r < n_out; ++r) {
        const float *wr = W + (uint32_t)r * (uint32_t)n_in;
        /* 4 partial sums: keeps the FPU pipeline busy (no fused dependency chain) */
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int c = 0;
        for (; c + 3 < n_in; c += 4) {
            a0 += wr[c]     * x[c];
            a1 += wr[c + 1] * x[c + 1];
            a2 += wr[c + 2] * x[c + 2];
            a3 += wr[c + 3] * x[c + 3];
        }
        for (; c < n_in; ++c)
            a0 += wr[c] * x[c];
        float acc = b[r] + ((a0 + a1) + (a2 + a3));
        y[r] = (relu && acc < 0.0f) ? 0.0f : acc;
    }
}

void mlp_forward(const float *w, const net_dims_t *d, const float *x,
                 float *h1buf, float *h2buf, float *out)
{
    const float *W1 = w;
    const float *b1 = W1 + d->h1 * d->n_in;
    const float *W2 = b1 + d->h1;
    const float *b2 = W2 + d->h2 * d->h1;
    const float *W3 = b2 + d->h2;
    const float *b3 = W3 + d->n_out * d->h2;
    dense(W1, b1, x, h1buf, d->h1, d->n_in, 1);
    dense(W2, b2, h1buf, h2buf, d->h2, d->h1, 1);
    dense(W3, b3, h2buf, out, d->n_out, d->h2, 0);
}

float wrap_pi(float x)
{
    float t = fmodf(x + PI_F, TWO_PI_F);   /* C fmod keeps the sign of x */
    if (t < 0.0f)
        t += TWO_PI_F;                     /* -> Python's floored modulo */
    return t - PI_F;
}

static float clip1(float x)
{
    return x > 1.0f ? 1.0f : (x < -1.0f ? -1.0f : x);
}

float apply_norm(int kind, float p1, float p2, float x)
{
    switch (kind) {
    case NORM_DIV:     return x / p1;
    case NORM_CLIPDIV: return clip1(x / p1);
    case NORM_GAIN:    return x * p1;
    case NORM_ASINH:   return clip1(asinhf(x / p1) / p2);
    default:           return x;
    }
}

static uint32_t splitmix32(uint32_t *z)
{
    uint32_t v = (*z += 0x9E3779B9u);
    v = (v ^ (v >> 16)) * 0x85EBCA6Bu;
    v = (v ^ (v >> 13)) * 0xC2B2AE35u;
    return v ^ (v >> 16);
}

void rng_seed(rng_t *r, uint32_t seed)
{
    uint32_t z = seed ? seed : 1u;
    for (int i = 0; i < 4; ++i)
        r->s[i] = splitmix32(&z);
    if (!(r->s[0] | r->s[1] | r->s[2] | r->s[3]))
        r->s[0] = 1u;
    r->has_spare = 0;
    r->spare = 0.0f;
}

/* xoshiro128+ */
uint32_t rng_u32(rng_t *r)
{
    uint32_t *s = r->s;
    uint32_t result = s[0] + s[3];
    uint32_t t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 11) | (s[3] >> 21);
    return result;
}

float rng_uniform(rng_t *r)
{
    /* 24 random bits, offset by half an LSB: strictly inside (0, 1) */
    return ((float)(rng_u32(r) >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

float rng_normal(rng_t *r)
{
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    float u1 = rng_uniform(r), u2 = rng_uniform(r);
    float rad = sqrtf(-2.0f * logf(u1));
    float ang = TWO_PI_F * u2;
    r->spare = rad * sinf(ang);
    r->has_spare = 1;
    return rad * cosf(ang);
}

float sac_action(float mu, float log_std, float ls_min, float ls_max, int explore,
                 rng_t *r, float *eps_out, float *ls_out)
{
    float ls = log_std < ls_min ? ls_min : (log_std > ls_max ? ls_max : log_std);
    float eps = explore ? rng_normal(r) : 0.0f;
    *ls_out = ls;
    *eps_out = eps;
    return tanhf(mu + expf(ls) * eps);
}
