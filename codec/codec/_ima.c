/*
 * Minimal reentrant IMA/DVI ADPCM acceleration core.
 * No Python headers required; loaded through ctypes.
 *
 * The only mutable state is on the caller's stack; the tables below are
 * static const, so every function is safe to call concurrently.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define ADPCM_NIDX 89

static const int32_t STEP_TABLE[ADPCM_NIDX] = {
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
};

static const int INDEX_TABLE[8] = {-1, -1, -1, -1, 2, 4, 6, 8};

/* Q[idx][mag] = canonical IMA magnitude reconstruction (integer units). */
static int32_t Q[ADPCM_NIDX][8];
static int Q_ready = 0;

static void build_tables(void) {
    if (Q_ready) return;
    for (int i = 0; i < ADPCM_NIDX; i++) {
        int step = STEP_TABLE[i];
        int base = step >> 3;
        for (int m = 0; m < 8; m++) {
            int d = base;
            if (m & 4) d += step;
            if (m & 2) d += step >> 1;
            if (m & 1) d += step >> 2;
            Q[i][m] = d;
        }
    }
    Q_ready = 1;
}

static int clamp_idx(int idx) {
    if (idx < 0) return 0;
    if (idx > 88) return 88;
    return idx;
}

/* Explicit one-time table build (called once at library load). */
void adpcm_init(void) { build_tables(); }

/* Returns 0 when all samples finite, otherwise 1=nan 2=+inf 3=-inf. */
int adpcm_check_finite(const double *x, int64_t n, int64_t *bad) {
    for (int64_t i = 0; i < n; i++) {
        double v = x[i];
        if (isnan(v)) { if (bad) *bad = i; return 1; }
        if (isinf(v)) { if (bad) *bad = i; return v > 0.0 ? 2 : 3; }
    }
    return 0;
}

/*
 * Primed enhanced-predictor encode.
 * Nibble 0 is left zero (reserved); nibbles i>=1 code sample i.
 * out must hold ceil(n/2) bytes.  On success *sse receives reconstruction SSE.
 */
int adpcm_encode_primed(const double *x, int64_t n, unsigned char *out,
                        int idx0, double k, double *sse_out) {
    build_tables();
    if (n <= 0) { if (sse_out) *sse_out = 0.0; return 0; }
    int64_t nbytes = (n + 1) / 2;
    for (int64_t b = 0; b < nbytes; b++) out[b] = 0;

    double y1 = x[0], y2 = x[0];
    int idx = clamp_idx(idx0);
    double sse = 0.0;
    const double INV = 1.0 / 32768.0;

    for (int64_t i = 1; i < n; i++) {
        double slope = y1 - y2;
        double p = y1 + k * slope;
        if (p > 1.5) p = 1.5;
        else if (p < -1.5) p = -1.5;

        double r = x[i] - p;
        double ri = r * 32768.0;
        int sign = 0;
        double ad = ri;
        if (ri < 0.0) { sign = 8; ad = -ri; }

        int best_m = 0;
        double best_err = fabs(ad - (double)Q[idx][0]);
        for (int m = 1; m < 8; m++) {
            double e = fabs(ad - (double)Q[idx][m]);
            if (e < best_err) { best_err = e; best_m = m; }
        }
        double diff = (double)Q[idx][best_m] * INV;
        double y = sign ? (p - diff) : (p + diff);
        if (y > 1.0) y = 1.0;
        else if (y < -1.0) y = -1.0;

        int code = best_m | sign;
        int64_t j = i >> 1;
        if (i & 1) out[j] |= (unsigned char)((code << 4) & 0xF0);
        else out[j] |= (unsigned char)(code & 0x0F);

        double d = x[i] - y;
        sse += d * d;

        y2 = y1;
        y1 = y;
        idx = clamp_idx(idx + INDEX_TABLE[best_m]);
    }
    if (sse_out) *sse_out = sse;
    return 0;
}

/* Pick the index whose quantizer best matches |x1-x0|. */
int adpcm_initial_index(const double *x, int64_t n) {
    build_tables();
    if (n < 2) return 0;
    double target = fabs((x[1] - x[0]) * 32768.0);
    int best_i = 0;
    double best_err = -1.0;
    for (int i = 0; i < ADPCM_NIDX; i++) {
        for (int m = 0; m < 8; m++) {
            double e = fabs(target - (double)Q[i][m]);
            if (best_err < 0.0 || e < best_err) { best_err = e; best_i = i; }
        }
    }
    return best_i;
}

/* Primed decode matching adpcm_encode_primed. */
void adpcm_decode_primed(const unsigned char *data, int64_t n, double *out,
                         double x0, int idx0, double k) {
    build_tables();
    if (n <= 0) return;
    const double INV = 1.0 / 32768.0;
    out[0] = x0;
    double y1 = x0, y2 = x0;
    int idx = clamp_idx(idx0);
    for (int64_t i = 1; i < n; i++) {
        int64_t j = i >> 1;
        int code = (i & 1) ? ((data[j] >> 4) & 0x0F) : (data[j] & 0x0F);
        int mag = code & 7;
        double p = y1 + k * (y1 - y2);
        if (p > 1.5) p = 1.5;
        else if (p < -1.5) p = -1.5;
        double diff = (double)Q[idx][mag] * INV;
        double y = (code & 8) ? (p - diff) : (p + diff);
        if (y > 1.0) y = 1.0;
        else if (y < -1.0) y = -1.0;
        out[i] = y;
        y2 = y1;
        y1 = y;
        idx = clamp_idx(idx + INDEX_TABLE[mag]);
    }
}

/* Textbook integer IMA/DVI decode, nibble i -> sample i. */
void adpcm_decode_standard(const unsigned char *data, int64_t n, double *out) {
    int pred = 0, idx = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t j = i >> 1;
        int code = (i & 1) ? ((data[j] >> 4) & 0x0F) : (data[j] & 0x0F);
        int step = STEP_TABLE[idx];
        int diff = step >> 3;
        if (code & 4) diff += step;
        if (code & 2) diff += step >> 1;
        if (code & 1) diff += step >> 2;
        if (code & 8) pred -= diff; else pred += diff;
        if (pred > 32767) pred = 32767;
        else if (pred < -32768) pred = -32768;
        out[i] = (double)pred / 32768.0;
        idx += INDEX_TABLE[code & 7];
        idx = clamp_idx(idx);
    }
}
