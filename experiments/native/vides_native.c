/* Process-level harness for the experiments suite.
 *
 * Linked against the same compiled objects as the Python module (see
 * experiments/common.py: build_native), it calls the internal NEGF
 * processes directly -- the batched solver and the batched contact
 * self-energy -- on the CPU and on the GPU with identical inputs, and
 * prints one JSON object with the differences and timings.
 *
 *   vides_native rgf   VARIANT n Nc NB eta reps      VARIANT = std|lake|mode
 *   vides_native decim CELL n NB eta reps            CELL = gnr|hsrc|hdrn
 *   vides_native selfh LEAD n NE eta kind            LEAD = src|drn, kind = gnr|random
 *
 * All inputs are generated deterministically, so a run is reproducible.
 * When no GPU is usable, GPU fields are null.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vides_rgf_batch.h"
#include "cmatrix.h"
#include "cfree_cmatrix.h"
#include "cmatinv.h"
#include "selfH_dec.h"

complex **selfGNR(double E, double *Em1, int N, double thop, double eta);
void selfGNR_cell(double *Em1, int N, double thop,
                  complex **W0, complex **BETA, complex **BETADAGA);
complex **selfH_new(double E, complex ***diag, complex ***updiag,
                    complex ***lowdiag, int N, int Nc, int lead, double eta);
complex **create_updiagGNR(int i, int n, double thop);
complex **create_lowdiagGNR(int i, int n, double thop);

/* ------------------------------------------------------------------ */
static unsigned long long rs = 88172645463325252ULL;
static double rnd(void)
{ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
  return (double)(rs >> 11) / 9007199254740992.0 * 2.0 - 1.0; }

static complex ***blocks(int Nc, int n)
{ complex ***b = malloc((size_t)Nc * sizeof *b); int k, i, j;
  for (k = 0; k < Nc; k++) { b[k] = cmatrix(0, n-1, 0, n-1);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) { b[k][i][j].r = 0; b[k][i][j].i = 0; } }
  return b; }
static void free_blocks(complex ***b, int Nc, int n)
{ int k; for (k = 0; k < Nc; k++) cfree_cmatrix(b[k], 0, n-1, 0, n-1); free(b); }

/* Hermitian chain, four-slice periodic, general complex couplings:
   slice s has on-site D[s%4] and couples to s+1 by C[s%4]. */
static void random_chain(int n, int Nc, complex ***D, complex ***U, complex ***L)
{ int s, i, j, t;
  complex ***Dt = blocks(4, n), ***Ct = blocks(4, n);
  for (t = 0; t < 4; t++) for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
    Ct[t][i][j].r = 0.8 * rnd() / sqrt((double)n) + (i == j ? -1.5 : 0); Ct[t][i][j].i = 0.4 * rnd() / sqrt((double)n);
    if (j >= i) { Dt[t][i][j].r = (i == j ? 0.6 * rnd() : 0.5 * rnd() / sqrt((double)n));
                  Dt[t][i][j].i = (i == j ? 0 : 0.3 * rnd() / sqrt((double)n));
                  Dt[t][j][i].r = Dt[t][i][j].r; Dt[t][j][i].i = -Dt[t][i][j].i; } }
  for (s = 0; s < Nc; s++) for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
    D[s][i][j] = Dt[s % 4][i][j];
    if (s < Nc - 1) U[s][i][j] = Ct[s % 4][i][j];
    if (s > 0) { L[s][i][j].r = Ct[(s-1) % 4][j][i].r; L[s][i][j].i = -Ct[(s-1) % 4][j][i].i; } }
  free_blocks(Dt, 4, n); free_blocks(Ct, 4, n);
}

/* Graphene-ribbon slices, as GNR_charge_T.c builds them. */
static void gnr_chain(int n, int Nc, double phi, complex ***D, complex ***U, complex ***L)
{ int i, j, k; complex **t;
  for (i = 0; i < Nc; i++) for (j = 0; j < n; j++) D[i][j][j].r = phi;
  for (i = 0; i < Nc - 1; i++) { t = create_updiagGNR(i, n, -2.7);
    for (j = 0; j < n; j++) for (k = 0; k < n; k++) U[i][j][k] = t[j][k]; cfree_cmatrix(t, 0, n-1, 0, n-1); }
  for (i = 1; i < Nc; i++) { t = create_lowdiagGNR(i - 1, n, -2.7);
    for (j = 0; j < n; j++) for (k = 0; k < n; k++) L[i][j][k] = t[j][k]; cfree_cmatrix(t, 0, n-1, 0, n-1); }
}

/* Contact self-energies with a genuine anti-Hermitian part. */
static complex ***random_sigmas(int NB, int n)
{ complex ***s = blocks(NB, n); int b, i, j;
  for (b = 0; b < NB; b++) for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
    s[b][i][j].r = 0.2 * rnd() / sqrt((double)n); s[b][i][j].i = -0.15 * fabs(rnd()) / sqrt((double)n) - (i == j ? 0.05 : 0); }
  return s; }

/* ---- metrics -------------------------------------------------------- */
typedef struct { double maxabs, relpeak, worstelem; } diffm;

static diffm dcompare(const double *a, const double *b, size_t N)
{ diffm m = {0, 0, 0}; double peak = 0; size_t k;
  for (k = 0; k < N; k++) {
    double d = fabs(a[k] - b[k]), s = fmax(fabs(a[k]), fabs(b[k]));
    if (d > m.maxabs) m.maxabs = d; if (s > peak) peak = s;
    if (s > 1e-300 && d / s > m.worstelem) m.worstelem = d / s; }
  m.relpeak = peak > 0 ? m.maxabs / peak : 0; return m; }

static double cdiff(complex **a, complex **b, int n)
{ double d = 0, m = 0; int i, j;
  for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
    d = fmax(d, hypot(a[i][j].r - b[i][j].r, a[i][j].i - b[i][j].i));
    m = fmax(m, hypot(a[i][j].r, a[i][j].i)); }
  return m > 0 ? d / m : d; }

/* Residual of the lead's fixed-point equation, relative to |sigma|:
   sigma = BETA_30 [(W0 + (E + i eta) I - sigma@(3,3))^-1]_00 BETADAGA_03 */
static double lead_residual(complex **sig, double E, double eta, int N,
                            complex **W0, complex **B, complex **BD)
{ int M = 4 * N, i, j, k, l; double r = 0, m = 0;
  complex **W = cmatrix(0, M-1, 0, M-1), **G;
  for (i = 0; i < M; i++) for (j = 0; j < M; j++) W[i][j] = W0[i][j];
  for (i = 0; i < M; i++) { W[i][i].r = E + W[i][i].r; W[i][i].i += eta; }
  for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
    W[3*N+i][3*N+j].r -= sig[i][j].r; W[3*N+i][3*N+j].i -= sig[i][j].i; }
  G = cmatinv(W, M);
  for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
    double fr = 0, fi = 0;
    for (k = 0; k < N; k++) for (l = 0; l < N; l++) {
      complex b = B[3*N+i][k], g = G[k][l], c = BD[l][3*N+j];
      double tr = b.r*g.r - b.i*g.i, ti = b.r*g.i + b.i*g.r;
      fr += tr*c.r - ti*c.i; fi += tr*c.i + ti*c.r; }
    r = fmax(r, hypot(fr - sig[i][j].r, fi - sig[i][j].i));
    m = fmax(m, hypot(sig[i][j].r, sig[i][j].i)); }
  cfree_cmatrix(W, 0, M-1, 0, M-1); cfree_cmatrix(G, 0, M-1, 0, M-1);
  return m > 0 ? r / m : r; }

/* ---- JSON helpers --------------------------------------------------- */
static void jnum(const char *k, double v, int comma)
{ if (isfinite(v)) printf("\"%s\": %.17g%s", k, v, comma ? ", " : "");
  else printf("\"%s\": null%s", k, comma ? ", " : ""); }
static void jarr(const char *k, const double *v, int n, int comma)
{ int i; printf("\"%s\": [", k);
  for (i = 0; i < n; i++) { if (isfinite(v[i])) printf("%.10g", v[i]); else printf("null"); if (i < n-1) printf(", "); }
  printf("]%s", comma ? ", " : ""); }
static void jdiff(const char *k, diffm d, int comma)
{ printf("\"%s\": {", k); jnum("maxabs", d.maxabs, 1); jnum("relpeak", d.relpeak, 1);
  jnum("worstelem", d.worstelem, 0); printf("}%s", comma ? ", " : ""); }

/* ---- rgf ------------------------------------------------------------ */
static int cmd_rgf(int argc, char **argv)
{
  const char *var = argv[2];
  int n = atoi(argv[3]), Nc = atoi(argv[4]), NB = atoi(argv[5]), reps = atoi(argv[7]);
  double eta = atof(argv[6]);
  int variant = !strcmp(var, "lake") ? VIDES_RGF_LAKE : !strcmp(var, "mode") ? VIDES_RGF_MODE : VIDES_RGF_STD;
  int Nreal = variant == VIDES_RGF_MODE ? 2 * n : 0, nout = Nreal ? Nreal : n, b, r, gpu, chunk;
  int *order = malloc((size_t)n * sizeof(int));
  complex ***D = blocks(Nc, n), ***U = blocks(Nc, n), ***L = blocks(Nc, n), ***SS, ***SD;
  double *E = malloc((size_t)NB * sizeof(double));
  size_t NA = (size_t)NB * Nc * nout;
  double *A1c = calloc(NA, 8), *A2c = calloc(NA, 8), *Tc = calloc(NB, 8);
  double *A1g = calloc(NA, 8), *A2g = calloc(NA, 8), *Tg = calloc(NB, 8);
  double tcpu, tg_first = NAN, tg_best = NAN, t0;
  vides_rgf_desc d;
  (void)argc;

  for (b = 0; b < n; b++) order[b] = b;
  random_chain(n, Nc, D, U, L);
  SS = random_sigmas(NB, n); SD = random_sigmas(NB, n);
  for (b = 0; b < NB; b++) E[b] = -2.0 + 4.0 * (b + 0.5) / NB;

  memset(&d, 0, sizeof d);
  d.n = n; d.Nc = Nc; d.NB = NB; d.variant = variant; d.flagtrans = 1;
  d.eta = eta; d.Nreal = Nreal; d.order = order;

  t0 = vides_now();
  if (vides_rgf_batch_cpu(&d, E, D, U, L, SS, SD, A1c, A2c, Tc)) { printf("{\"error\": \"cpu rgf failed\"}\n"); return 1; }
  tcpu = vides_now() - t0;

  gpu = vides_gpu_available();
  /* The GPU driver holds a whole batch on the device at once; split NB
     into the chunks the entry points would use. */
  chunk = gpu ? vides_gpu_batch_size(&d) : 0;
  if (gpu && chunk < 1) gpu = 0;
  if (gpu) for (r = 0; r < (reps < 1 ? 1 : reps) + 1; r++) {
    double dt;
    int s0;
    t0 = vides_now();
    for (s0 = 0; s0 < NB; s0 += chunk) {
      vides_rgf_desc dc = d;
      size_t off = (size_t)s0 * Nc * nout;
      dc.NB = NB - s0 < chunk ? NB - s0 : chunk;
      if (vides_rgf_batch_gpu(&dc, E + s0, D, U, L, SS + s0, SD + s0, A1g + off, A2g + off, Tg + s0)) {
        printf("{\"error\": \"gpu rgf failed\"}\n"); return 1; }
    }
    dt = vides_now() - t0;
    if (r == 0) tg_first = dt; else if (isnan(tg_best) || dt < tg_best) tg_best = dt;
  }

  printf("{\"cmd\": \"rgf\", \"variant\": \"%s\", \"n\": %d, \"Nc\": %d, \"NB\": %d, \"nout\": %d, ", var, n, Nc, NB, nout);
  jnum("eta", eta, 1); printf("\"gpu\": %s, \"backend\": \"%s\", ", gpu ? "true" : "false", vides_gpu_describe());
  jnum("t_cpu", tcpu, 1); jnum("t_gpu_first", tg_first, 1); jnum("t_gpu", tg_best, 1);
  printf("\"chunk\": %d, ", chunk);
  if (gpu) { jdiff("A1", dcompare(A1c, A1g, NA), 1); jdiff("A2", dcompare(A2c, A2g, NA), 1); jdiff("T", dcompare(Tc, Tg, NB), 1); }
  jarr("E", E, NB, 1); jarr("T_cpu", Tc, NB, 1); jarr("T_gpu", gpu ? Tg : Tc, NB, 0);
  printf("}\n");
  return 0;
}

/* ---- decim ---------------------------------------------------------- */
static void make_cell(const char *cell, int n, complex **W0, complex **B, complex **BD)
{
  if (!strcmp(cell, "gnr")) {
    double *Em1 = malloc((size_t)4 * n * sizeof(double)); int i;
    for (i = 0; i < 4 * n; i++) Em1[i] = -0.2 + 0.05 * (i % 3);
    selfGNR_cell(Em1, n, -2.7, W0, B, BD); free(Em1);
  } else {
    int Nc = 13; complex ***D = blocks(Nc, n), ***U = blocks(Nc, n), ***L = blocks(Nc, n);
    random_chain(n, Nc, D, U, L);
    selfH_dec_cell(D, U, L, n, Nc, !strcmp(cell, "hdrn") ? 1 : 0, W0, B, BD);
    free_blocks(D, Nc, n); free_blocks(U, Nc, n); free_blocks(L, Nc, n);
  }
}

static int cmd_decim(int argc, char **argv)
{
  const char *cell = argv[2];
  int n = atoi(argv[3]), NB = atoi(argv[4]), reps = atoi(argv[6]), M = 4 * n, b, r, gpu, i;
  double eta = atof(argv[5]), tcpu, tg_first = NAN, tg_best = NAN, t0;
  complex **W0 = cmatrix(0, M-1, 0, M-1), **B = cmatrix(0, M-1, 0, M-1), **BD = cmatrix(0, M-1, 0, M-1);
  complex ***cpu = malloc((size_t)NB * sizeof *cpu), ***gp = malloc((size_t)NB * sizeof *gp);
  double *E = malloc((size_t)NB * 8), *dif = malloc((size_t)NB * 8), *rc = malloc((size_t)NB * 8), *rg = malloc((size_t)NB * 8);
  double worst = 0, rcmax = 0, rgmax = 0;
  (void)argc;

  make_cell(cell, n, W0, B, BD);
  for (b = 0; b < NB; b++) { E[b] = -2.5 + 5.0 * (b + 0.5) / NB;
    cpu[b] = cmatrix(0, n-1, 0, n-1); gp[b] = cmatrix(0, n-1, 0, n-1); }

  t0 = vides_now();
  vides_decimation_batch_cpu(M, NB, E, eta, W0, B, BD, 3*n, n, cpu);
  tcpu = vides_now() - t0;

  gpu = vides_gpu_available();
  if (gpu) for (r = 0; r < (reps < 1 ? 1 : reps) + 1; r++) {
    double dt; t0 = vides_now();
    if (vides_decimation_batch_gpu(M, NB, E, eta, W0, B, BD, 3*n, n, gp)) { printf("{\"error\": \"gpu decimation failed\"}\n"); return 1; }
    dt = vides_now() - t0;
    if (r == 0) tg_first = dt; else if (isnan(tg_best) || dt < tg_best) tg_best = dt;
  }

  for (b = 0; b < NB; b++) {
    rc[b] = lead_residual(cpu[b], E[b], eta, n, W0, B, BD);
    rg[b] = gpu ? lead_residual(gp[b], E[b], eta, n, W0, B, BD) : NAN;
    dif[b] = gpu ? cdiff(cpu[b], gp[b], n) : NAN;
    if (gpu && dif[b] > worst) worst = dif[b];
    if (rc[b] > rcmax) rcmax = rc[b]; if (gpu && rg[b] > rgmax) rgmax = rg[b];
  }

  printf("{\"cmd\": \"decim\", \"cell\": \"%s\", \"n\": %d, \"NB\": %d, ", cell, n, NB);
  jnum("eta", eta, 1); printf("\"gpu\": %s, ", gpu ? "true" : "false");
  jnum("t_cpu", tcpu, 1); jnum("t_gpu_first", tg_first, 1); jnum("t_gpu", tg_best, 1);
  jnum("worst_rel_diff", gpu ? worst : NAN, 1); jnum("residual_cpu", rcmax, 1); jnum("residual_gpu", gpu ? rgmax : NAN, 1);
  jarr("E", E, NB, 1); jarr("rel_diff", dif, NB, 1); jarr("res_cpu", rc, NB, 1); jarr("res_gpu", rg, NB, 0);
  printf("}\n");
  for (i = 0; i < NB; i++) { cfree_cmatrix(cpu[i], 0, n-1, 0, n-1); cfree_cmatrix(gp[i], 0, n-1, 0, n-1); }
  return 0;
}

/* ---- selfh: eigen (selfH_new) vs decimation (selfH_dec), CPU -------- */
static int cmd_selfh(int argc, char **argv)
{
  int lead = !strcmp(argv[2], "drn") ? 1 : 0, n = atoi(argv[3]), NE = atoi(argv[4]), Nc = 13, M = 4 * n, e;
  double eta = atof(argv[5]), t_eig = 0, t_dec = 0, t0;
  const char *kind = argv[6];
  complex ***D = blocks(Nc, n), ***U = blocks(Nc, n), ***L = blocks(Nc, n);
  complex **W0 = cmatrix(0, M-1, 0, M-1), **B = cmatrix(0, M-1, 0, M-1), **BD = cmatrix(0, M-1, 0, M-1);
  double *E = malloc((size_t)NE * 8), *dif = malloc((size_t)NE * 8), *re = malloc((size_t)NE * 8), *rd = malloc((size_t)NE * 8);
  double worst = 0, remax = 0, rdmax = 0;
  (void)argc;

  if (!strcmp(kind, "gnr")) gnr_chain(n, Nc, -0.2, D, U, L); else random_chain(n, Nc, D, U, L);
  selfH_dec_cell(D, U, L, n, Nc, lead, W0, B, BD);
  for (e = 0; e < NE; e++) {
    complex **a, **b;
    E[e] = -2.5 + 5.0 * (e + 0.5) / NE;
    t0 = vides_now(); a = selfH_new(E[e], D, U, L, n, Nc, lead, eta); t_eig += vides_now() - t0;
    t0 = vides_now(); b = selfH_dec(E[e], D, U, L, n, Nc, lead, eta); t_dec += vides_now() - t0;
    dif[e] = cdiff(a, b, n);
    re[e] = lead_residual(a, E[e], eta, n, W0, B, BD);
    rd[e] = lead_residual(b, E[e], eta, n, W0, B, BD);
    if (dif[e] > worst) worst = dif[e]; if (re[e] > remax) remax = re[e]; if (rd[e] > rdmax) rdmax = rd[e];
    cfree_cmatrix(a, 0, n-1, 0, n-1); cfree_cmatrix(b, 0, n-1, 0, n-1);
  }
  printf("{\"cmd\": \"selfh\", \"lead\": \"%s\", \"kind\": \"%s\", \"n\": %d, \"NE\": %d, ", argv[2], kind, n, NE);
  jnum("eta", eta, 1); jnum("t_eig", t_eig, 1); jnum("t_dec", t_dec, 1);
  jnum("worst_rel_diff", worst, 1); jnum("residual_eig", remax, 1); jnum("residual_dec", rdmax, 1);
  jarr("E", E, NE, 1); jarr("rel_diff", dif, NE, 1); jarr("res_eig", re, NE, 1); jarr("res_dec", rd, NE, 0);
  printf("}\n");
  return 0;
}

int main(int argc, char **argv)
{
  if (argc >= 8 && !strcmp(argv[1], "rgf"))   return cmd_rgf(argc, argv);
  if (argc >= 7 && !strcmp(argv[1], "decim")) return cmd_decim(argc, argv);
  if (argc >= 7 && !strcmp(argv[1], "selfh")) return cmd_selfh(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "info")) {
    printf("{\"cmd\": \"info\", \"gpu\": %s, \"backend\": \"%s\"}\n",
           vides_gpu_available() ? "true" : "false", vides_gpu_describe());
    return 0;
  }
  fprintf(stderr,
    "usage: vides_native rgf   std|lake|mode n Nc NB eta reps\n"
    "       vides_native decim gnr|hsrc|hdrn n NB eta reps\n"
    "       vides_native selfh src|drn n NE eta gnr|random\n"
    "       vides_native info\n");
  return 2;
}
