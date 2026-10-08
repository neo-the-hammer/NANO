/* Fortran-ABI zgemm_/zgetrf_/zgetri_ for the emulation test, so the stock
 * ViDES routines (cmatmul.c, cmatinv.c) can run without a real LAPACK.
 * Not part of the shipped library. */
#include <stdlib.h>
#include <string.h>
#include "emu_linalg.h"

void zgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
            emu_cplx *ALPHA, emu_cplx *A, int *LDA, emu_cplx *B, int *LDB,
            emu_cplx *BETA, emu_cplx *C, int *LDC)
{
  int ca = (*TRANSA == 'C' || *TRANSA == 'c');
  int cb = (*TRANSB == 'C' || *TRANSB == 'c');
  emu_gemm(ca, cb, *M, *N, *K, *ALPHA, A, *LDA, B, *LDB, *BETA, C, *LDC);
}

void zgetrf_(int *M, int *N, emu_cplx *A, int *LDA, int *IPIV, int *INFO)
{
  *INFO = emu_getrf(A, *N, *LDA, IPIV);   /* square use only */
  (void)M;
}

void zgetri_(int *N, emu_cplx *A, int *LDA, int *IPIV,
             emu_cplx *WORK, int *LWORK, int *INFO)
{
  int n = *N;
  emu_cplx *C = (emu_cplx *)malloc((size_t)n*n*sizeof(emu_cplx));
  (void)WORK; (void)LWORK;
  if (!C) { *INFO = -1; return; }
  *INFO = emu_getri(A, n, *LDA, IPIV, C, n);
  memcpy(A, C, (size_t)n*n*sizeof(emu_cplx));
  free(C);
}

/* ------------------------------------------------------------------ */
/* zgeev_ (eigenvalues + right eigenvectors only), for running the stock */
/* selfH_new() in tests.  Plain shifted QR on the full matrix to Schur   */
/* form, then back-substitution on the triangular factor.  O(n^3) per    */
/* iteration, fine for the 2N x 2N matrices selfH_new uses.  Not a       */
/* LAPACK replacement.                                                   */
/* ------------------------------------------------------------------ */
static emu_cplx emu_add(emu_cplx a, emu_cplx b){ emu_cplx c; c.r=a.r+b.r; c.i=a.i+b.i; return c; }
static emu_cplx emu_conj(emu_cplx a){ a.i=-a.i; return a; }
static double emu_abs(emu_cplx a){ return sqrt(a.r*a.r+a.i*a.i); }
static emu_cplx emu_csqrt(emu_cplx z)
{
  double m=emu_abs(z), r=sqrt(0.5*(m+z.r)), i=sqrt(0.5*(m-z.r));
  emu_cplx s; s.r=r; s.i=(z.i<0)?-i:i; return s;
}
#define AT(A,r,c,ld) (A)[(r)+(size_t)(c)*(ld)]

void zgeev_(char *JOBVL, char *JOBVR, int *N, emu_cplx *A, int *LDA,
            emu_cplx *W, emu_cplx *VL, int *LDVL, emu_cplx *VR, int *LDVR,
            emu_cplx *WORK, int *LWORK, double *RWORK, int *INFO)
{
  int n=*N, lda=*LDA, ldvr=*LDVR, i,j,k,it,hi;
  emu_cplx *T=(emu_cplx*)malloc((size_t)n*n*sizeof(emu_cplx));
  emu_cplx *Q=(emu_cplx*)calloc((size_t)n*n,sizeof(emu_cplx));
  emu_cplx *Qk=(emu_cplx*)malloc((size_t)n*n*sizeof(emu_cplx));
  emu_cplx *R=(emu_cplx*)malloc((size_t)n*n*sizeof(emu_cplx));
  emu_cplx *tmp=(emu_cplx*)malloc((size_t)n*n*sizeof(emu_cplx));
  emu_cplx *y=(emu_cplx*)malloc((size_t)n*sizeof(emu_cplx));
  double anorm=0;
  (void)JOBVL;(void)VL;(void)LDVL;(void)WORK;(void)LWORK;(void)RWORK;
  *INFO=0;
  for(j=0;j<n;j++) for(i=0;i<n;i++){ AT(T,i,j,n)=AT(A,i,j,lda); anorm+=emu_abs(AT(T,i,j,n)); }
  for(i=0;i<n;i++){ AT(Q,i,i,n).r=1; }
  if (anorm==0) anorm=1;
  hi=n-1;
  for(it=0; hi>0 && it<100000; it++){
    /* deflate: last row of the active block below the diagonal ~ 0 */
    double off=0;
    for(j=0;j<hi;j++) off+=emu_abs(AT(T,hi,j,n));
    if (off <= 1e-15*anorm){ for(j=0;j<hi;j++){AT(T,hi,j,n).r=0;AT(T,hi,j,n).i=0;} hi--; continue; }
    /* Wilkinson shift from the trailing 2x2 of the active block */
    {
      emu_cplx a=AT(T,hi-1,hi-1,n), b=AT(T,hi-1,hi,n), c=AT(T,hi,hi-1,n), d=AT(T,hi,hi,n);
      emu_cplx tr=emu_add(a,d), det=emu_sub(emu_mul(a,d),emu_mul(b,c));
      emu_cplx h; h.r=0.5*tr.r; h.i=0.5*tr.i;
      emu_cplx disc=emu_csqrt(emu_sub(emu_mul(h,h),det));
      emu_cplx l1=emu_add(h,disc), l2=emu_sub(h,disc), mu;
      mu = (emu_abs(emu_sub(l1,d))<emu_abs(emu_sub(l2,d)))?l1:l2;
      if (it%11==10){ mu.r+=1e-3*anorm; }          /* exceptional shift */
      /* QR of the active block (T - mu I) by modified Gram-Schmidt */
      int m=hi+1;
      for(j=0;j<m;j++) for(i=0;i<m;i++){ AT(tmp,i,j,n)=AT(T,i,j,n); if(i==j) AT(tmp,i,j,n)=emu_sub(AT(tmp,i,j,n),mu); }
      for(j=0;j<m;j++) for(i=0;i<m;i++){ AT(R,i,j,n).r=0; AT(R,i,j,n).i=0; }
      for(j=0;j<m;j++){
        /* Two Gram-Schmidt passes.  One pass loses orthogonality as the
           shift nears an eigenvalue (T - mu I nearly singular), which
           breaks the similarity; the second pass restores it. */
        int pass;
        for(pass=0;pass<2;pass++)
          for(k=0;k<j;k++){
            emu_cplx s={0,0};
            for(i=0;i<m;i++) s=emu_add(s,emu_mul(emu_conj(AT(Qk,i,k,n)),AT(tmp,i,j,n)));
            AT(R,k,j,n)=emu_add(AT(R,k,j,n),s);
            for(i=0;i<m;i++) AT(tmp,i,j,n)=emu_sub(AT(tmp,i,j,n),emu_mul(s,AT(Qk,i,k,n)));
          }
        double nr=0; for(i=0;i<m;i++) nr+=emu_abs2(AT(tmp,i,j,n)); nr=sqrt(nr);
        AT(R,j,j,n).r=nr; AT(R,j,j,n).i=0;
        for(i=0;i<m;i++){ emu_cplx v=AT(tmp,i,j,n); if(nr>0){v.r/=nr;v.i/=nr;} else {v.r=(i==j);v.i=0;} AT(Qk,i,j,n)=v; }
      }
      /* T_active = R Qk + mu I ; the off-block parts transform too */
      for(j=0;j<m;j++) for(i=0;i<m;i++){
        emu_cplx s={0,0}; for(k=0;k<m;k++) s=emu_add(s,emu_mul(AT(R,i,k,n),AT(Qk,k,j,n)));
        if(i==j) s=emu_add(s,mu);
        AT(tmp,i,j,n)=s; }
      for(j=0;j<m;j++) for(i=0;i<m;i++) AT(T,i,j,n)=AT(tmp,i,j,n);
      /* columns right of the active block: T[0:m, m:] = Qk^H T[0:m, m:] */
      for(j=m;j<n;j++){
        for(i=0;i<m;i++){ emu_cplx s={0,0}; for(k=0;k<m;k++) s=emu_add(s,emu_mul(emu_conj(AT(Qk,k,i,n)),AT(T,k,j,n))); y[i]=s; }
        for(i=0;i<m;i++) AT(T,i,j,n)=y[i];
      }
      /* accumulate Q[:,0:m] = Q[:,0:m] Qk */
      for(i=0;i<n;i++){
        for(j=0;j<m;j++){ emu_cplx s={0,0}; for(k=0;k<m;k++) s=emu_add(s,emu_mul(AT(Q,i,k,n),AT(Qk,k,j,n))); y[j]=s; }
        for(j=0;j<m;j++) AT(Q,i,j,n)=y[j];
      }
    }
  }
  if (hi>0) *INFO=1;
  for(i=0;i<n;i++) W[i]=AT(T,i,i,n);
  /* eigenvectors of the triangular T, mapped back through Q */
  if (*JOBVR=='V' || *JOBVR=='v')
    for(k=0;k<n;k++){
      for(i=0;i<n;i++){ y[i].r=0; y[i].i=0; }
      y[k].r=1;
      for(i=k-1;i>=0;i--){
        emu_cplx s={0,0}, den=emu_sub(AT(T,i,i,n),AT(T,k,k,n));
        for(j=i+1;j<=k;j++) s=emu_add(s,emu_mul(AT(T,i,j,n),y[j]));
        if (emu_abs(den)<1e-14*anorm){ den.r=1e-14*anorm; den.i=0; }
        y[i]=emu_div(s,den); y[i].r=-y[i].r; y[i].i=-y[i].i;
      }
      double nr=0; int big=0; double bm=-1;
      for(i=0;i<n;i++){ emu_cplx s={0,0}; for(j=0;j<=k;j++) s=emu_add(s,emu_mul(AT(Q,i,j,n),y[j])); AT(VR,i,k,ldvr)=s; nr+=emu_abs2(s); if(emu_abs(s)>bm){bm=emu_abs(s);big=i;} }
      nr=sqrt(nr);
      { emu_cplx ph=AT(VR,big,k,ldvr); double pm=emu_abs(ph); ph.r/=pm; ph.i/=pm; ph=emu_conj(ph);
        for(i=0;i<n;i++){ emu_cplx v=emu_mul(AT(VR,i,k,ldvr),ph); v.r/=nr; v.i/=nr; AT(VR,i,k,ldvr)=v; } }
    }
  free(T);free(Q);free(Qk);free(R);free(tmp);free(y);
}
