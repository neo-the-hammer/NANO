/* Compare the stock selfH_new() (eigen-decomposition / mode matching)
 * with selfH_dec() (Lopez-Sancho decimation) on the same leads.
 * Not part of the shipped library -- see run_selfenergy_check.sh. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "complex.h"
#include "cmatrix.h"
#include "cfree_cmatrix.h"
#include "selfH_dec.h"
#include "cmatinv.h"

complex **selfH_new(double E, complex ***diag, complex ***updiag, complex ***lowdiag,
                    int N, int Nc, int lead, double eta);
complex **create_updiagGNR(int i, int n, double thop);
complex **create_lowdiagGNR(int i, int n, double thop);

static unsigned long long st = 0x2545F4914F6CDD1DULL;
static double rnd(void){ st^=st<<13; st^=st>>7; st^=st<<17; return (double)(st>>11)/9007199254740992.0*2-1; }

static complex ***blocks(int Nc, int n)
{ complex ***b = malloc(Nc*sizeof(*b)); int k,i,j;
  for(k=0;k<Nc;k++){ b[k]=cmatrix(0,n-1,0,n-1); for(i=0;i<n;i++) for(j=0;j<n;j++){b[k][i][j].r=0;b[k][i][j].i=0;} }
  return b; }

/* Graphene nanoribbon slices, built as GNR_charge_T.c builds them. */
static void build_gnr(int n, int Nc, double phi, complex ***D, complex ***U, complex ***L)
{ int i,j,k; complex **t;
  for(i=0;i<Nc;i++) for(j=0;j<n;j++) D[i][j][j].r = phi;
  for(i=0;i<Nc-1;i++){ t=create_updiagGNR(i,n,-2.7); for(j=0;j<n;j++) for(k=0;k<n;k++) U[i][j][k]=t[j][k]; cfree_cmatrix(t,0,n-1,0,n-1); }
  for(i=1;i<Nc;i++){ t=create_lowdiagGNR(i-1,n,-2.7); for(j=0;j<n;j++) for(k=0;k<n;k++) L[i][j][k]=t[j][k]; cfree_cmatrix(t,0,n-1,0,n-1); }
}

/* Random four-slice-periodic chain: slice s has on-site block D[s%4]
   (Hermitian) and couples to s+1 by C[s%4] (general complex); the
   reverse coupling is its conjugate transpose, so H is Hermitian. */
static void build_random(int n, int Nc, complex ***D, complex ***U, complex ***L)
{ complex Dt[4][8][8], Ct[4][8][8]; int s,i,j;
  for(s=0;s<4;s++) for(i=0;i<n;i++) for(j=0;j<n;j++){
    Ct[s][i][j].r=0.8*rnd()+(i==j?-1.5:0); Ct[s][i][j].i=0.5*rnd();
    if(j>=i){ Dt[s][i][j].r=(i==j?0.6*rnd():0.4*rnd()); Dt[s][i][j].i=(i==j?0:0.3*rnd()); Dt[s][j][i].r=Dt[s][i][j].r; Dt[s][j][i].i=-Dt[s][i][j].i; } }
  for(s=0;s<Nc;s++) for(i=0;i<n;i++) for(j=0;j<n;j++){
    D[s][i][j]=Dt[s%4][i][j];
    if(s<Nc-1) U[s][i][j]=Ct[s%4][i][j];
    if(s>0){ L[s][i][j].r=Ct[(s-1)%4][j][i].r; L[s][i][j].i=-Ct[(s-1)%4][j][i].i; } }
}

/* Residual of the lead's own fixed-point equation, which the exact
   self-energy satisfies whichever method produced it:
     sigma = BETA_30 [ (wmH - sigma placed at block (3,3))^-1 ]_00 BETADAGA_03
   Returned relative to |sigma|.  A method-independent accuracy check. */
static double residual(complex **sig, double E, complex ***D, complex ***U, complex ***L,
                       int n, int Nc, int lead, double eta)
{ int M=4*n,i,j,k,l; double r=0,m=0;
  complex **W=cmatrix(0,M-1,0,M-1),**B=cmatrix(0,M-1,0,M-1),**BD=cmatrix(0,M-1,0,M-1),**G;
  selfH_dec_cell(D,U,L,n,Nc,lead,W,B,BD);
  for(i=0;i<M;i++){ W[i][i].r=E+W[i][i].r; W[i][i].i+=eta; }
  for(i=0;i<n;i++) for(j=0;j<n;j++){ W[3*n+i][3*n+j].r-=sig[i][j].r; W[3*n+i][3*n+j].i-=sig[i][j].i; }
  G=cmatinv(W,M);
  for(i=0;i<n;i++) for(j=0;j<n;j++){
    double fr=0,fi=0;
    for(k=0;k<n;k++) for(l=0;l<n;l++){
      complex b=B[3*n+i][k], g=G[k][l], c=BD[l][3*n+j];
      double tr=b.r*g.r-b.i*g.i, ti=b.r*g.i+b.i*g.r;
      fr+=tr*c.r-ti*c.i; fi+=tr*c.i+ti*c.r; }
    r=fmax(r,hypot(fr-sig[i][j].r,fi-sig[i][j].i)); m=fmax(m,hypot(sig[i][j].r,sig[i][j].i)); }
  cfree_cmatrix(W,0,M-1,0,M-1); cfree_cmatrix(B,0,M-1,0,M-1); cfree_cmatrix(BD,0,M-1,0,M-1); cfree_cmatrix(G,0,M-1,0,M-1);
  return m>0?r/m:r;
}

static int compare(const char *label, int n, int Nc, complex ***D, complex ***U, complex ***L,
                   double eta, double tol)
{ double Es[] = {-2.31,-1.37,-0.71,-0.33,0.27,0.83,1.49,2.61}; int ok=1, e, lead;
  for(lead=0; lead<2; lead++){
    double worst=0, scale=0, wE=0, reig=0, rdec=0;
    for(e=0;e<8;e++){
      complex **a=selfH_new(Es[e],D,U,L,n,Nc,lead,eta), **b=selfH_dec(Es[e],D,U,L,n,Nc,lead,eta);
      double d=0,m=0; int i,j;
      for(i=0;i<n;i++) for(j=0;j<n;j++){ d=fmax(d,hypot(a[i][j].r-b[i][j].r,a[i][j].i-b[i][j].i)); m=fmax(m,hypot(a[i][j].r,a[i][j].i)); }
      if(m>0 && d/m>worst){ worst=d/m; wE=Es[e]; } if(m>scale) scale=m;
      reig=fmax(reig,residual(a,Es[e],D,U,L,n,Nc,lead,eta));
      rdec=fmax(rdec,residual(b,Es[e],D,U,L,n,Nc,lead,eta));
      cfree_cmatrix(a,0,n-1,0,n-1); cfree_cmatrix(b,0,n-1,0,n-1);
    }
    /* The eigen method is the more accurate of the two: it satisfies the
       lead's fixed-point equation to ~1e-14, while decimation loses
       accuracy as eta -> 0 (it needs ~1/eta effective doublings) and
       reaches ~2e-7 at eta = 1e-5 -- the same algorithm, and so the same
       accuracy, the stock GNR path (selfGNR -> Gzerozero) has always had.
       Pass if the two agree to tol; the residuals are printed so the
       accuracy trade stays visible. */
    { int pass = (worst<=tol);
      printf("  %-24s eta=%-6g %-6s |S_eig-S_dec|/|S|=%-8.1e residual: eig %-8.1e dec %-8.1e  %s\n",
             label, eta, lead?"drain":"source", worst, reig, rdec, pass?"ok":"MISMATCH");
      if(!pass) ok=0; }
  }
  return ok;
}

int main(void)
{ int ok=1, n, Nc=13;
  complex ***D,***U,***L;
  printf("selfH_new (eigen) vs selfH_dec (decimation), same leads\n");
  for(n=4;n<=8;n+=4){
    char lab[64];
    D=blocks(Nc,n); U=blocks(Nc,n); L=blocks(Nc,n); build_gnr(n,Nc,-0.2,D,U,L);
    sprintf(lab,"GNR slices, n=%d",n);
    ok&=compare(lab,n,Nc,D,U,L,1e-5,1e-6); ok&=compare(lab,n,Nc,D,U,L,1e-3,1e-6);
    D=blocks(Nc,n); U=blocks(Nc,n); L=blocks(Nc,n); build_random(n,Nc,D,U,L);
    sprintf(lab,"random periodic, n=%d",n);
    ok&=compare(lab,n,Nc,D,U,L,1e-5,1e-6); ok&=compare(lab,n,Nc,D,U,L,1e-3,1e-6);
  }
  printf("%s\n", ok?"SELF-ENERGY CHECK PASSED":"SELF-ENERGY CHECK FAILED");
  return ok?0:1;
}
