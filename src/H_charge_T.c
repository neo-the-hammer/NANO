// ======================================================================
//  Copyright (c) 2011, P. D'Amico, G. Fiori, G. Iannaccone  University of Pisa
//  
//  This file is released under the BSD license.
//  See the file "license.txt" for information on usage and
//  redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
// ====================================================================== 

#include "H_charge_T.h"
#include "vides_rgf_batch.h"
#include "selfH_dec.h"
#include <time.h>
static PyObject* py_H_charge_T(PyObject* self, PyObject* args)
{

  time_t t1, t2;
  int i, j, k, ix, ie, NE, n, Nc, nx, rows, columns, orbitals,rank,NUM,bandsflag,print_Hamiltonian;
  complex ***DIAG, ***UPDIAG, ***LOWDIAG;
  complex zero, **SIGMAS, **SIGMAD;
  double E, dE, Emin, Emax, *tempo, eta, A1sum, A2sum;
  double **A1, **A2, *ncarcnt, T, Eupper;
  double Elower, *Phi, *EE, *TE, mu1, mu2,Egap,*Ei,**DOS;
  double vt, Temperature, *orb,*testo;
  PyObject *obj,*temp_obj;
  PyArrayObject *Phi_array, *hamiltonian,*Ei_array,*ncar_array,
    *EE_array,*TE_array;
  //  Py_complex cc;
  //  complex ccc;
  
  setvbuf(stdout,(char *) NULL,_IONBF,1);
  print_Hamiltonian=0;
  
/*  
  Reads the data from python class.
*/
  
  import_array();
  if (!PyArg_ParseTuple(args,"O",&obj))
    {
      //printf"NOT A CLASS! \n");
      exit(0);
    }
  
  temp_obj=PyObject_GetAttrString(obj,"rank");
  rank=(int)PyInt_AsLong(temp_obj);
  
  if (!rank) printf("*****************************************\n");
  if (!rank) printf("*** Computing NEGF in HAMILTONIAN *******\n"); 
  if (!rank) printf("*****************************************\n\n");


  temp_obj=PyObject_GetAttrString(obj,"n");
  n=(int)PyInt_AsLong(temp_obj);
  if (!rank) printf("n # atoms in a slice = %d \n",n);
  temp_obj=PyObject_GetAttrString(obj,"Nc");
  Nc=(int)PyInt_AsLong(temp_obj);
  if (!rank) printf("Number of slices = %d \n",Nc);
  temp_obj=PyObject_GetAttrString(obj,"Eupper");
  Eupper=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Upper Energy defined by the user = %lg \n",Eupper);
  temp_obj=PyObject_GetAttrString(obj,"Elower");
  Elower=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Lower Energy defined by the user = %lg \n",Elower);
  temp_obj=PyObject_GetAttrString(obj,"eta");
  eta=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Eta = %lg \n",eta);
  temp_obj=PyObject_GetAttrString(obj,"dE");
  dE=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Delta Energy = %lg \n",dE);
  temp_obj=PyObject_GetAttrString(obj,"Temp");
  Temperature=(double)PyFloat_AsDouble(temp_obj);
  vt=kboltz*Temperature/q;
  if (!rank) printf("Vt = %lg \n",vt);
  if (!rank) printf("Temperature = %lg \n",Temperature);
  temp_obj=PyObject_GetAttrString(obj,"mu1");
  mu1=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Electrochemical potential of the source = %lg \n",mu1);
  temp_obj=PyObject_GetAttrString(obj,"mu2");
  mu2=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Electrochemical potential of the drain = %lg \n",mu2);
  temp_obj=PyObject_GetAttrString(obj,"Egap");
  Egap=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Egap of the material = %lg \n",Egap);
  
  if (PyObject_HasAttrString(obj,"Ham"))
    {
      temp_obj=PyObject_GetAttrString(obj,"Ham");
      print_Hamiltonian=(int)PyInt_AsLong(temp_obj);
    }

/*
  USE THE FOLLOWING TWO LINES FOR CELL REPRESENTATION, OTHERWISE COMMENT IT!!
*/

/*
  Nc = (Nc)/4;
  n = n*4;
*/

  Phi=dvector(0,n*Nc-1);
  for (i=0;i<n*Nc;i++)
    Phi[i]=0;
  Phi_array=(PyArrayObject*)PyObject_GetAttrString(obj,"Phi");
  nx=Phi_array->dimensions[0];
  for (i=0;i<nx;i++){
    tempo=(double *)(Phi_array->data+i*Phi_array->strides[0]);
    Phi[i]=*tempo;
  }

  Ei=dvector(0,n*Nc-1);
  Ei_array=(PyArrayObject*)PyObject_GetAttrString(obj,"Ei");
  nx=Ei_array->dimensions[0];
  for (i=0;i<nx;i++){
    tempo=(double *)(Ei_array->data+i*Ei_array->strides[0]);
    Ei[i]=*tempo;
  }


  
  hamiltonian = (PyArrayObject*)PyObject_GetAttrString(obj,"H");
  Py_DECREF(hamiltonian);

/*
  columns-2 is the number of orbitals (hoppings) of the problem.
  n*(columns-2) is built in order to construct the matrices with dimension (#atoms x #orbitals)^2
*/


  rows = hamiltonian->dimensions[0];
  columns = hamiltonian->dimensions[1];

  if (!rank) printf("Rows in the list= %i\n", rows);  
  if (!rank) printf("Columns in the list= %i\n", columns);


  orb =  (double *)(hamiltonian->data);
  orbitals = (int)(*orb);
  if (!rank) printf("# orbitals = %i \n", orbitals);
  if (!rank) printf("# dimension before = %i \n", n);

  NUM = n*orbitals;

  if (!rank) printf("# dimension after = %i \n", NUM);

  
  
/*
 Variable initialization and creation
*/

  ie=0;
  zero.r=0;
  zero.i=0;
  
  
  Emin=300;
  Emax=-10;

  ncarcnt=dvector(0,n*Nc-1);
  for (i=0;i<n*Nc;i++)
    {
      ncarcnt[i]=0;
    }

  A1=dmatrix(0,Nc-1,0,n-1);
  A2=dmatrix(0,Nc-1,0,n-1);

/*  
  The following computes the energy range in which NEGF 
  is solved if not specified by the user.
*/

  if ((Elower< -900)||(Eupper>900))
    
    for (i=0;i<n*Nc;i++){
      if (Emax<=(-Phi[i]))
	Emax=(-Phi[i]);
      if (Emin>=(-Phi[i]))
	Emin=(-Phi[i]);
    }
  
  
  Emin=min(Emin,(mu2));
  Emin=min((mu1),Emin);
  Emax=max(Emax,(mu2));
  Emax=max((mu1),Emax);
  if (!rank) printf("Emin =%lg, Emax =%lg, Egap = %lg, vt = %lg \n",Emin,Emax,Egap,vt);
  if (Eupper>900)
    Eupper=Emax+(Egap)*0.5+10*(vt);
  if (Elower<-900)
    Elower=Emin-(Egap)*0.5-10*(vt);
  
  if (!rank) printf("Eupper = %lg \n",Eupper);
  if (!rank) printf("Elower = %lg \n",Elower);
  if (!rank) printf("Current computed energy \n ");
  
 
/*
  NE is an approximation for the number of energy levels.
*/

  NE=(int)((Eupper-Elower)/dE);
/*
  ..... add 10 to NE, just to be sure to overestimate.
*/
  NE=NE+10;
  
/* 
  Once known the number of energy levels I create the energy level and
  the transmission coefficient matrix.
*/

  EE=dvector(0,NE-1);
  TE=dvector(0,NE-1);
  //  DOS=dmatrix(0,2*Nc-1,0,NE);

/*
  The DIAG, UPDIAG and LOWDIAG vectors of matrices are filled according to the format
  specified in the python class "hamiltonian". The parameters n and Nc has to be passed
  accordingly to the kind of hamiltonian one want to obtain, cell or slice one.
*/

  py_hamiltonian(hamiltonian, n, Nc, rows, orbitals, Phi, &DIAG, &UPDIAG, &LOWDIAG);

/*   // I check if I have to compute the bands or not */
//
//   CHECK THIS OUT: IF I UNCOMMENT THIS PART, WHEN RUNNING
//   IN THE ../TEST DIRECTORY THE SCRIPT TEST_BILAYER_CLASS.PY
//   I GOT THE ERROR "INTEGER REQUIRED"
//  

/*   temp_obj=PyObject_GetAttrString(obj,"bands"); */
/*   bandsflag=(int)PyInt_AsLong(temp_obj); */
/*   if (PyObject_GetAttrString(obj,"bands")!=NULL) */
/*     if (bandsflag==1) */
/*       { */
/* 	printf(" Computing the bands......\n"); */
/*        	bands(DIAG,UPDIAG,LOWDIAG,NUM); */
/* 	printf("\n"); */
/* 	printf("--------------------------------\n"); */
/* 	printf("\n"); */
/* 	printf("The bands are saved in the file Bands.dat \n"); */
/* 	printf("\n"); */
/* 	printf("--------------------------------\n"); */
/* 	printf("\n"); */
/* 	//	exit(0); */
/*       } */

/*
  The following loops write the matrices on 3 files
*/

  if (print_Hamiltonian==1)
    if (!rank)
      {
	FILE *fp;
	fp=fopen("H_diagH","w");
	for (k=0;k<Nc;k++){
	  for (i=0;i<NUM;i++){
	    for (j=0;j<NUM;j++)
	      fprintf(fp,"%lg+i%lg ",DIAG[k][i][j].r,DIAG[k][i][j].i);
	    // I print the real part!
	    fprintf(fp,"\n");
	  }
	  //      fprintf(fp,"***********************\n");
	  fprintf(fp,"\n");
	}
	fclose(fp);
  
	fp=fopen("H_updiagH","w");
	for (k=0;k<Nc;k++){
	  //fprintf(fp,"INDEX  %d\n",k);
	  for (i=0;i<NUM;i++){
	    for (j=0;j<NUM;j++){
	      fprintf(fp,"%lg+i%lg ",UPDIAG[k][i][j].r,UPDIAG[k][i][j].i);
	    }
	    fprintf(fp,"\n");
	  }
	  //      fprintf(fp,"***********************\n");
	  fprintf(fp,"\n");
	}
	fclose(fp);
  
	fp=fopen("H_lowdiagH","w");
	for (k=0;k<Nc;k++){
	  // fprintf(fp,"INDEX  %d\n",k);
	  for (i=0;i<NUM;i++){
	    for (j=0;j<NUM;j++){
	      fprintf(fp,"%lg+i%lg ",LOWDIAG[k][i][j].r,LOWDIAG[k][i][j].i);
	    }
	    fprintf(fp,"\n");
	  }
	  //      fprintf(fp,"***********************\n");
	  fprintf(fp,"\n");
	}
	fclose(fp);
	printf("\n\n\n\n\n\nHamiltonian has been printed in the following files\n");
	printf("H_diag    : Diagonal block\n");
	printf("H_updiag  : Up-Diagonal block\n");
	printf("H_lowdiag : Low-Diagonal block\n\n\n\n\n");
	exit(0);
      }
    
//  SIGMAS=cmatrix(0,n-1,0,n-1);
//  SIGMAD=cmatrix(0,n-1,0,n-1);
  
  E=Elower;

  // The energy points are independent of one another, so instead of one
  // NEGF solve per energy they are handed to vides_rgf_batch() a chunk at
  // a time.  This path uses the Lake recursion, matching the LDOS_Lake()
  // call it replaces.  With a GPU present the chunk becomes the batch
  // dimension of the cuBLAS batched calls; without one the driver loops
  // over the chunk calling the same LDOS_Lake() as before.
  {
    vides_rgf_desc vdesc;
    complex ***SSch,***SDch;
    double *Ech,*A1ch,*A2ch,*Tch;
    int NBmax,nb,ib,use_dec,pipe,cur,nbc[2];
    double *E2[2]={NULL,NULL},*T2[2]={NULL,NULL},*A12[2]={NULL,NULL},*A22[2]={NULL,NULL};
    complex ***SS2[2]={NULL,NULL},***SD2[2]={NULL,NULL};
    vides_rgf_job *job;
    double t_neg0,t_self=0,t_solve=0,t0;
    complex **W0s=NULL,**W0d=NULL,**Bs=NULL,**BDs=NULL,**Bd=NULL,**BDd=NULL;

    vdesc.n=NUM;
    vdesc.Nc=Nc;
    vdesc.NB=1;
    vdesc.variant=VIDES_RGF_LAKE;
    vdesc.flagtrans=1;
    vdesc.eta=eta;
    vdesc.Nreal=0;
    vdesc.order=NULL;

    NBmax=vides_negf_chunk(&vdesc);
    if (NBmax>NE) NBmax=NE;
    if (NBmax<1) NBmax=1;

    if (!rank) printf("NEGF backend: %s, %d energies per batch \n",
                      vides_gpu_describe(),NBmax);

    // Contact self-energy method (see vides_selfh_use_decimation()).
    use_dec=vides_selfh_use_decimation(eta);
    // Pipeline (see vides_rgf_batch.h): with the eigen self-energy, which
    // runs on the CPU only, the GPU solves chunk k while the CPU threads
    // build the self-energies of chunk k+1, so two chunk buffers.
    pipe=(!use_dec)&&vides_pipeline_enabled();
    for (ib=0;ib<(pipe?2:1);ib++)
      {
        E2[ib] =(double *)malloc((size_t)NBmax*sizeof(double));
        T2[ib] =(double *)malloc((size_t)NBmax*sizeof(double));
        SS2[ib]=(complex ***)malloc((size_t)NBmax*sizeof(complex **));
        SD2[ib]=(complex ***)malloc((size_t)NBmax*sizeof(complex **));
        A12[ib]=(double *)malloc((size_t)NBmax*Nc*NUM*sizeof(double));
        A22[ib]=(double *)malloc((size_t)NBmax*Nc*NUM*sizeof(double));
        if (!E2[ib]||!T2[ib]||!SS2[ib]||!SD2[ib]||!A12[ib]||!A22[ib])
          {
            printf("Out of memory allocating the NEGF energy batch \n");
            exit(0);
          }
      }

    // The leads' decimation cells are energy independent: build them once.
    if (!rank) printf("Contact self-energy: %s%s, %d CPU threads \n",
                      use_dec ? "decimation (selfH_dec)" : "eigen-decomposition (selfH_new)",
                      pipe ? ", pipelined with the GPU solve" : "",
                      vides_cpu_threads());
    if (use_dec)
      {
        W0s=cmatrix(0,4*NUM-1,0,4*NUM-1);  Bs=cmatrix(0,4*NUM-1,0,4*NUM-1);
        BDs=cmatrix(0,4*NUM-1,0,4*NUM-1);  W0d=cmatrix(0,4*NUM-1,0,4*NUM-1);
        Bd=cmatrix(0,4*NUM-1,0,4*NUM-1);   BDd=cmatrix(0,4*NUM-1,0,4*NUM-1);
        selfH_dec_cell(DIAG,UPDIAG,LOWDIAG,NUM,Nc,0,W0s,Bs,BDs);
        selfH_dec_cell(DIAG,UPDIAG,LOWDIAG,NUM,Nc,1,W0d,Bd,BDd);
      }

    // Fill buffer k with the next chunk of energies and their self-energies
    // (lead = 0(1) corresponds to right(left) lead).  Returns its size.
#define FILL_CHUNK(k) do {                                                  \
      nb=0;                                                                 \
      while ((nb<NBmax)&&(E<=(Eupper+dE*0.5)))                              \
        { E2[k][nb]=E; nb++; E+=dE; }                                       \
      if (nb>0) {                                                           \
        if (use_dec)                                                        \
          {                                                                 \
            vides_decimation_batch(4*NUM,nb,E2[k],eta,W0s,Bs,BDs,3*NUM,NUM,SS2[k]); \
            vides_decimation_batch(4*NUM,nb,E2[k],eta,W0d,Bd,BDd,3*NUM,NUM,SD2[k]); \
          }                                                                 \
        else                                                                \
          {                                                                 \
            vides_selfH_new_batch(nb,E2[k],DIAG,UPDIAG,LOWDIAG,NUM,Nc,0,eta,SS2[k]); \
            vides_selfH_new_batch(nb,E2[k],DIAG,UPDIAG,LOWDIAG,NUM,Nc,1,eta,SD2[k]); \
          }                                                                 \
      }                                                                     \
      nbc[k]=nb;                                                            \
    } while (0)

    t_neg0=vides_now();
    cur=0;
    t0=vides_now();
    FILL_CHUNK(0);
    t_self+=vides_now()-t0;
    while (nbc[cur]>0){

      Ech=E2[cur]; Tch=T2[cur]; SSch=SS2[cur]; SDch=SD2[cur];
      A1ch=A12[cur]; A2ch=A22[cur];
      vdesc.NB=nbc[cur];
      if (pipe)
        {
          // GPU solve of this chunk in the background ...
          job=vides_rgf_start(&vdesc,Ech,DIAG,UPDIAG,LOWDIAG,SSch,SDch,
                              A1ch,A2ch,Tch);
          // ... while the CPU threads prepare the next one.
          t0=vides_now();
          FILL_CHUNK(1-cur);
          t_self+=vides_now()-t0;
          t0=vides_now();
          if (vides_rgf_wait(job)!=0)
            {
              printf("NEGF batch failed \n");
              exit(0);
            }
          t_solve+=vides_now()-t0;   // only the part not hidden
        }
      else
        {
          t0=vides_now();
          if (vides_rgf_batch(&vdesc,Ech,DIAG,UPDIAG,LOWDIAG,SSch,SDch,
                              A1ch,A2ch,Tch)!=0)
            {
              printf("NEGF batch failed \n");
              exit(0);
            }
          t_solve+=vides_now()-t0;
        }
      nb=nbc[cur];

      for (ib=0;ib<nb;ib++)
        {
          double Eb,*A1b,*A2b;
          Eb=Ech[ib];
          A1b=A1ch+(size_t)ib*Nc*NUM;
          A2b=A2ch+(size_t)ib*Nc*NUM;

          cfree_cmatrix(SSch[ib],0,NUM-1,0,NUM-1);
          cfree_cmatrix(SDch[ib],0,NUM-1,0,NUM-1);

            // Compute the charge
            for (i=0;i<Nc;i++){
              for (j=0;j<n;j++)
        	{
        	  // electrons
        	  ix=j+i*n;
	  
        	  // For each atom, I sum over the total number of orbitals
        	  A1sum=0;
        	  A2sum=0;
        	  for (k=0;k<orbitals;k++)
        	    {
        	      A1sum+=A1b[i*NUM+k+j*orbitals];
        	      A2sum+=A2b[i*NUM+k+j*orbitals];
        	    }

        	  if (A1sum>=1e-40)
        	    if ((Eb>=Ei[j+i*n]))
        	      {
        		ncarcnt[ix]=ncarcnt[ix]
        		  -2*(
        		      A1sum*Fermi_Dirac((Eb-(mu1))/(vt))*dE);
        	      }

        	  if (A2sum>=1e-40)
        	    if ((Eb>=Ei[j+i*n]))
        	      {
        		ncarcnt[ix]=ncarcnt[ix]
        		  -2*(
        		      A2sum*Fermi_Dirac((Eb-(mu2))/(vt))*dE);
        	      }
	  
        	  //holes
        	  if (A1sum>=1e-40)
        	    if ((Eb<Ei[j+i*n]))
        	      {
        		ncarcnt[ix]=ncarcnt[ix]
        		  +2*(
        		      A1sum*(1-Fermi_Dirac((Eb-(mu1))/(vt)))*dE);
        	      }
	  
        	  if (A2sum>=1e-40)
        	    if ((Eb<Ei[j+i*n]))
        	      {
        		ncarcnt[ix]=ncarcnt[ix]
        		  +2*(
        		      A2sum*(1-Fermi_Dirac((Eb-(mu2))/(vt)))*dE);
        	      }
        	  // last modifications
        	  //	  DOS[ix][ie]+=A1sum+A2sum;
	  
        	}
            }  
    

          EE[ie]=Eb;
          TE[ie]=Tch[ib];
          ie++;
        }

      if (!pipe)
        {
          t0=vides_now();
          FILL_CHUNK(0);
          t_self+=vides_now()-t0;
        }
      else
        cur=1-cur;
    }
#undef FILL_CHUNK

    vides_profile_report("Hamiltonian",vides_now()-t_neg0,t_self,t_solve);

    if (use_dec)
      {
        cfree_cmatrix(W0s,0,4*NUM-1,0,4*NUM-1);  cfree_cmatrix(Bs,0,4*NUM-1,0,4*NUM-1);
        cfree_cmatrix(BDs,0,4*NUM-1,0,4*NUM-1);  cfree_cmatrix(W0d,0,4*NUM-1,0,4*NUM-1);
        cfree_cmatrix(Bd,0,4*NUM-1,0,4*NUM-1);   cfree_cmatrix(BDd,0,4*NUM-1,0,4*NUM-1);
      }

    for (ib=0;ib<2;ib++)
      {
        free(E2[ib]); free(T2[ib]); free(SS2[ib]); free(SD2[ib]);
        free(A12[ib]); free(A22[ib]);
      }
  }

  if (!rank) printf("\n\n*****************************************\n");
  if (!rank) printf("****** END OF NEGF in HAMILTONIAN *******\n"); 
  if (!rank) printf("*****************************************\n\n");

  //  (void) time(&t2); 
  //  printf("time LDOS on rank %d = %d secs \n", rank,(int) t2-t1);

/*   FILE *fp3; */
/*   fp3=fopen("DOS.out","w"); */
/*   for (i=0;i<2*Nc;i++) */
/*     { */
/*       for (j=0;j<ie;j++) */
/* 	fprintf(fp3,"%lg %d %lg \n",EE[j],i,DOS[i][j]); */
/*       fprintf(fp3,"\n"); */
/*     } */
  

/*   exit(0); */




/*
  Set the attribute of the class to send data out.
*/

/*   temp_obj=  PyArray_FromDimsAndData(1,&ie,PyArray_DOUBLE,(char*)EE); */
/*   PyObject_SetAttrString(obj,"E",temp_obj); */
/*   temp_obj=  PyArray_FromDimsAndData(1,&ie,PyArray_DOUBLE,(char*)TE); */
/*   PyObject_SetAttrString(obj,"T",temp_obj); */
/*   ie=Nc*n; */
/*   temp_obj=  PyArray_FromDimsAndData(1,&ie,PyArray_DOUBLE,(char*)ncarcnt); */
/*   PyObject_SetAttrString(obj,"charge",temp_obj); */

  // I return back the free_charge, EE, TE
  ncar_array=(PyArrayObject*)PyObject_GetAttrString(obj,"charge");
  for (i=0;i<n*Nc;i++)
    *(double *)(ncar_array->data+i*ncar_array->strides[0])=ncarcnt[i];
  free_dvector(ncarcnt,0,n*Nc-1);

  EE_array=(PyArrayObject*)PyObject_GetAttrString(obj,"E");
  for (i=0;i<ie;i++)
    *(double *)(EE_array->data+i*EE_array->strides[0])=EE[i];
  free_dvector(EE,0,NE-1);

  TE_array=(PyArrayObject*)PyObject_GetAttrString(obj,"T");
  for (i=0;i<ie;i++)
    *(double *)(TE_array->data+i*TE_array->strides[0])=TE[i];
  free_dvector(TE,0,NE-1);

  Py_DECREF(EE_array);
  Py_DECREF(TE_array);
  Py_DECREF(ncar_array);
  Py_DECREF(Phi_array);
  Py_DECREF(Ei_array);

/*  
  Free the memory.
*/

  cfree_c3tensor(DIAG,0,Nc-1,0,NUM-1,0,NUM-1);
  cfree_c3tensor(LOWDIAG,0,Nc-1,0,NUM-1,0,NUM-1);
  cfree_c3tensor(UPDIAG,0,Nc-1,0,NUM-1,0,NUM-1);
  free_dvector(Phi, 0, n*Nc-1);
  free_dvector(Ei, 0, n*Nc-1);
  //  free_dvector(EE, 0, NE-1);
  //  free_dvector(TE, 0, NE-1);
  //  free_dvector(ncarcnt,0,n*Nc-1);

  char *s = "Hello from CNT_charge_T";
  return Py_BuildValue("s", s);
 
}  
