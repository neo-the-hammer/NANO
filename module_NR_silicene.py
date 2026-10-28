#Copyright (c) 2004-2015, G. Fiori, G. Iannaccone, University of Pisa.
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#    - Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#    - Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#    - All advertising materials mentioning features or use of this software
#        must display the following acknowledgement:
#        This product includes software developed by G.Fiori and G.Iannaccone at
#        University of Pisa.
#    - Neither the name of the University of Pisa nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY
#EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE AUTHORS
#BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#                       SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#                       INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#THE POSSIBILITY OF SUCH DAMAGE.

from NanoTCAD_ViDES import *
class NR_silicene:
    acc=0.384;
    def __init__(self,n,L):
        self.Nc=int(4*(floor((floor(L/NR_silicene.acc)-1)/3)));
        self.n=n;
        self.Phi=zeros(n*self.Nc);
        self.Eupper=1000.0;
        self.Elower=-1000.0;
        self.dE=1e-3;
        self.thop=-0.735;
        self.eta=1e-8;
        self.mu1=0;
        self.mu2=0;
        self.Temp=300;
        self.E=zeros(int(NEmax));
        self.T=zeros(int(NEmax));
        self.charge=zeros(self.n*self.Nc);
        self.rank=0;
        self.atoms_coordinates();
        self.defects_list=[]
        self.onsite_E=-1.5;
    def atoms_coordinates(self):
        GNR_atoms_coordinates(self);
        self.x=array(self.x);
        self.y=array(self.y);
        self.z=array(self.z);
        return;
    def gap(self):
        return GNRgap(self);
    def charge_T(self):
        
        M=self.Nc;
        N=self.n;
        t=self.thop;
        Energy = 0.0
        Ene = 0.0
        p = 0.0
        d = 0.0
        orbitals = [1, 0]
        hamiltonian = []
        zeroes = [0, 0, 0, 0]
        ene = [Energy, 0, 0, Ene]
        coupling1 = [t, 0, 0, p]
        coupling2 = [t*1.12, 0, 0, p]
        orbitals = orbitals + zeroes
        hamiltonian.append(orbitals)
        
        for j in range(M):
            for i in range(N):
                n = i + 1 + j*N
                p = [n,n]
                p =  p + ene
                hamiltonian.append(p)
        
        
        
        for j in range(1, M-1, +4):
            for i in range(1, N):
                n = i + 1 + j*N
                m = i + (j+1)*N
                p = [n,m]
                p =  p + coupling1
                hamiltonian.append(p)
        #      hamiltonian.append([m, n, t, p, d])
    
        for j in range(3, M-1, +4):
            for i in range(0, N-1):
                n = i + 1 + j*N
                m = i + 2 + (j+1)*N
                p = [n,m]
                p =  p + coupling1
                hamiltonian.append(p)
        #     hamiltonian.append([m, n, t, p, d])


        # nell'if ripristinare il fattore t*1.12
        for j in range(0, M-1, +4):
            for i in range(N):
                n = i + 1 + j*N
                m = i + 1 + (j+1)*N
                if i == 0:
                    p = [n,m]
                    p =  p + coupling2
                    hamiltonian.append(p)
                #          hamiltonian.append([m, n, t*1.12, p, d])
                else :
                    p = [n,m]
                    p =  p + coupling1
                    hamiltonian.append(p)
        #         hamiltonian.append([m, n, t, p, d])

        for j in range(1, M-1, +4):
            for i in range(N):
                n = i + 1 + j*N
                m = i + 1 + (j+1)*N
                p = [n,m]
                p =  p + coupling1
                hamiltonian.append(p)
        #            hamiltonian.append([m, n, t, p, d])

        # nell'if ripristinare il fattore t*1.12
        for j in range(2, M-1, +4):
            for i in range(N):
                n = i + 1 + j*N
                m = i + 1 + (j+1)*N
                if i == (N-1):
                    p = [n,m]
                    p =  p + coupling2
                    hamiltonian.append(p)
                #                 hamiltonian.append([m, n, t*1.12, p, d])
                else :
                    p = [n,m]
                    p =  p + coupling1
                    hamiltonian.append(p)
        #                hamiltonian.append([m, n, t, p, d])

        for j in range(3, M-1, +4):
            for i in range(N):
                n = i + 1 + j*N
                m = i + 1 + (j+1)*N
                p = [n,m]
                p =  p + coupling1
                hamiltonian.append(p)
        #         hamiltonian.append([m, n, t, p, d])

        H = Hamiltonian(N,M)
        # I work on the defects
        ind=array(self.defects_list,dtype=int);
        H.H=array(hamiltonian,dtype=complex)
        H.H[ind,2]=self.onsite_E;
        H.Eupper = self.Eupper;
        H.Elower = self.Elower;
        H.rank=self.rank;
        H.dE=self.dE;
        H.Phi=self.Phi;
        H.Ei=-self.Phi;
        H.eta=self.eta;
        H.mu1=self.mu1;
        H.mu2=self.mu2;
        H.Egap=self.gap();

        H.charge_T()

        self.E=array(H.E);
        self.T=array(H.T);
        self.charge=array(H.charge);
        del hamiltonian,H
        return;

    def current(self):
        vt=kboltz*self.Temp/q;
        E=array(self.E);
        T=array(self.T);
        arg=2*q*q/(2*pi*hbar)*T*(Fermi((E-self.mu1)/vt)-Fermi((E-self.mu2)/vt))*self.dE
        return sum(arg);
