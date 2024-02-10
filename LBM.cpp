#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>


using namespace std;

#define TX 200
#define TY 200
#define TZ 200
#define PX 10
#define PY 10
#define PZ 10
#define Q 19
#define QG 7

const int ex[Q]={0, 1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0};
const int ey[Q]={0, 0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1};
const int ez[Q]={0, 0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1};
const int op[Q]={0, 2,  1,  4,  3,  6,  5, 10,  9,  8,  7,  14, 13,12, 11, 18, 17, 16, 15};
const double w[Q]={1.0/3,1.0/18,1.0/18,1.0/18,1.0/18,1.0/18,1.0/18,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36};
const double J[QG]={1.0/4,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8};

int i, j, k, l, n;

double dx, dy, dz, dt;
double rhoi, rholA, rhogA, rholB, rhogB, uxi, uyi, uzi;
double niuA, niuB, xiA, xiB;

double ***rho, ***rho0, ***ux, ***uy, ***uz, ***ux0, ***uy0, ***uz0, ***p;

double ***rhoA, ***rhoA0;
double ***phiA, ***psiA, ***TscA, ***pA;
double ****fA, mA[Q], ****fApost, MA[Q], sA[Q];
double ***fscABx, ***fscABy, ***fscABz, ***fscAAx, ***fscAAy, ***fscAAz, ***fscAx, ***fscAy, ***fscAz;
double ****sscA;
double GAB, ***GAA, rA;

double ***rhoB, ***rhoB0;
double ***phiB, ***psiB, ***TscB, ***pB;
double ****fB, mB[Q], ****fBpost, MB[Q], sB[Q];
double ***fscBAx, ***fscBAy, ***fscBAz, ***fscBBx, ***fscBBy, ***fscBBz, ***fscBx, ***fscBy, ***fscBz;
double ****sscB;
double ***GBB, rB;

double ***solid, ***bounce;

int ***data, ***LG;

double rhoAin, rhoBin, uxin, uyin, uzin;
double rhoAout, rhoBout, uxout, uyout, uzout;
double lA, lB;
double gvx, gvy, gvz;

double error;

double ***vof, ***dvofx, ***dvofy, ***dvofz, ***ddvof;

int destright, destleft, destfront, destback, destup, destdown;
double *send_right, *send_left, *send_front, *send_back, *send_up, *send_down;
double *recv_right, *recv_left, *recv_front, *recv_back, *recv_up, *recv_down;
double *sendfA_right, *sendfA_left, *sendfA_front, *sendfA_back, *sendfA_up, *sendfA_down;
double *recvfA_right, *recvfA_left, *recvfA_front, *recvfA_back, *recvfA_up, *recvfA_down;
double *sendfB_right, *sendfB_left, *sendfB_front, *sendfB_back, *sendfB_up, *sendfB_down;
double *recvfB_right, *recvfB_left, *recvfB_front, *recvfB_back, *recvfB_up, *recvfB_down;
double *sendmacro_right, *sendmacro_left, *sendmacro_front, *sendmacro_back, *sendmacro_up, *sendmacro_down;
double *recvmacro_right, *recvmacro_left, *recvmacro_front, *recvmacro_back, *recvmacro_up, *recvmacro_down;
const int tagrfA=1001, tagrfB=1002, taglfA=1003, taglfB=1004, tagffA=1005, tagffB=1006, tagbfA=1007, tagbfB=1008, tagufA=1009, tagufB=1010, tagdfA=1011, tagdfB=1012;
const int tagrm=2001, taglm=2002, tagfm=2003, tagbm=2004, tagum=2005, tagdm=2006;
const int Nmacro=2;//rhoA, rhoB

int mpisize, mpirank, rankx, ranky, rankz;
int lengthx, lengthy, lengthz;
int startx, starty, startz;
int endx, endy, endz;
int NX, NY, NZ;

double ***C, ***C0;
double ***CA, ***CB, ***CA0, ***CB0;
double ****Scst;
double ****g, ****gpost, mg[QG], Mg[QG], ****sg;
double DA, DB;
double ***kr, ***Ceq;
double kr0, Ceq0;
double ***D;
double MMA, MMB, Vm, H;
double Cinitial, Cinlet, Coutlet;
double errorC;
int upd;

double *sendg_right, *sendg_left, *sendg_front, *sendg_back, *sendg_up, *sendg_down;
double *recvg_right, *recvg_left, *recvg_front, *recvg_back, *recvg_up, *recvg_down;
const int tagrg=3001, taglg=3002, tagfg=3003, tagbg=3004, tagug=3005, tagdg=3006;
const int tagrC=4001, taglC=4002, tagfC=4003, tagbC=4004, taguC=4005, tagdC=4006;
const int tagrS=5001, taglS=5002, tagfS=5003, tagbS=5004, taguS=5005, tagdS=5006;

double ***Pnu, ***Pcr;
double ***chi, ***Ceqnu, ***Krnu;

double vmnu, gammanu, kBnu, NAnu, Tnu, J0nu;
int tlim, tnu;
double rnu, Jnu, Ksp;

double ***SmA, ***SmB;
double ***SmC;

void ProcessorID();
void Parameter();
double meq(int l,double rho,double ux,double uy,double uz);
double feq(int l,double rho,double ux,double uy,double uz);
void Phipsi();
void Initial();
void Initial2();
void Fsc();
void Collision();
void Infosendrecvf();
void Boundary();
void Boundary2();
void InletZH();
void Streaming();
void Macro();
void Infosendrecvmacro();
void Getbounce();//inletoutlet
void Phisolid();//inletoutlet
void Phisolid2();
void Vof();//inletoutlet
void Input();
void Output(int m);
void Errorcheck();

void Parameterg();
double geq(int l,double C,double ux,double uy,double uz);
double mgeq(int l,double C,double ux,double uy,double uz);
void Initialg();
void Dproperty();
void CSTsource();
void Collisiong();
void Infosendrecvg();
void Boundaryg();
void Boundaryg2();
void Inletg();
void Outletg();
void Streamingg();
void Macrog();
void InfosendrecvC();
void Infosendrecvsolid();
void Getcacb();
void Errorcheckg();

void Parameternu();
void Initialnu();
void Nuclear();
void Nuclearevolution();
void Nuclearupdate();

int main(int argc,char**argv)
{
    double startwtime, endwtime;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    startwtime=MPI_Wtime();
    ProcessorID();
    Parameter();
    Parameternu();
    Parameterg();
    Input();
    Initial();
    Initialnu();
    Initialg();
    //Initial2();

    int ns=100000;

    for(n=0;;n++)
    {
        if(n<ns)
        {
            Fsc();
            Collision();
            Infosendrecvf();
            Boundary();
            Streaming();
            Macro();
            Infosendrecvmacro();//？
            Phisolid();
            Vof();
        }

        if(n==ns)
        {
            Initial2();
            Initialnu();
            Initialg();
        }

        if(n>=ns)
        {
            Fsc();
            Collision();
            Infosendrecvf();
            Boundary();
            Streaming();
            InletZH();
            CSTsource();
            Collisiong();
            Infosendrecvg();
            Boundaryg2();
            Streamingg();

            Macro();
            Infosendrecvmacro();//？
            Phisolid2();
            Macrog();
            InfosendrecvC();
            Nuclear();
            Nuclearevolution();
            Infosendrecvsolid();
            Nuclearupdate();
            Vof();
            Dproperty();
            Getcacb();
        }
        

        if(n%100==0)
        {
            Errorcheck();
            Errorcheckg();
        }

        if(n%10000==0) Output(n);
        if(n>2100000) break;
    }

    
    endwtime=MPI_Wtime();
    MPI_Finalize();
    return 0;
}

double meq(int l,double rho,double ux,double uy,double uz)
{
	double meq;
		switch(l)
	    {
		case 0: {meq=rho; break;}
		case 1: {meq=-11.0*rho+19.0*rho*(ux*ux+uy*uy+uz*uz); break;}
		case 2: {meq=3.0*rho-11.0/2*rho*(ux*ux+uy*uy+uz*uz); break;}
		case 3: {meq=rho*ux; break;}
		case 4: {meq=-2.0/3*rho*ux; break;}
		case 5: {meq=rho*uy; break;}
		case 6: {meq=-2.0/3*rho*uy; break;}
		case 7: {meq=rho*uz; break;}
		case 8: {meq=-2.0/3*rho*uz; break;}
		case 9: {meq=2.0*rho*ux*ux-rho*(uy*uy+uz*uz); break;}
		case 10: {meq=-1.0/2*(2.0*rho*ux*ux-rho*(uy*uy+uz*uz)); break;}
		case 11: {meq=rho*(uy*uy-uz*uz); break;}
		case 12: {meq=-1.0/2*rho*(uy*uy-uz*uz); break;}
		case 13: {meq=rho*ux*uy; break;}
		case 14: {meq=rho*uy*uz; break;}
		case 15: {meq=rho*ux*uz; break;}
		case 16: {meq=0.0; break;}
		case 17: {meq=0.0; break;}
		case 18: {meq=0.0; break;}
		default: meq=0.0;
		}
		return meq;
}

double feq(int l,double rho,double ux,double uy,double uz)
{
	double eu,uv,feq;
	eu=ex[l]*ux+ey[l]*uy+ez[l]*uz;
	uv=ux*ux+uy*uy+uz*uz;
	feq=w[l]*rho*(1.0+3.0*eu+4.5*eu*eu-1.5*uv);
	return feq;
}

void ProcessorID()
{
    rankz=mpirank/(PX*PY);
    ranky=(mpirank-rankz*PX*PY)/PX;
    rankx=mpirank-rankz*PX*PY-ranky*PX;

    if (rankx<(TX%PX))
    {
        lengthx=TX/PX+1;
        startx=rankx*lengthx-1;
        endx=startx+lengthx-1;
    }
    else
    {
        lengthx=TX/PX;
        startx=rankx*lengthx+TX%PX-1;
        endx=startx+lengthx-1;
    }

    if (ranky<(TY%PY))
    {
        lengthy=TY/PY+1;
        starty=ranky*lengthy-1;
        endy=starty+lengthy-1;
    }
    else
    {
        lengthy=TY/PY;
        starty=ranky*lengthy+TY%PY-1;
        endy=starty+lengthy-1;
    }

    if (rankz<(TZ%PZ))
    {
        lengthz=TZ/PZ+1;
        startz=rankz*lengthz-1;
        endz=startz+lengthz-1;
    }
    else
    {
        lengthz=TZ/PZ;
        startz=rankz*lengthz+TZ%PZ-1;
        endz=startz+lengthz-1;
    }

    NX=lengthx+2;
    NY=lengthy+2;
    NZ=lengthz+2;
    destright=(rankx+1)%PX+ranky*PX+rankz*PX*PY;
    destleft=(rankx+PX-1)%PX+ranky*PX+rankz*PX*PY;
    destfront=rankx+(ranky+1)%PY*PX+rankz*PX*PY;
    destback=rankx+(ranky+PY-1)%PY*PX+rankz*PX*PY;
    destup=rankx+ranky*PX+(rankz+1)%PZ*PX*PY;
    destdown=rankx+ranky*PX+(rankz+PZ-1)%PZ*PX*PY;
    
}

void Parameter()
{
    dx=1.0;
    dy=1.0;
    dz=1.0;
    dt=1.0;

    rhoi=1.0;
    uxi=0.0;
    uyi=0.0;
    uzi=0.0;

    rholA=6.25;
    rhogA=0.20;
    rholB=0.62;
    rhogB=0.04;
    GAB=1.3;
    rA=6.24;
    rB=0.59;
    lA=1.5;
    lB=1.0;

    gvx=0.0;
    gvy=0.0;
    gvz=0.0;

    rhoAin=rholA;
    rhoBin=rhogB;
    rhoAout=rhogA;
    rhoBout=rholB;

    double weA=1.4;
    double wqA=1.2;
    double wpiA=1.4;
    double wmA=1.98;
    double dA=1.0;
    niuA=1.0;
    xiA=0.5;
    sA[0]=0.0;
	sA[1]=1.0/(9.0/2*xiA+0.5);
	sA[2]=weA;
	sA[3]=0.0;
    sA[3]=1.0/(dA+0.5);
	sA[4]=wqA;
	sA[5]=0.0;
    sA[5]=1.0/(dA+0.5);
	sA[6]=wqA;
	sA[7]=0.0;
    sA[7]=1.0/(dA+0.5);
	sA[8]=wqA;
	sA[9]=1.0/(3.0*niuA+0.5);
	sA[10]=wpiA;
	sA[11]=1.0/(3.0*niuA+0.5);
	sA[12]=wpiA;
	sA[13]=1.0/(3.0*niuA+0.5);
	sA[14]=1.0/(3.0*niuA+0.5);
	sA[15]=1.0/(3.0*niuA+0.5);
	sA[16]=wmA;
	sA[17]=wmA;
	sA[18]=wmA;

    double weB=1.4;
    double wqB=1.2;
    double wpiB=1.4;
    double wmB=1.98;
    double dB=1.0;
    niuB=1.0;
    xiB=0.5;
    sB[0]=0.0;
	sB[1]=1.0/(9.0/2*xiB+0.5);
	sB[2]=weB;
	sB[3]=0.0;
    sB[3]=1.0/(dB+0.5);
	sB[4]=wqB;
	sB[5]=0.0;
    sB[5]=1.0/(dB+0.5);
	sB[6]=wqB;
	sB[7]=0.0;
    sB[7]=1.0/(dB+0.5);
	sB[8]=wqB;
	sB[9]=1.0/(3.0*niuB+0.5);
	sB[10]=wpiB;
	sB[11]=1.0/(3.0*niuB+0.5);
	sB[12]=wpiB;
	sB[13]=1.0/(3.0*niuB+0.5);
	sB[14]=1.0/(3.0*niuB+0.5);
	sB[15]=1.0/(3.0*niuB+0.5);
	sB[16]=wmB;
	sB[17]=wmB;
	sB[18]=wmB;

//double ***rho, ***rho0, ***ux, ***uy, ***uz, ***ux0, ***uy0, ***uz0;
    rho=new double**[NX];
    rho0=new double**[NX];
    ux=new double**[NX];
    uy=new double**[NX];
    uz=new double**[NX];
    ux0=new double**[NX];
    uy0=new double**[NX];
    uz0=new double**[NX];
    p=new double**[NX];

    for (i=0;i<NX;i++)
    {
        rho[i]=new double*[NY];
        rho0[i]=new double*[NY];
        ux[i]=new double*[NY];
        uy[i]=new double*[NY];
        uz[i]=new double*[NY];
        ux0[i]=new double*[NY];
        uy0[i]=new double*[NY];
        uz0[i]=new double*[NY];
        p[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            rho[i][j]=new double[NZ];
            rho0[i][j]=new double[NZ];
            ux[i][j]=new double[NZ];
            uy[i][j]=new double[NZ];
            uz[i][j]=new double[NZ];
            ux0[i][j]=new double[NZ];
            uy0[i][j]=new double[NZ];
            uz0[i][j]=new double[NZ];
            p[i][j]=new double[NZ];
        }
/*
double ***rhoA, ***rhoA0;
double ***phiA, ***psiA, ***TscA, ***pA;
double ****fA, mA[Q], ****fApost, MA[Q], sA[Q];
double ***fscABx, ***fscABy, ***fscABz, ***fscAAx, ***fscAAy, ***fscAAz, ***fscAx, ***fscAy, ***fscAz;
double ****sscA;
double GAB, ***GAA, rA;
*/
    rhoA=new double**[NX];
    rhoA0=new double**[NX];
    phiA=new double**[NX];
    psiA=new double**[NX];
    TscA=new double**[NX];
    pA=new double**[NX];
    fA=new double***[NX];
    fApost=new double***[NX];
    fscABx=new double**[NX];
    fscABy=new double**[NX];
    fscABz=new double**[NX];
    fscAAx=new double**[NX];
    fscAAy=new double**[NX];
    fscAAz=new double**[NX];
    fscAx=new double**[NX];
    fscAy=new double**[NX];
    fscAz=new double**[NX];
    sscA=new double***[NX];
    GAA=new double**[NX];

    for (i=0;i<NX;i++)
    {
        rhoA[i]=new double*[NY];
        rhoA0[i]=new double*[NY];
        phiA[i]=new double*[NY];
        psiA[i]=new double*[NY];
        TscA[i]=new double*[NY];
        pA[i]=new double*[NY];
        fA[i]=new double**[NY];
        fApost[i]=new double**[NY];
        fscABx[i]=new double*[NY];
        fscABy[i]=new double*[NY];
        fscABz[i]=new double*[NY];
        fscAAx[i]=new double*[NY];
        fscAAy[i]=new double*[NY];
        fscAAz[i]=new double*[NY];
        fscAx[i]=new double*[NY];
        fscAy[i]=new double*[NY];
        fscAz[i]=new double*[NY];
        sscA[i]=new double**[NY];
        GAA[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            rhoA[i][j]=new double[NZ];
            rhoA0[i][j]=new double[NZ];
            phiA[i][j]=new double[NZ];
            psiA[i][j]=new double[NZ];
            TscA[i][j]=new double[NZ];
            pA[i][j]=new double[NZ];
            fA[i][j]=new double*[NZ];
            fApost[i][j]=new double*[NZ];
            fscABx[i][j]=new double[NZ];
            fscABy[i][j]=new double[NZ];
            fscABz[i][j]=new double[NZ];
            fscAAx[i][j]=new double[NZ];
            fscAAy[i][j]=new double[NZ];
            fscAAz[i][j]=new double[NZ];
            fscAx[i][j]=new double[NZ];
            fscAy[i][j]=new double[NZ];
            fscAz[i][j]=new double[NZ];
            sscA[i][j]=new double*[NZ];
            GAA[i][j]=new double[NZ];
        }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
            for (k=0;k<NZ;k++)
            {
                fA[i][j][k]=new double[Q];
                fApost[i][j][k]=new double[Q];
                sscA[i][j][k]=new double[Q];
            }
/*
double ***rhoB, ***rhoB0;
double ***phiB, ***psiB, ***TscB, ***pB;
double ****fB, mB[Q], ****fBpost, MB[Q], sB[Q];
double ***fscBAx, ***fscBAy, ***fscBAz, ***fscBBx, ***fscBBy, ***fscBBz, ***fscBx, ***fscBy, ***fscBz;
double ****sscB;
double ***GBB, rB;
*/
    rhoB=new double**[NX];
    rhoB0=new double**[NX];
    phiB=new double**[NX];
    psiB=new double**[NX];
    TscB=new double**[NX];
    pB=new double**[NX];
    fB=new double***[NX];
    fBpost=new double***[NX];
    fscBAx=new double**[NX];
    fscBAy=new double**[NX];
    fscBAz=new double**[NX];
    fscBBx=new double**[NX];
    fscBBy=new double**[NX];
    fscBBz=new double**[NX];
    fscBx=new double**[NX];
    fscBy=new double**[NX];
    fscBz=new double**[NX];
    sscB=new double***[NX];
    GBB=new double**[NX];

    for (i=0;i<NX;i++)
    {
        rhoB[i]=new double*[NY];
        rhoB0[i]=new double*[NY];
        phiB[i]=new double*[NY];
        psiB[i]=new double*[NY];
        TscB[i]=new double*[NY];
        pB[i]=new double*[NY];
        fB[i]=new double**[NY];
        fBpost[i]=new double**[NY];
        fscBAx[i]=new double*[NY];
        fscBAy[i]=new double*[NY];
        fscBAz[i]=new double*[NY];
        fscBBx[i]=new double*[NY];
        fscBBy[i]=new double*[NY];
        fscBBz[i]=new double*[NY];
        fscBx[i]=new double*[NY];
        fscBy[i]=new double*[NY];
        fscBz[i]=new double*[NY];
        sscB[i]=new double**[NY];
        GBB[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            rhoB[i][j]=new double[NZ];
            rhoB0[i][j]=new double[NZ];
            phiB[i][j]=new double[NZ];
            psiB[i][j]=new double[NZ];
            TscB[i][j]=new double[NZ];
            pB[i][j]=new double[NZ];
            fB[i][j]=new double*[NZ];
            fBpost[i][j]=new double*[NZ];
            fscBAx[i][j]=new double[NZ];
            fscBAy[i][j]=new double[NZ];
            fscBAz[i][j]=new double[NZ];
            fscBBx[i][j]=new double[NZ];
            fscBBy[i][j]=new double[NZ];
            fscBBz[i][j]=new double[NZ];
            fscBx[i][j]=new double[NZ];
            fscBy[i][j]=new double[NZ];
            fscBz[i][j]=new double[NZ];
            sscB[i][j]=new double*[NZ];
            GBB[i][j]=new double[NZ];
        }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
            for (k=0;k<NZ;k++)
            {
                fB[i][j][k]=new double[Q];
                fBpost[i][j][k]=new double[Q];
                sscB[i][j][k]=new double[Q];
            }
/*
double ***solid, ***bounce;
int ***data, ***LG;
double error;
double ***vof, ***dvofx, ***dvofy, ***dvofz, ***ddvof;
*/
    solid=new double**[NX];
    bounce=new double**[NX];
    data=new int**[NX];
    LG=new int**[NX];
    vof=new double**[NX];
    dvofx=new double**[NX];
    dvofy=new double**[NX];
    dvofz=new double**[NX];
    ddvof=new double**[NX];

    for (i=0;i<NX;i++)
    {
        solid[i]=new double*[NY];
        bounce[i]=new double*[NY];
        data[i]=new int*[NY];
        LG[i]=new int*[NY];
        vof[i]=new double*[NY];
        dvofx[i]=new double*[NY];
        dvofy[i]=new double*[NY];
        dvofz[i]=new double*[NY];
        ddvof[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            solid[i][j]=new double[NZ];
            bounce[i][j]=new double[NZ];
            data[i][j]=new int[NZ];
            LG[i][j]=new int[NZ];
            vof[i][j]=new double[NZ];
            dvofx[i][j]=new double[NZ];
            dvofy[i][j]=new double[NZ];
            dvofz[i][j]=new double[NZ];
            ddvof[i][j]=new double[NZ];
        }
/*
double **sendfA_right, **sendfA_left, **sendfA_back, **sendfA_front, **sendfA_up, **sendfA_down;
double **recvfA_right, **recvfA_left, **recvfA_back, **recvfA_front, **recvfA_up, **recvfA_down;
double **sendmacro_right, **sendmacro_left, **sendmacro_front, **sendmacro_back, **sendmacro_up, **sendmacro_down;
double **recvmacro_right, **recvmacro_left, **recvmacro_front, **recvmacro_back, **recvmacro_up, **recvmacro_down;
*/
    send_right=new double[(NY-2)*(NZ-2)];
    send_left=new double[(NY-2)*(NZ-2)];
    send_front=new double[NX*(NZ-2)];
    send_back=new double[NX*(NZ-2)];
    send_up=new double[NX*NY];
    send_down=new double[NX*NY];
    recv_right=new double[(NY-2)*(NZ-2)];
    recv_left=new double[(NY-2)*(NZ-2)];
    recv_front=new double[NX*(NZ-2)];
    recv_back=new double[NX*(NZ-2)];
    recv_up=new double[NX*NY];
    recv_down=new double[NX*NY];

    sendfA_right=new double[(NY-2)*(NZ-2)*Q];
    sendfA_left=new double[(NY-2)*(NZ-2)*Q];
    sendfA_front=new double[NX*(NZ-2)*Q];
    sendfA_back=new double[NX*(NZ-2)*Q];
    sendfA_up=new double[NX*NY*Q];
    sendfA_down=new double[NX*NY*Q];
    recvfA_right=new double[(NY-2)*(NZ-2)*Q];
    recvfA_left=new double[(NY-2)*(NZ-2)*Q];
    recvfA_front=new double[NX*(NZ-2)*Q];
    recvfA_back=new double[NX*(NZ-2)*Q];
    recvfA_up=new double[NX*NY*Q];
    recvfA_down=new double[NX*NY*Q];

    /*for (i=0;i<(NY-2)*(NZ-2);i++)
    {
        sendfA_right[i]=new double[Q];
        sendfA_left[i]=new double[Q];
        recvfA_right[i]=new double[Q];
        recvfA_left[i]=new double[Q];
    }

    for (i=0;i<NX*(NZ-2);i++)
    {
        sendfA_front[i]=new double[Q];
        sendfA_back[i]=new double[Q];
        recvfA_front[i]=new double[Q];
        recvfA_back[i]=new double[Q];
    }

    for (i=0;i<NX*NY;i++)
    {
        sendfA_up[i]=new double[Q];
        sendfA_down[i]=new double[Q];
        recvfA_up[i]=new double[Q];
        recvfA_down[i]=new double[Q];
    }*/

    sendfB_right=new double[(NY-2)*(NZ-2)*Q];
    sendfB_left=new double[(NY-2)*(NZ-2)*Q];
    sendfB_front=new double[NX*(NZ-2)*Q];
    sendfB_back=new double[NX*(NZ-2)*Q];
    sendfB_up=new double[NX*NY*Q];
    sendfB_down=new double[NX*NY*Q];
    recvfB_right=new double[(NY-2)*(NZ-2)*Q];
    recvfB_left=new double[(NY-2)*(NZ-2)*Q];
    recvfB_front=new double[NX*(NZ-2)*Q];
    recvfB_back=new double[NX*(NZ-2)*Q];
    recvfB_up=new double[NX*NY*Q];
    recvfB_down=new double[NX*NY*Q];

    /*for (i=0;i<(NY-2)*(NZ-2);i++)
    {
        sendfB_right[i]=new double[Q];
        sendfB_left[i]=new double[Q];
        recvfB_right[i]=new double[Q];
        recvfB_left[i]=new double[Q];
    }

    for (i=0;i<NX*(NZ-2);i++)
    {
        sendfB_front[i]=new double[Q];
        sendfB_back[i]=new double[Q];
        recvfB_front[i]=new double[Q];
        recvfB_back[i]=new double[Q];
    }

    for (i=0;i<NX*NY;i++)
    {
        sendfB_up[i]=new double[Q];
        sendfB_down[i]=new double[Q];
        recvfB_up[i]=new double[Q];
        recvfB_down[i]=new double[Q];
    }*/

    sendmacro_right=new double[(NY-2)*(NZ-2)*Nmacro];
    sendmacro_left=new double[(NY-2)*(NZ-2)*Nmacro];
    sendmacro_front=new double[NX*(NZ-2)*Nmacro];
    sendmacro_back=new double[NX*(NZ-2)*Nmacro];
    sendmacro_up=new double[NX*NY*Nmacro];
    sendmacro_down=new double[NX*NY*Nmacro];
    recvmacro_right=new double[(NY-2)*(NZ-2)*Nmacro];
    recvmacro_left=new double[(NY-2)*(NZ-2)*Nmacro];
    recvmacro_front=new double[NX*(NZ-2)*Nmacro];
    recvmacro_back=new double[NX*(NZ-2)*Nmacro];
    recvmacro_up=new double[NX*NY*Nmacro];
    recvmacro_down=new double[NX*NY*Nmacro];

    /*for (i=0;i<(NY-2)*(NZ-2);i++)
    {
        sendmacro_right[i]=new double[Nmacro];
        sendmacro_left[i]=new double[Nmacro];
        recvmacro_right[i]=new double[Nmacro];
        recvmacro_left[i]=new double[Nmacro];
    }

    for (i=0;i<NX*(NZ-2);i++)
    {
        sendmacro_front[i]=new double[Nmacro];
        sendmacro_back[i]=new double[Nmacro];
        recvmacro_front[i]=new double[Nmacro];
        recvmacro_back[i]=new double[Nmacro];
    }

    for (i=0;i<NX*NY;i++)
    {
        sendmacro_up[i]=new double[Nmacro];
        sendmacro_down[i]=new double[Nmacro];
        recvmacro_up[i]=new double[Nmacro];
        recvmacro_down[i]=new double[Nmacro];
    }*/
}

void Phipsi()
{
    phiA[i][j][k]=1.0-exp(-rhoA[i][j][k]/rA);
    phiB[i][j][k]=1.0-exp(-rhoB[i][j][k]/rB);

    double a,b,R,TcA,TcB,rhocA,rhocB;
	R=0.2;
	a=0.09926;
	b=0.18727;
	TcA=1.0;
	TcB=0.0070601248;
	rhocA=0.13044;
	rhocB=0.13044;
    pA[i][j][k]=rhoA[i][j][k]*R*TscA[i][j][k]*TcA*(1.0+b*rhoA[i][j][k]/4.0+(b*rhoA[i][j][k]/4.0)*(b*rhoA[i][j][k]/4.0)-(b*rhoA[i][j][k]/4.0)*(b*rhoA[i][j][k]/4.0)*(b*rhoA[i][j][k]/4.0))/((1.0-b*rhoA[i][j][k]/4.0)*(1.0-b*rhoA[i][j][k]/4.0)*(1.0-b*rhoA[i][j][k]/4.0))-a*rhoA[i][j][k]*rhoA[i][j][k];
    if(pA[i][j][k]<rhoA[i][j][k]/3.0)
    {
        GAA[i][j][k]=-1.0;
    }
    else
    {
        GAA[i][j][k]=1.0;
    }
    psiA[i][j][k]=sqrt(2.0/GAA[i][j][k]*(pA[i][j][k]-rhoA[i][j][k]/3.0));
    pB[i][j][k]=rhoB[i][j][k]/3.0;
    p[i][j][k]=pA[i][j][k]+pB[i][j][k]+GAB*phiA[i][j][k]*phiB[i][j][k];
}

void Initial()
{
    /*for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                data[i][j][k]=0;
            }
    if(rankz==0)
    {
        for(i=0;i<NX;i++)
            for(j=0;j<NY;j++)
            {
                data[i][j][0]=3;
            }
    }
    if(rankz==PZ-1)
    {
        for(i=0;i<NX;i++)
            for(j=0;j<NY;j++)
            {
                data[i][j][NZ-1]=3;
            }
    }
    if(ranky==0)
    {
        for(i=0;i<NX;i++)
            for(k=0;k<NZ;k++)
            {
                data[i][0][k]=3;
            }
    }
    if(ranky==PY-1)
    {
        for(i=0;i<NX;i++)
            for(k=0;k<NZ;k++)
            {
                data[i][NY-1][k]=3;
            }
    }
    if(rankx==0)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                data[0][j][k]=data[1][j][k];
            }
    }
    if(rankx==PX-1)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                data[NX-1][j][k]=data[NX-2][j][k];
            }
    }*/


    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                rho[i][j][k]=1.0;
                rho0[i][j][k]=1.0;
                ux[i][j][k]=0.0;
                uy[i][j][k]=0.0;
                uz[i][j][k]=0.0;
                ux0[i][j][k]=0.0;
                uy0[i][j][k]=0.0;
                uz0[i][j][k]=0.0;
                bounce[i][j][k]=0.0;

                fscAx[i][j][k]=0.0;
                fscAy[i][j][k]=0.0;
                fscAz[i][j][k]=0.0;
                fscBx[i][j][k]=0.0;
                fscBy[i][j][k]=0.0;
                fscBz[i][j][k]=0.0;

                TscA[i][j][k]=0.85;
                TscB[i][j][k]=0.7;

                if(data[i][j][k]==0)//gas
                {
                    rhoA[i][j][k]=rhogA;
                    rhoB[i][j][k]=rholB;
                    solid[i][j][k]=0.0;
                }
                else if(data[i][j][k]==1)//liquid
                {
                    rhoA[i][j][k]=rholA;
                    rhoB[i][j][k]=rhogB;
                    solid[i][j][k]=0.0;
                }
                else if(data[i][j][k]==2)//dissolved solid
                {
                    rhoA[i][j][k]=rholA;
                    rhoB[i][j][k]=rhogB;
                    solid[i][j][k]=1.0;
                }
                else if(data[i][j][k]==3)//rock
                {
                    rhoA[i][j][k]=rholA;
                    rhoB[i][j][k]=rhogB;
                    solid[i][j][k]=2.0;
                }
                rho[i][j][k]=rhoA[i][j][k]+rhoB[i][j][k];
                Phipsi();
                for(l=0;l<Q;l++)
                {
                    fA[i][j][k][l]=feq(l,rhoA[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]);
                    fB[i][j][k][l]=feq(l,rhoB[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]);
                }
            }
    Getbounce();
    Vof();
}

void Initial2()
{
    int NXS=5;
    double averhoA, averhoB;
    int aven;
    double dp=0.0;
    MPI_Status status;
    double temp, temp2, temp3;
    
    if(rankx==0)
    {
        aven=0;
        averhoA=0.0;
        averhoB=0.0;
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                averhoA=averhoA+rhoA[NXS][j][k];
                averhoB=averhoB+rhoB[NXS][j][k];
                aven=aven+1;
            }
        rhoAin=averhoA/aven;
        rhoBin=averhoB/aven+dp;  
    }
    else
    {
        rhoAin=0.0;
        rhoBin=0.0;
    }

    temp=rhoAin;
    MPI_Reduce(&temp,&temp2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(mpirank==0)
    {
        for(i=1;i<mpisize;i++)
        {
            MPI_Send(&temp2,1,MPI_DOUBLE,i,8883,MPI_COMM_WORLD);
        }
        temp3=temp2;
    }
    else
    {
        MPI_Recv(&temp3,1,MPI_DOUBLE,0,8883,MPI_COMM_WORLD,&status);
    }
    rhoAin=temp3;

    temp=rhoBin;
    MPI_Reduce(&temp,&temp2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(mpirank==0)
    {
        for(i=1;i<mpisize;i++)
        {
            MPI_Send(&temp2,1,MPI_DOUBLE,i,8884,MPI_COMM_WORLD);
        }
        temp3=temp2;
    }
    else
    {
        MPI_Recv(&temp3,1,MPI_DOUBLE,0,8884,MPI_COMM_WORLD,&status);
    }
    rhoBin=temp3;
    rhoBin=rhoBin+rhoAin-0.1;
    rhoAin=0.1;

    /*if(rankx==PX-1)
    {
        aven=0;
        averhoA=0.0;
        averhoB=0.0;
        for(j=2;j<NY-2;j++)
            for(k=2;k<NZ-2;k++)
            {
                averhoA=averhoA+rhoA[NX-NXS-1][j][k];
                averhoB=averhoB+rhoB[NX-NXS-1][j][k];
                aven=aven+1;
            }
        rhoAout=averhoA/aven;
        rhoBout=averhoB/aven;
    }
    else
    {
        rhoAout=0.0;
        rhoBout=0.0;
    }

    temp=rhoAout;
    MPI_Reduce(&temp,&temp2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(mpirank==0)
    {
        for(i=1;i<mpisize;i++)
        {
            MPI_Send(&temp2,1,MPI_DOUBLE,i,8885,MPI_COMM_WORLD);
        }
        temp3=temp2;
    }
    else
    {
        MPI_Recv(&temp3,1,MPI_DOUBLE,0,8885,MPI_COMM_WORLD,&status);
    }
    rhoAout=temp3;

    temp=rhoBout;
    MPI_Reduce(&temp,&temp2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(mpirank==0)
    {
        for(i=1;i<mpisize;i++)
        {
            MPI_Send(&temp2,1,MPI_DOUBLE,i,8886,MPI_COMM_WORLD);
        }
        temp3=temp2;
    }
    else
    {
        MPI_Recv(&temp3,1,MPI_DOUBLE,0,8886,MPI_COMM_WORLD,&status);
    }
    rhoBout=temp3;*/


    if(rankx==0)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                rhoA[0][j][k]=rhoAin;
                rhoB[0][j][k]=rhoBin;
                rhoA[1][j][k]=rhoAin;
                rhoB[1][j][k]=rhoBin;
                ux[1][j][k]=0.0;
                uy[1][j][k]=0.0;
                uz[1][j][k]=0.0;
                for(l=0;l<Q;l++)
                {
                    fA[0][j][k][l]=feq(l,rhoA[0][j][k],ux[0][j][k],uy[0][j][k],uz[0][j][k]);
                    fA[1][j][k][l]=feq(l,rhoA[1][j][k],ux[1][j][k],uy[1][j][k],uz[1][j][k]);
                    fB[0][j][k][l]=feq(l,rhoB[0][j][k],ux[0][j][k],uy[0][j][k],uz[0][j][k]);
                    fB[1][j][k][l]=feq(l,rhoB[1][j][k],ux[1][j][k],uy[1][j][k],uz[1][j][k]);
                }
            }

        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                solid[0][j][k]=solid[2][j][k];
                solid[1][j][k]=solid[2][j][k];
            }
    }
    /*if(rankx==PX-1)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                rhoA[NX-1][j][k]=rhoAout;
                rhoB[NX-1][j][k]=rhoBout;
                rhoA[NX-2][j][k]=rhoAout;
                rhoB[NX-2][j][k]=rhoBout;
                ux[NX-2][j][k]=0.0;
                uy[NX-2][j][k]=0.0;
                uz[NX-2][j][k]=0.0;
                for(l=0;l<Q;l++)
                {
                    fA[NX-1][j][k][l]=feq(l,rhoA[NX-1][j][k],ux[NX-1][j][k],uy[NX-1][j][k],uz[NX-1][j][k]);
                    fA[NX-2][j][k][l]=feq(l,rhoA[NX-2][j][k],ux[NX-2][j][k],uy[NX-2][j][k],uz[NX-2][j][k]);
                    fB[NX-1][j][k][l]=feq(l,rhoB[NX-1][j][k],ux[NX-1][j][k],uy[NX-1][j][k],uz[NX-1][j][k]);
                    fB[NX-2][j][k][l]=feq(l,rhoB[NX-2][j][k],ux[NX-2][j][k],uy[NX-2][j][k],uz[NX-2][j][k]);
                }
            }

        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                solid[NX-1][j][k]=solid[NX-3][j][k];
                solid[NX-2][j][k]=solid[NX-1][j][k];
            }
    }*/
    Phisolid2();
    Getbounce();
    Vof();
}

void Fsc()
{
    double epsilon=0.0;
    double beta=0.20;
    int ip,jp,kp;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]<1.0)
                {
                fscABx[i][j][k]=phiB[i+1][j][k]/6.0+phiB[i+1][j+1][k]/12.0+phiB[i+1][j-1][k]/12.0+phiB[i+1][j][k+1]/12.0+phiB[i+1][j][k-1]/12.0-phiB[i-1][j][k]/6.0-phiB[i-1][j+1][k]/12.0-phiB[i-1][j-1][k]/12.0-phiB[i-1][j][k+1]/12.0-phiB[i-1][j][k-1]/12.0;
                fscABy[i][j][k]=phiB[i][j+1][k]/6.0+phiB[i+1][j+1][k]/12.0+phiB[i-1][j+1][k]/12.0+phiB[i][j+1][k+1]/12.0+phiB[i][j+1][k-1]/12.0-phiB[i][j-1][k]/6.0-phiB[i+1][j-1][k]/12.0-phiB[i-1][j-1][k]/12.0-phiB[i][j-1][k+1]/12.0-phiB[i][j-1][k-1]/12.0;
	            fscABz[i][j][k]=phiB[i][j][k+1]/6.0+phiB[i+1][j][k+1]/12.0+phiB[i-1][j][k+1]/12.0+phiB[i][j+1][k+1]/12.0+phiB[i][j-1][k+1]/12.0-phiB[i][j][k-1]/6.0-phiB[i+1][j][k-1]/12.0-phiB[i-1][j][k-1]/12.0-phiB[i][j+1][k-1]/12.0-phiB[i][j-1][k-1]/12.0;
                //fscABx[i][j][k]=phiB[i+1][j][k]*2.0/9.0+phiB[i+1][j+1][k]/18.0+phiB[i+1][j-1][k]/18.0+phiB[i+1][j][k+1]/18.0+phiB[i+1][j][k-1]/18.0+phiB[i+1][j+1][k+1]/72.0+phiB[i+1][j-1][k+1]/72.0+phiB[i+1][j+1][k-1]/72.0+phiB[i+1][j-1][k-1]/72.0-phiB[i-1][j][k]*2.0/9.0-phiB[i-1][j+1][k]/18.0-phiB[i-1][j-1][k]/18.0-phiB[i-1][j][k+1]/18.0-phiB[i-1][j][k-1]/18.0-phiB[i-1][j+1][k+1]/72.0-phiB[i-1][j-1][k+1]/72.0-phiB[i-1][j+1][k-1]/72.0-phiB[i-1][j-1][k-1]/72.0;
                //fscABy[i][j][k]=phiB[i][j+1][k]*2.0/9.0+phiB[i+1][j+1][k]/18.0+phiB[i-1][j+1][k]/18.0+phiB[i][j+1][k+1]/18.0+phiB[i][j+1][k-1]/18.0+phiB[i+1][j+1][k+1]/72.0+phiB[i-1][j+1][k+1]/72.0+phiB[i+1][j+1][k-1]/72.0+phiB[i-1][j+1][k-1]/72.0-phiB[i][j-1][k]*2.0/9.0-phiB[i+1][j-1][k]/18.0-phiB[i-1][j-1][k]/18.0-phiB[i][j-1][k+1]/18.0-phiB[i][j-1][k-1]/18.0-phiB[i+1][j-1][k+1]/72.0-phiB[i-1][j-1][k+1]/72.0-phiB[i+1][j-1][k-1]/72.0-phiB[i-1][j-1][k-1]/72.0;
	            //fscABz[i][j][k]=phiB[i][j][k+1]*2.0/9.0+phiB[i+1][j][k+1]/18.0+phiB[i-1][j][k+1]/18.0+phiB[i][j+1][k+1]/18.0+phiB[i][j-1][k+1]/18.0+phiB[i+1][j+1][k+1]/72.0+phiB[i-1][j+1][k+1]/72.0+phiB[i+1][j-1][k+1]/72.0+phiB[i-1][j-1][k+1]/72.0-phiB[i][j][k-1]*2.0/9.0-phiB[i+1][j][k-1]/18.0-phiB[i-1][j][k-1]/18.0-phiB[i][j+1][k-1]/18.0-phiB[i][j-1][k-1]/18.0-phiB[i+1][j+1][k-1]/72.0-phiB[i-1][j+1][k-1]/72.0-phiB[i+1][j-1][k-1]/72.0-phiB[i-1][j-1][k-1]/72.0;
                fscABx[i][j][k]=-(1.0-beta)*GAB*phiA[i][j][k]*fscABx[i][j][k];
                fscABy[i][j][k]=-(1.0-beta)*GAB*phiA[i][j][k]*fscABy[i][j][k];
                fscABz[i][j][k]=-(1.0-beta)*GAB*phiA[i][j][k]*fscABz[i][j][k];

                fscAAx[i][j][k]=psiA[i+1][j][k]/6.0+psiA[i+1][j+1][k]/12.0+psiA[i+1][j-1][k]/12.0+psiA[i+1][j][k+1]/12.0+psiA[i+1][j][k-1]/12.0-psiA[i-1][j][k]/6.0-psiA[i-1][j+1][k]/12.0-psiA[i-1][j-1][k]/12.0-psiA[i-1][j][k+1]/12.0-psiA[i-1][j][k-1]/12.0;
                fscAAy[i][j][k]=psiA[i][j+1][k]/6.0+psiA[i+1][j+1][k]/12.0+psiA[i-1][j+1][k]/12.0+psiA[i][j+1][k+1]/12.0+psiA[i][j+1][k-1]/12.0-psiA[i][j-1][k]/6.0-psiA[i+1][j-1][k]/12.0-psiA[i-1][j-1][k]/12.0-psiA[i][j-1][k+1]/12.0-psiA[i][j-1][k-1]/12.0;
	            fscAAz[i][j][k]=psiA[i][j][k+1]/6.0+psiA[i+1][j][k+1]/12.0+psiA[i-1][j][k+1]/12.0+psiA[i][j+1][k+1]/12.0+psiA[i][j-1][k+1]/12.0-psiA[i][j][k-1]/6.0-psiA[i+1][j][k-1]/12.0-psiA[i-1][j][k-1]/12.0-psiA[i][j+1][k-1]/12.0-psiA[i][j-1][k-1]/12.0;
                //fscAAx[i][j][k]=psiA[i+1][j][k]*2.0/9.0+psiA[i+1][j+1][k]/18.0+psiA[i+1][j-1][k]/18.0+psiA[i+1][j][k+1]/18.0+psiA[i+1][j][k-1]/18.0+psiA[i+1][j+1][k+1]/72.0+psiA[i+1][j-1][k+1]/72.0+psiA[i+1][j+1][k-1]/72.0+psiA[i+1][j-1][k-1]/72.0-psiA[i-1][j][k]*2.0/9.0-psiA[i-1][j+1][k]/18.0-psiA[i-1][j-1][k]/18.0-psiA[i-1][j][k+1]/18.0-psiA[i-1][j][k-1]/18.0-psiA[i-1][j+1][k+1]/72.0-psiA[i-1][j-1][k+1]/72.0-psiA[i-1][j+1][k-1]/72.0-psiA[i-1][j-1][k-1]/72.0;
                //fscAAy[i][j][k]=psiA[i][j+1][k]*2.0/9.0+psiA[i+1][j+1][k]/18.0+psiA[i-1][j+1][k]/18.0+psiA[i][j+1][k+1]/18.0+psiA[i][j+1][k-1]/18.0+psiA[i+1][j+1][k+1]/72.0+psiA[i-1][j+1][k+1]/72.0+psiA[i+1][j+1][k-1]/72.0+psiA[i-1][j+1][k-1]/72.0-psiA[i][j-1][k]*2.0/9.0-psiA[i+1][j-1][k]/18.0-psiA[i-1][j-1][k]/18.0-psiA[i][j-1][k+1]/18.0-psiA[i][j-1][k-1]/18.0-psiA[i+1][j-1][k+1]/72.0-psiA[i-1][j-1][k+1]/72.0-psiA[i+1][j-1][k-1]/72.0-psiA[i-1][j-1][k-1]/72.0;
	            //fscAAz[i][j][k]=psiA[i][j][k+1]*2.0/9.0+psiA[i+1][j][k+1]/18.0+psiA[i-1][j][k+1]/18.0+psiA[i][j+1][k+1]/18.0+psiA[i][j-1][k+1]/18.0+psiA[i+1][j+1][k+1]/72.0+psiA[i-1][j+1][k+1]/72.0+psiA[i+1][j-1][k+1]/72.0+psiA[i-1][j-1][k+1]/72.0-psiA[i][j][k-1]*2.0/9.0-psiA[i+1][j][k-1]/18.0-psiA[i-1][j][k-1]/18.0-psiA[i][j+1][k-1]/18.0-psiA[i][j-1][k-1]/18.0-psiA[i+1][j+1][k-1]/72.0-psiA[i-1][j+1][k-1]/72.0-psiA[i+1][j-1][k-1]/72.0-psiA[i-1][j-1][k-1]/72.0;
                
                fscAAx[i][j][k]=-(1.0-beta)*GAA[i][j][k]*psiA[i][j][k]*fscAAx[i][j][k];
                fscAAy[i][j][k]=-(1.0-beta)*GAA[i][j][k]*psiA[i][j][k]*fscAAy[i][j][k];
                fscAAz[i][j][k]=-(1.0-beta)*GAA[i][j][k]*psiA[i][j][k]*fscAAz[i][j][k];

                for(l=0;l<Q;l++)
                {
                    ip=i+ex[l];
                    jp=j+ey[l];
                    kp=k+ez[l];
                    fscABx[i][j][k]=fscABx[i][j][k]-beta/2.0*GAB*phiA[ip][jp][kp]*phiB[ip][jp][kp]*w[l]*3.0*ex[l];
                    fscABy[i][j][k]=fscABy[i][j][k]-beta/2.0*GAB*phiA[ip][jp][kp]*phiB[ip][jp][kp]*w[l]*3.0*ey[l];
                    fscABz[i][j][k]=fscABz[i][j][k]-beta/2.0*GAB*phiA[ip][jp][kp]*phiB[ip][jp][kp]*w[l]*3.0*ez[l];
                    fscAAx[i][j][k]=fscAAx[i][j][k]-beta/2.0*GAA[i][j][k]*psiA[ip][jp][kp]*psiA[ip][jp][kp]*w[l]*3.0*ex[l];
                    fscAAy[i][j][k]=fscAAy[i][j][k]-beta/2.0*GAA[i][j][k]*psiA[ip][jp][kp]*psiA[ip][jp][kp]*w[l]*3.0*ey[l];
                    fscAAz[i][j][k]=fscAAz[i][j][k]-beta/2.0*GAA[i][j][k]*psiA[ip][jp][kp]*psiA[ip][jp][kp]*w[l]*3.0*ez[l];
                }

                fscAx[i][j][k]=fscABx[i][j][k]+fscAAx[i][j][k]+gvx;
                fscAy[i][j][k]=fscABy[i][j][k]+fscAAy[i][j][k]+gvy;
                fscAz[i][j][k]=fscABz[i][j][k]+fscAAz[i][j][k]+gvz;

                sscA[i][j][k][0]=0.0;
	            sscA[i][j][k][1]=38.0*(ux[i][j][k]*fscAx[i][j][k]+uy[i][j][k]*fscAy[i][j][k]+uz[i][j][k]*fscAz[i][j][k]);
	            sscA[i][j][k][2]=-11.0*(ux[i][j][k]*fscAx[i][j][k]+uy[i][j][k]*fscAy[i][j][k]+uz[i][j][k]*fscAz[i][j][k]);
	            sscA[i][j][k][3]=fscAx[i][j][k];
	            sscA[i][j][k][4]=-2.0/3*fscAx[i][j][k];
	            sscA[i][j][k][5]=fscAy[i][j][k];
	            sscA[i][j][k][6]=-2.0/3*fscAy[i][j][k];
	            sscA[i][j][k][7]=fscAz[i][j][k];
	            sscA[i][j][k][8]=-2.0/3*fscAz[i][j][k];
	            sscA[i][j][k][9]=2.0*(2.0*ux[i][j][k]*fscAx[i][j][k]-uy[i][j][k]*fscAy[i][j][k]-uz[i][j][k]*fscAz[i][j][k]);
	            sscA[i][j][k][10]=-2.0*ux[i][j][k]*fscAx[i][j][k]+uy[i][j][k]*fscAy[i][j][k]+uz[i][j][k]*fscAz[i][j][k];
	            sscA[i][j][k][11]=2.0*(uy[i][j][k]*fscAy[i][j][k]-uz[i][j][k]*fscAz[i][j][k]);
	            sscA[i][j][k][12]=-uy[i][j][k]*fscAy[i][j][k]+uz[i][j][k]*fscAz[i][j][k];
	            sscA[i][j][k][13]=uy[i][j][k]*fscAx[i][j][k]+ux[i][j][k]*fscAy[i][j][k];
	            sscA[i][j][k][14]=uz[i][j][k]*fscAy[i][j][k]+uy[i][j][k]*fscAz[i][j][k];
	            sscA[i][j][k][15]=uz[i][j][k]*fscAx[i][j][k]+ux[i][j][k]*fscAz[i][j][k];
	            sscA[i][j][k][16]=0.0;
	            sscA[i][j][k][17]=0.0;
	            sscA[i][j][k][18]=0.0;

                fscBAx[i][j][k]=phiA[i+1][j][k]/6.0+phiA[i+1][j+1][k]/12.0+phiA[i+1][j-1][k]/12.0+phiA[i+1][j][k+1]/12.0+phiA[i+1][j][k-1]/12.0-phiA[i-1][j][k]/6.0-phiA[i-1][j+1][k]/12.0-phiA[i-1][j-1][k]/12.0-phiA[i-1][j][k+1]/12.0-phiA[i-1][j][k-1]/12.0;
                fscBAy[i][j][k]=phiA[i][j+1][k]/6.0+phiA[i+1][j+1][k]/12.0+phiA[i-1][j+1][k]/12.0+phiA[i][j+1][k+1]/12.0+phiA[i][j+1][k-1]/12.0-phiA[i][j-1][k]/6.0-phiA[i+1][j-1][k]/12.0-phiA[i-1][j-1][k]/12.0-phiA[i][j-1][k+1]/12.0-phiA[i][j-1][k-1]/12.0;
	            fscBAz[i][j][k]=phiA[i][j][k+1]/6.0+phiA[i+1][j][k+1]/12.0+phiA[i-1][j][k+1]/12.0+phiA[i][j+1][k+1]/12.0+phiA[i][j-1][k+1]/12.0-phiA[i][j][k-1]/6.0-phiA[i+1][j][k-1]/12.0-phiA[i-1][j][k-1]/12.0-phiA[i][j+1][k-1]/12.0-phiA[i][j-1][k-1]/12.0;
                //fscBAx[i][j][k]=phiA[i+1][j][k]*2.0/9.0+phiA[i+1][j+1][k]/18.0+phiA[i+1][j-1][k]/18.0+phiA[i+1][j][k+1]/18.0+phiA[i+1][j][k-1]/18.0+phiA[i+1][j+1][k+1]/72.0+phiA[i+1][j-1][k+1]/72.0+phiA[i+1][j+1][k-1]/72.0+phiA[i+1][j-1][k-1]/72.0-phiA[i-1][j][k]*2.0/9.0-phiA[i-1][j+1][k]/18.0-phiA[i-1][j-1][k]/18.0-phiA[i-1][j][k+1]/18.0-phiA[i-1][j][k-1]/18.0-phiA[i-1][j+1][k+1]/72.0-phiA[i-1][j-1][k+1]/72.0-phiA[i-1][j+1][k-1]/72.0-phiA[i-1][j-1][k-1]/72.0;
                //fscBAy[i][j][k]=phiA[i][j+1][k]*2.0/9.0+phiA[i+1][j+1][k]/18.0+phiA[i-1][j+1][k]/18.0+phiA[i][j+1][k+1]/18.0+phiA[i][j+1][k-1]/18.0+phiA[i+1][j+1][k+1]/72.0+phiA[i-1][j+1][k+1]/72.0+phiA[i+1][j+1][k-1]/72.0+phiA[i-1][j+1][k-1]/72.0-phiA[i][j-1][k]*2.0/9.0-phiA[i+1][j-1][k]/18.0-phiA[i-1][j-1][k]/18.0-phiA[i][j-1][k+1]/18.0-phiA[i][j-1][k-1]/18.0-phiA[i+1][j-1][k+1]/72.0-phiA[i-1][j-1][k+1]/72.0-phiA[i+1][j-1][k-1]/72.0-phiA[i-1][j-1][k-1]/72.0;
	            //fscBAz[i][j][k]=phiA[i][j][k+1]*2.0/9.0+phiA[i+1][j][k+1]/18.0+phiA[i-1][j][k+1]/18.0+phiA[i][j+1][k+1]/18.0+phiA[i][j-1][k+1]/18.0+phiA[i+1][j+1][k+1]/72.0+phiA[i-1][j+1][k+1]/72.0+phiA[i+1][j-1][k+1]/72.0+phiA[i-1][j-1][k+1]/72.0-phiA[i][j][k-1]*2.0/9.0-phiA[i+1][j][k-1]/18.0-phiA[i-1][j][k-1]/18.0-phiA[i][j+1][k-1]/18.0-phiA[i][j-1][k-1]/18.0-phiA[i+1][j+1][k-1]/72.0-phiA[i-1][j+1][k-1]/72.0-phiA[i+1][j-1][k-1]/72.0-phiA[i-1][j-1][k-1]/72.0;
                
                fscBAx[i][j][k]=-GAB*(1.0-beta)*phiB[i][j][k]*fscBAx[i][j][k];
                fscBAy[i][j][k]=-GAB*(1.0-beta)*phiB[i][j][k]*fscBAy[i][j][k];
                fscBAz[i][j][k]=-GAB*(1.0-beta)*phiB[i][j][k]*fscBAz[i][j][k];

 /*               fscBBx[i][j][k]=psiB[i+1][j][k]/6.0+psiB[i+1][j+1][k]/12.0+psiB[i+1][j-1][k]/12.0+psiB[i+1][j][k+1]/12.0+psiB[i+1][j][k-1]/12.0-psiB[i-1][j][k]/6.0-psiB[i-1][j+1][k]/12.0-psiB[i-1][j-1][k]/12.0-psiB[i-1][j][k+1]/12.0-psiB[i-1][j][k-1]/12.0;
                fscBBy[i][j][k]=psiB[i][j+1][k]/6.0+psiB[i+1][j+1][k]/12.0+psiB[i-1][j+1][k]/12.0+psiB[i][j+1][k+1]/12.0+psiB[i][j+1][k-1]/12.0-psiB[i][j-1][k]/6.0-psiB[i+1][j-1][k]/12.0-psiB[i-1][j-1][k]/12.0-psiB[i][j-1][k+1]/12.0-psiB[i][j-1][k-1]/12.0;
	            fscBBz[i][j][k]=psiB[i][j][k+1]/6.0+psiB[i+1][j][k+1]/12.0+psiB[i-1][j][k+1]/12.0+psiB[i][j+1][k+1]/12.0+psiB[i][j-1][k+1]/12.0-psiB[i][j][k-1]/6.0-psiB[i+1][j][k-1]/12.0-psiB[i-1][j][k-1]/12.0-psiB[i][j+1][k-1]/12.0-psiB[i][j-1][k-1]/12.0;
                fscBBx[i][j][k]=-GBB[i][j][k]*psiB[i][j][k]*fscBBx[i][j][k];
                fscBBy[i][j][k]=-GBB[i][j][k]*psiB[i][j][k]*fscBBy[i][j][k];
                fscBBz[i][j][k]=-GBB[i][j][k]*psiB[i][j][k]*fscBBz[i][j][k];*/

                for(l=0;l<Q;l++)
                {
                    ip=i+ex[l];
                    jp=j+ey[l];
                    kp=k+ez[l];
                    fscBAx[i][j][k]=fscBAx[i][j][k]-beta/2.0*GAB*phiA[ip][jp][kp]*phiB[ip][jp][kp]*w[l]*3.0*ex[l];
                    fscBAy[i][j][k]=fscBAy[i][j][k]-beta/2.0*GAB*phiA[ip][jp][kp]*phiB[ip][jp][kp]*w[l]*3.0*ey[l];
                    fscBAz[i][j][k]=fscBAz[i][j][k]-beta/2.0*GAB*phiA[ip][jp][kp]*phiB[ip][jp][kp]*w[l]*3.0*ez[l];
                }

                fscBx[i][j][k]=fscBAx[i][j][k]+gvx;
                fscBy[i][j][k]=fscBAy[i][j][k]+gvy;
                fscBz[i][j][k]=fscBAz[i][j][k]+gvz;
//                fscBx[i][j][k]=fscBAx[i][j][k]+fscBBx[i][j][k];
//                fscBy[i][j][k]=fscBAy[i][j][k]+fscBBy[i][j][k];
//                fscBz[i][j][k]=fscBAz[i][j][k]+fscBBz[i][j][k];

                sscB[i][j][k][0]=0.0;
	            sscB[i][j][k][1]=38.0*(ux[i][j][k]*fscBx[i][j][k]+uy[i][j][k]*fscBy[i][j][k]+uz[i][j][k]*fscBz[i][j][k]);
	            sscB[i][j][k][2]=-11.0*(ux[i][j][k]*fscBx[i][j][k]+uy[i][j][k]*fscBy[i][j][k]+uz[i][j][k]*fscBz[i][j][k]);
	            sscB[i][j][k][3]=fscBx[i][j][k];
	            sscB[i][j][k][4]=-2.0/3*fscBx[i][j][k];
	            sscB[i][j][k][5]=fscBy[i][j][k];
	            sscB[i][j][k][6]=-2.0/3*fscBy[i][j][k];
	            sscB[i][j][k][7]=fscBz[i][j][k];
	            sscB[i][j][k][8]=-2.0/3*fscBz[i][j][k];
	            sscB[i][j][k][9]=2.0*(2.0*ux[i][j][k]*fscBx[i][j][k]-uy[i][j][k]*fscBy[i][j][k]-uz[i][j][k]*fscBz[i][j][k]);
	            sscB[i][j][k][10]=-2.0*ux[i][j][k]*fscBx[i][j][k]+uy[i][j][k]*fscBy[i][j][k]+uz[i][j][k]*fscBz[i][j][k];
	            sscB[i][j][k][11]=2.0*(uy[i][j][k]*fscBy[i][j][k]-uz[i][j][k]*fscBz[i][j][k]);
	            sscB[i][j][k][12]=-uy[i][j][k]*fscBy[i][j][k]+uz[i][j][k]*fscBz[i][j][k];
	            sscB[i][j][k][13]=uy[i][j][k]*fscBx[i][j][k]+ux[i][j][k]*fscBy[i][j][k];
	            sscB[i][j][k][14]=uz[i][j][k]*fscBy[i][j][k]+uy[i][j][k]*fscBz[i][j][k];
	            sscB[i][j][k][15]=uz[i][j][k]*fscBx[i][j][k]+ux[i][j][k]*fscBz[i][j][k];
	            sscB[i][j][k][16]=0.0;
	            sscB[i][j][k][17]=0.0;
	            sscB[i][j][k][18]=0.0;
                }
            }
}

void Collision()
{
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]<1.0)
                {
                    mA[0]=fA[i][j][k][0]+fA[i][j][k][1]+fA[i][j][k][2]+fA[i][j][k][3]+fA[i][j][k][4]+fA[i][j][k][5]+fA[i][j][k][6]+fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]+fA[i][j][k][11]+fA[i][j][k][12]+fA[i][j][k][13]+fA[i][j][k][14]+fA[i][j][k][15]+fA[i][j][k][16]+fA[i][j][k][17]+fA[i][j][k][18];
	                mA[1]=-30.0*fA[i][j][k][0]-11.0*fA[i][j][k][1]-11.0*fA[i][j][k][2]-11.0*fA[i][j][k][3]-11.0*fA[i][j][k][4]-11.0*fA[i][j][k][5]-11.0*fA[i][j][k][6]+8.0*fA[i][j][k][7]+8.0*fA[i][j][k][8]+8.0*fA[i][j][k][9]+8.0*fA[i][j][k][10]+8.0*fA[i][j][k][11]+8.0*fA[i][j][k][12]+8.0*fA[i][j][k][13]+8.0*fA[i][j][k][14]+8.0*fA[i][j][k][15]+8.0*fA[i][j][k][16]+8.0*fA[i][j][k][17]+8.0*fA[i][j][k][18];
	                mA[2]=12.0*fA[i][j][k][0]-4.0*fA[i][j][k][1]-4.0*fA[i][j][k][2]-4.0*fA[i][j][k][3]-4.0*fA[i][j][k][4]-4.0*fA[i][j][k][5]-4.0*fA[i][j][k][6]+fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]+fA[i][j][k][11]+fA[i][j][k][12]+fA[i][j][k][13]+fA[i][j][k][14]+fA[i][j][k][15]+fA[i][j][k][16]+fA[i][j][k][17]+fA[i][j][k][18];
	                mA[3]=fA[i][j][k][1]-fA[i][j][k][2]+fA[i][j][k][7]-fA[i][j][k][8]+fA[i][j][k][9]-fA[i][j][k][10]+fA[i][j][k][11]-fA[i][j][k][12]+fA[i][j][k][13]-fA[i][j][k][14];
	                mA[4]=-4.0*fA[i][j][k][1]+4.0*fA[i][j][k][2]+fA[i][j][k][7]-fA[i][j][k][8]+fA[i][j][k][9]-fA[i][j][k][10]+fA[i][j][k][11]-fA[i][j][k][12]+fA[i][j][k][13]-fA[i][j][k][14];
	                mA[5]=fA[i][j][k][3]-fA[i][j][k][4]+fA[i][j][k][7]+fA[i][j][k][8]-fA[i][j][k][9]-fA[i][j][k][10]+fA[i][j][k][15]-fA[i][j][k][16]+fA[i][j][k][17]-fA[i][j][k][18];
	                mA[6]=-4.0*fA[i][j][k][3]+4.0*fA[i][j][k][4]+fA[i][j][k][7]+fA[i][j][k][8]-fA[i][j][k][9]-fA[i][j][k][10]+fA[i][j][k][15]-fA[i][j][k][16]+fA[i][j][k][17]-fA[i][j][k][18];
	                mA[7]=fA[i][j][k][5]-fA[i][j][k][6]+fA[i][j][k][11]+fA[i][j][k][12]-fA[i][j][k][13]-fA[i][j][k][14]+fA[i][j][k][15]+fA[i][j][k][16]-fA[i][j][k][17]-fA[i][j][k][18];
	                mA[8]=-4.0*fA[i][j][k][5]+4.0*fA[i][j][k][6]+fA[i][j][k][11]+fA[i][j][k][12]-fA[i][j][k][13]-fA[i][j][k][14]+fA[i][j][k][15]+fA[i][j][k][16]-fA[i][j][k][17]-fA[i][j][k][18];
	                mA[9]=2.0*fA[i][j][k][1]+2.0*fA[i][j][k][2]-fA[i][j][k][3]-fA[i][j][k][4]-fA[i][j][k][5]-fA[i][j][k][6]+fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]+fA[i][j][k][11]+fA[i][j][k][12]+fA[i][j][k][13]+fA[i][j][k][14]-2.0*fA[i][j][k][15]-2.0*fA[i][j][k][16]-2.0*fA[i][j][k][17]-2.0*fA[i][j][k][18];
	                mA[10]=-4.0*fA[i][j][k][1]-4.0*fA[i][j][k][2]+2.0*fA[i][j][k][3]+2.0*fA[i][j][k][4]+2.0*fA[i][j][k][5]+2.0*fA[i][j][k][6]+fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]+fA[i][j][k][11]+fA[i][j][k][12]+fA[i][j][k][13]+fA[i][j][k][14]-2.0*fA[i][j][k][15]-2.0*fA[i][j][k][16]-2.0*fA[i][j][k][17]-2.0*fA[i][j][k][18];
	                mA[11]=fA[i][j][k][3]+fA[i][j][k][4]-fA[i][j][k][5]-fA[i][j][k][6]+fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]-fA[i][j][k][11]-fA[i][j][k][12]-fA[i][j][k][13]-fA[i][j][k][14];
	                mA[12]=-2.0*fA[i][j][k][3]-2.0*fA[i][j][k][4]+2.0*fA[i][j][k][5]+2.0*fA[i][j][k][6]+fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]-fA[i][j][k][11]-fA[i][j][k][12]-fA[i][j][k][13]-fA[i][j][k][14];
	                mA[13]=fA[i][j][k][7]-fA[i][j][k][8]-fA[i][j][k][9]+fA[i][j][k][10];
	                mA[14]=fA[i][j][k][15]-fA[i][j][k][16]-fA[i][j][k][17]+fA[i][j][k][18];
	                mA[15]=fA[i][j][k][11]-fA[i][j][k][12]-fA[i][j][k][13]+fA[i][j][k][14];
	                mA[16]=fA[i][j][k][7]-fA[i][j][k][8]+fA[i][j][k][9]-fA[i][j][k][10]-fA[i][j][k][11]+fA[i][j][k][12]-fA[i][j][k][13]+fA[i][j][k][14];
	                mA[17]=-fA[i][j][k][7]-fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]+fA[i][j][k][15]-fA[i][j][k][16]+fA[i][j][k][17]-fA[i][j][k][18];
	                mA[18]=fA[i][j][k][11]+fA[i][j][k][12]-fA[i][j][k][13]-fA[i][j][k][14]-fA[i][j][k][15]-fA[i][j][k][16]+fA[i][j][k][17]+fA[i][j][k][18];
	                for(l=0;l<Q;l++)
	                {
		                MA[l]=mA[l]-sA[l]*(mA[l]-meq(l,rhoA[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]))+(1-sA[l]/2.0)*sscA[i][j][k][l];
	                }
	                fApost[i][j][k][0]=1.0/19*MA[0]-5.0/399*MA[1]+1.0/21*MA[2];
	                fApost[i][j][k][1]=1.0/19*MA[0]-11.0/2394*MA[1]-1.0/63*MA[2]+1.0/10*MA[3]-1.0/10*MA[4]+1.0/18*MA[9]-1.0/18*MA[10];
	                fApost[i][j][k][2]=1.0/19*MA[0]-11.0/2394*MA[1]-1.0/63*MA[2]-1.0/10*MA[3]+1.0/10*MA[4]+1.0/18*MA[9]-1.0/18*MA[10];
	                fApost[i][j][k][3]=1.0/19*MA[0]-11.0/2394*MA[1]-1.0/63*MA[2]+1.0/10*MA[5]-1.0/10*MA[6]-1.0/36*MA[9]+1.0/36*MA[10]+1.0/12*MA[11]-1.0/12*MA[12];
	                fApost[i][j][k][4]=1.0/19*MA[0]-11.0/2394*MA[1]-1.0/63*MA[2]-1.0/10*MA[5]+1.0/10*MA[6]-1.0/36*MA[9]+1.0/36*MA[10]+1.0/12*MA[11]-1.0/12*MA[12];
	                fApost[i][j][k][5]=1.0/19*MA[0]-11.0/2394*MA[1]-1.0/63*MA[2]+1.0/10*MA[7]-1.0/10*MA[8]-1.0/36*MA[9]+1.0/36*MA[10]-1.0/12*MA[11]+1.0/12*MA[12];
	                fApost[i][j][k][6]=1.0/19*MA[0]-11.0/2394*MA[1]-1.0/63*MA[2]-1.0/10*MA[7]+1.0/10*MA[8]-1.0/36*MA[9]+1.0/36*MA[10]-1.0/12*MA[11]+1.0/12*MA[12];
	                fApost[i][j][k][7]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]+1.0/10*MA[3]+1.0/40*MA[4]+1.0/10*MA[5]+1.0/40*MA[6]+1.0/36*MA[9]+1.0/72*MA[10]+1.0/12*MA[11]+1.0/24*MA[12]+1.0/4*MA[13]+1.0/8*MA[16]-1.0/8*MA[17];
	                fApost[i][j][k][8]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]-1.0/10*MA[3]-1.0/40*MA[4]+1.0/10*MA[5]+1.0/40*MA[6]+1.0/36*MA[9]+1.0/72*MA[10]+1.0/12*MA[11]+1.0/24*MA[12]-1.0/4*MA[13]-1.0/8*MA[16]-1.0/8*MA[17];
	                fApost[i][j][k][9]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]+1.0/10*MA[3]+1.0/40*MA[4]-1.0/10*MA[5]-1.0/40*MA[6]+1.0/36*MA[9]+1.0/72*MA[10]+1.0/12*MA[11]+1.0/24*MA[12]-1.0/4*MA[13]+1.0/8*MA[16]+1.0/8*MA[17];
	                fApost[i][j][k][10]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]-1.0/10*MA[3]-1.0/40*MA[4]-1.0/10*MA[5]-1.0/40*MA[6]+1.0/36*MA[9]+1.0/72*MA[10]+1.0/12*MA[11]+1.0/24*MA[12]+1.0/4*MA[13]-1.0/8*MA[16]+1.0/8*MA[17];
	                fApost[i][j][k][11]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]+1.0/10*MA[3]+1.0/40*MA[4]+1.0/10*MA[7]+1.0/40*MA[8]+1.0/36*MA[9]+1.0/72*MA[10]-1.0/12*MA[11]-1.0/24*MA[12]+1.0/4*MA[15]-1.0/8*MA[16]+1.0/8*MA[18];
	                fApost[i][j][k][12]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]-1.0/10*MA[3]-1.0/40*MA[4]+1.0/10*MA[7]+1.0/40*MA[8]+1.0/36*MA[9]+1.0/72*MA[10]-1.0/12*MA[11]-1.0/24*MA[12]-1.0/4*MA[15]+1.0/8*MA[16]+1.0/8*MA[18];
	                fApost[i][j][k][13]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]+1.0/10*MA[3]+1.0/40*MA[4]-1.0/10*MA[7]-1.0/40*MA[8]+1.0/36*MA[9]+1.0/72*MA[10]-1.0/12*MA[11]-1.0/24*MA[12]-1.0/4*MA[15]-1.0/8*MA[16]-1.0/8*MA[18];
	                fApost[i][j][k][14]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]-1.0/10*MA[3]-1.0/40*MA[4]-1.0/10*MA[7]-1.0/40*MA[8]+1.0/36*MA[9]+1.0/72*MA[10]-1.0/12*MA[11]-1.0/24*MA[12]+1.0/4*MA[15]+1.0/8*MA[16]-1.0/8*MA[18];
	                fApost[i][j][k][15]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]+1.0/10*MA[5]+1.0/40*MA[6]+1.0/10*MA[7]+1.0/40*MA[8]-1.0/18*MA[9]-1.0/36*MA[10]+1.0/4*MA[14]+1.0/8*MA[17]-1.0/8*MA[18];
	                fApost[i][j][k][16]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]-1.0/10*MA[5]-1.0/40*MA[6]+1.0/10*MA[7]+1.0/40*MA[8]-1.0/18*MA[9]-1.0/36*MA[10]-1.0/4*MA[14]-1.0/8*MA[17]-1.0/8*MA[18];
	                fApost[i][j][k][17]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]+1.0/10*MA[5]+1.0/40*MA[6]-1.0/10*MA[7]-1.0/40*MA[8]-1.0/18*MA[9]-1.0/36*MA[10]-1.0/4*MA[14]+1.0/8*MA[17]+1.0/8*MA[18];
	                fApost[i][j][k][18]=1.0/19*MA[0]+4.0/1197*MA[1]+1.0/252*MA[2]-1.0/10*MA[5]-1.0/40*MA[6]-1.0/10*MA[7]-1.0/40*MA[8]-1.0/18*MA[9]-1.0/36*MA[10]+1.0/4*MA[14]-1.0/8*MA[17]+1.0/8*MA[18];

                    mB[0]=fB[i][j][k][0]+fB[i][j][k][1]+fB[i][j][k][2]+fB[i][j][k][3]+fB[i][j][k][4]+fB[i][j][k][5]+fB[i][j][k][6]+fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]+fB[i][j][k][11]+fB[i][j][k][12]+fB[i][j][k][13]+fB[i][j][k][14]+fB[i][j][k][15]+fB[i][j][k][16]+fB[i][j][k][17]+fB[i][j][k][18];
	                mB[1]=-30.0*fB[i][j][k][0]-11.0*fB[i][j][k][1]-11.0*fB[i][j][k][2]-11.0*fB[i][j][k][3]-11.0*fB[i][j][k][4]-11.0*fB[i][j][k][5]-11.0*fB[i][j][k][6]+8.0*fB[i][j][k][7]+8.0*fB[i][j][k][8]+8.0*fB[i][j][k][9]+8.0*fB[i][j][k][10]+8.0*fB[i][j][k][11]+8.0*fB[i][j][k][12]+8.0*fB[i][j][k][13]+8.0*fB[i][j][k][14]+8.0*fB[i][j][k][15]+8.0*fB[i][j][k][16]+8.0*fB[i][j][k][17]+8.0*fB[i][j][k][18];
	                mB[2]=12.0*fB[i][j][k][0]-4.0*fB[i][j][k][1]-4.0*fB[i][j][k][2]-4.0*fB[i][j][k][3]-4.0*fB[i][j][k][4]-4.0*fB[i][j][k][5]-4.0*fB[i][j][k][6]+fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]+fB[i][j][k][11]+fB[i][j][k][12]+fB[i][j][k][13]+fB[i][j][k][14]+fB[i][j][k][15]+fB[i][j][k][16]+fB[i][j][k][17]+fB[i][j][k][18];
	                mB[3]=fB[i][j][k][1]-fB[i][j][k][2]+fB[i][j][k][7]-fB[i][j][k][8]+fB[i][j][k][9]-fB[i][j][k][10]+fB[i][j][k][11]-fB[i][j][k][12]+fB[i][j][k][13]-fB[i][j][k][14];
	                mB[4]=-4.0*fB[i][j][k][1]+4.0*fB[i][j][k][2]+fB[i][j][k][7]-fB[i][j][k][8]+fB[i][j][k][9]-fB[i][j][k][10]+fB[i][j][k][11]-fB[i][j][k][12]+fB[i][j][k][13]-fB[i][j][k][14];
	                mB[5]=fB[i][j][k][3]-fB[i][j][k][4]+fB[i][j][k][7]+fB[i][j][k][8]-fB[i][j][k][9]-fB[i][j][k][10]+fB[i][j][k][15]-fB[i][j][k][16]+fB[i][j][k][17]-fB[i][j][k][18];
	                mB[6]=-4.0*fB[i][j][k][3]+4.0*fB[i][j][k][4]+fB[i][j][k][7]+fB[i][j][k][8]-fB[i][j][k][9]-fB[i][j][k][10]+fB[i][j][k][15]-fB[i][j][k][16]+fB[i][j][k][17]-fB[i][j][k][18];
	                mB[7]=fB[i][j][k][5]-fB[i][j][k][6]+fB[i][j][k][11]+fB[i][j][k][12]-fB[i][j][k][13]-fB[i][j][k][14]+fB[i][j][k][15]+fB[i][j][k][16]-fB[i][j][k][17]-fB[i][j][k][18];
	                mB[8]=-4.0*fB[i][j][k][5]+4.0*fB[i][j][k][6]+fB[i][j][k][11]+fB[i][j][k][12]-fB[i][j][k][13]-fB[i][j][k][14]+fB[i][j][k][15]+fB[i][j][k][16]-fB[i][j][k][17]-fB[i][j][k][18];
	                mB[9]=2.0*fB[i][j][k][1]+2.0*fB[i][j][k][2]-fB[i][j][k][3]-fB[i][j][k][4]-fB[i][j][k][5]-fB[i][j][k][6]+fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]+fB[i][j][k][11]+fB[i][j][k][12]+fB[i][j][k][13]+fB[i][j][k][14]-2.0*fB[i][j][k][15]-2.0*fB[i][j][k][16]-2.0*fB[i][j][k][17]-2.0*fB[i][j][k][18];
	                mB[10]=-4.0*fB[i][j][k][1]-4.0*fB[i][j][k][2]+2.0*fB[i][j][k][3]+2.0*fB[i][j][k][4]+2.0*fB[i][j][k][5]+2.0*fB[i][j][k][6]+fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]+fB[i][j][k][11]+fB[i][j][k][12]+fB[i][j][k][13]+fB[i][j][k][14]-2.0*fB[i][j][k][15]-2.0*fB[i][j][k][16]-2.0*fB[i][j][k][17]-2.0*fB[i][j][k][18];
	                mB[11]=fB[i][j][k][3]+fB[i][j][k][4]-fB[i][j][k][5]-fB[i][j][k][6]+fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]-fB[i][j][k][11]-fB[i][j][k][12]-fB[i][j][k][13]-fB[i][j][k][14];
	                mB[12]=-2.0*fB[i][j][k][3]-2.0*fB[i][j][k][4]+2.0*fB[i][j][k][5]+2.0*fB[i][j][k][6]+fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]-fB[i][j][k][11]-fB[i][j][k][12]-fB[i][j][k][13]-fB[i][j][k][14];
	                mB[13]=fB[i][j][k][7]-fB[i][j][k][8]-fB[i][j][k][9]+fB[i][j][k][10];
	                mB[14]=fB[i][j][k][15]-fB[i][j][k][16]-fB[i][j][k][17]+fB[i][j][k][18];
	                mB[15]=fB[i][j][k][11]-fB[i][j][k][12]-fB[i][j][k][13]+fB[i][j][k][14];
	                mB[16]=fB[i][j][k][7]-fB[i][j][k][8]+fB[i][j][k][9]-fB[i][j][k][10]-fB[i][j][k][11]+fB[i][j][k][12]-fB[i][j][k][13]+fB[i][j][k][14];
	                mB[17]=-fB[i][j][k][7]-fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]+fB[i][j][k][15]-fB[i][j][k][16]+fB[i][j][k][17]-fB[i][j][k][18];
	                mB[18]=fB[i][j][k][11]+fB[i][j][k][12]-fB[i][j][k][13]-fB[i][j][k][14]-fB[i][j][k][15]-fB[i][j][k][16]+fB[i][j][k][17]+fB[i][j][k][18];
	                for(l=0;l<Q;l++)
	                {
		                MB[l]=mB[l]-sB[l]*(mB[l]-meq(l,rhoB[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]))+(1-sB[l]/2.0)*sscB[i][j][k][l];
	                }
	                fBpost[i][j][k][0]=1.0/19*MB[0]-5.0/399*MB[1]+1.0/21*MB[2];
	                fBpost[i][j][k][1]=1.0/19*MB[0]-11.0/2394*MB[1]-1.0/63*MB[2]+1.0/10*MB[3]-1.0/10*MB[4]+1.0/18*MB[9]-1.0/18*MB[10];
	                fBpost[i][j][k][2]=1.0/19*MB[0]-11.0/2394*MB[1]-1.0/63*MB[2]-1.0/10*MB[3]+1.0/10*MB[4]+1.0/18*MB[9]-1.0/18*MB[10];
	                fBpost[i][j][k][3]=1.0/19*MB[0]-11.0/2394*MB[1]-1.0/63*MB[2]+1.0/10*MB[5]-1.0/10*MB[6]-1.0/36*MB[9]+1.0/36*MB[10]+1.0/12*MB[11]-1.0/12*MB[12];
	                fBpost[i][j][k][4]=1.0/19*MB[0]-11.0/2394*MB[1]-1.0/63*MB[2]-1.0/10*MB[5]+1.0/10*MB[6]-1.0/36*MB[9]+1.0/36*MB[10]+1.0/12*MB[11]-1.0/12*MB[12];
	                fBpost[i][j][k][5]=1.0/19*MB[0]-11.0/2394*MB[1]-1.0/63*MB[2]+1.0/10*MB[7]-1.0/10*MB[8]-1.0/36*MB[9]+1.0/36*MB[10]-1.0/12*MB[11]+1.0/12*MB[12];
	                fBpost[i][j][k][6]=1.0/19*MB[0]-11.0/2394*MB[1]-1.0/63*MB[2]-1.0/10*MB[7]+1.0/10*MB[8]-1.0/36*MB[9]+1.0/36*MB[10]-1.0/12*MB[11]+1.0/12*MB[12];
	                fBpost[i][j][k][7]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]+1.0/10*MB[3]+1.0/40*MB[4]+1.0/10*MB[5]+1.0/40*MB[6]+1.0/36*MB[9]+1.0/72*MB[10]+1.0/12*MB[11]+1.0/24*MB[12]+1.0/4*MB[13]+1.0/8*MB[16]-1.0/8*MB[17];
	                fBpost[i][j][k][8]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]-1.0/10*MB[3]-1.0/40*MB[4]+1.0/10*MB[5]+1.0/40*MB[6]+1.0/36*MB[9]+1.0/72*MB[10]+1.0/12*MB[11]+1.0/24*MB[12]-1.0/4*MB[13]-1.0/8*MB[16]-1.0/8*MB[17];
	                fBpost[i][j][k][9]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]+1.0/10*MB[3]+1.0/40*MB[4]-1.0/10*MB[5]-1.0/40*MB[6]+1.0/36*MB[9]+1.0/72*MB[10]+1.0/12*MB[11]+1.0/24*MB[12]-1.0/4*MB[13]+1.0/8*MB[16]+1.0/8*MB[17];
	                fBpost[i][j][k][10]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]-1.0/10*MB[3]-1.0/40*MB[4]-1.0/10*MB[5]-1.0/40*MB[6]+1.0/36*MB[9]+1.0/72*MB[10]+1.0/12*MB[11]+1.0/24*MB[12]+1.0/4*MB[13]-1.0/8*MB[16]+1.0/8*MB[17];
	                fBpost[i][j][k][11]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]+1.0/10*MB[3]+1.0/40*MB[4]+1.0/10*MB[7]+1.0/40*MB[8]+1.0/36*MB[9]+1.0/72*MB[10]-1.0/12*MB[11]-1.0/24*MB[12]+1.0/4*MB[15]-1.0/8*MB[16]+1.0/8*MB[18];
	                fBpost[i][j][k][12]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]-1.0/10*MB[3]-1.0/40*MB[4]+1.0/10*MB[7]+1.0/40*MB[8]+1.0/36*MB[9]+1.0/72*MB[10]-1.0/12*MB[11]-1.0/24*MB[12]-1.0/4*MB[15]+1.0/8*MB[16]+1.0/8*MB[18];
	                fBpost[i][j][k][13]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]+1.0/10*MB[3]+1.0/40*MB[4]-1.0/10*MB[7]-1.0/40*MB[8]+1.0/36*MB[9]+1.0/72*MB[10]-1.0/12*MB[11]-1.0/24*MB[12]-1.0/4*MB[15]-1.0/8*MB[16]-1.0/8*MB[18];
	                fBpost[i][j][k][14]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]-1.0/10*MB[3]-1.0/40*MB[4]-1.0/10*MB[7]-1.0/40*MB[8]+1.0/36*MB[9]+1.0/72*MB[10]-1.0/12*MB[11]-1.0/24*MB[12]+1.0/4*MB[15]+1.0/8*MB[16]-1.0/8*MB[18];
	                fBpost[i][j][k][15]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]+1.0/10*MB[5]+1.0/40*MB[6]+1.0/10*MB[7]+1.0/40*MB[8]-1.0/18*MB[9]-1.0/36*MB[10]+1.0/4*MB[14]+1.0/8*MB[17]-1.0/8*MB[18];
	                fBpost[i][j][k][16]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]-1.0/10*MB[5]-1.0/40*MB[6]+1.0/10*MB[7]+1.0/40*MB[8]-1.0/18*MB[9]-1.0/36*MB[10]-1.0/4*MB[14]-1.0/8*MB[17]-1.0/8*MB[18];
	                fBpost[i][j][k][17]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]+1.0/10*MB[5]+1.0/40*MB[6]-1.0/10*MB[7]-1.0/40*MB[8]-1.0/18*MB[9]-1.0/36*MB[10]-1.0/4*MB[14]+1.0/8*MB[17]+1.0/8*MB[18];
	                fBpost[i][j][k][18]=1.0/19*MB[0]+4.0/1197*MB[1]+1.0/252*MB[2]-1.0/10*MB[5]-1.0/40*MB[6]-1.0/10*MB[7]-1.0/40*MB[8]-1.0/18*MB[9]-1.0/36*MB[10]+1.0/4*MB[14]-1.0/8*MB[17]+1.0/8*MB[18];

                    for(l=0;l<Q;l++)
	                {
		                fApost[i][j][k][l]=fApost[i][j][k][l]+w[l]*SmA[i][j][k];
                        fBpost[i][j][k][l]=fBpost[i][j][k][l]+w[l]*SmB[i][j][k];
	                }
                }
            }
}

void Infosendrecvf()
{
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
            for(l=0;l<Q;l++)
            {
                sendfA_right[l+(j-1+(k-1)*(NY-2))*Q]=fApost[NX-2][j][k][l];
                sendfA_left[l+(j-1+(k-1)*(NY-2))*Q]=fApost[1][j][k][l];
                sendfB_right[l+(j-1+(k-1)*(NY-2))*Q]=fBpost[NX-2][j][k][l];
                sendfB_left[l+(j-1+(k-1)*(NY-2))*Q]=fBpost[1][j][k][l];
            }
    MPI_Sendrecv(&sendfA_right[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destright,tagrfA,
                    &recvfA_left[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destleft,tagrfA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfA_left[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destleft,taglfA,
                    &recvfA_right[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destright,taglfA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_right[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destright,tagrfB,
                    &recvfB_left[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destleft,tagrfB,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_left[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destleft,taglfB,
                    &recvfB_right[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destright,taglfB,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
            for(l=0;l<Q;l++)
            {
                fApost[NX-1][j][k][l]=recvfA_right[l+(j-1+(k-1)*(NY-2))*Q];
                fApost[0][j][k][l]=recvfA_left[l+(j-1+(k-1)*(NY-2))*Q];
                fBpost[NX-1][j][k][l]=recvfB_right[l+(j-1+(k-1)*(NY-2))*Q];
                fBpost[0][j][k][l]=recvfB_left[l+(j-1+(k-1)*(NY-2))*Q];
            }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
            for(l=0;l<Q;l++)
            {
                sendfA_front[l+(i+(k-1)*(NX))*Q]=fApost[i][NY-2][k][l];
                sendfA_back[l+(i+(k-1)*(NX))*Q]=fApost[i][1][k][l];
                sendfB_front[l+(i+(k-1)*(NX))*Q]=fBpost[i][NY-2][k][l];
                sendfB_back[l+(i+(k-1)*(NX))*Q]=fBpost[i][1][k][l];
            }
    MPI_Sendrecv(&sendfA_front[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destfront,tagffA,
                    &recvfA_back[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destback,tagffA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfA_back[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destback,tagbfA,
                    &recvfA_front[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destfront,tagbfA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_front[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destfront,tagffB,
                    &recvfB_back[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destback,tagffB,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_back[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destback,tagbfB,
                    &recvfB_front[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destfront,tagbfB,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
            for(l=0;l<Q;l++)
            {
                fApost[i][NY-1][k][l]=recvfA_front[l+(i+(k-1)*(NX))*Q];
                fApost[i][0][k][l]=recvfA_back[l+(i+(k-1)*(NX))*Q];
                fBpost[i][NY-1][k][l]=recvfB_front[l+(i+(k-1)*(NX))*Q];
                fBpost[i][0][k][l]=recvfB_back[l+(i+(k-1)*(NX))*Q];
            }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
            for(l=0;l<Q;l++)
            {
                sendfA_up[l+(i+j*(NX))*Q]=fApost[i][j][NZ-2][l];
                sendfA_down[l+(i+j*(NX))*Q]=fApost[i][j][1][l];
                sendfB_up[l+(i+j*(NX))*Q]=fBpost[i][j][NZ-2][l];
                sendfB_down[l+(i+j*(NX))*Q]=fBpost[i][j][1][l];
            }
    MPI_Sendrecv(&sendfA_up[0],(NX)*(NY)*Q,MPI_DOUBLE,destup,tagufA,
                    &recvfA_down[0],(NX)*(NY)*Q,MPI_DOUBLE,destdown,tagufA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfA_down[0],(NX)*(NY)*Q,MPI_DOUBLE,destdown,tagdfA,
                    &recvfA_up[0],(NX)*(NY)*Q,MPI_DOUBLE,destup,tagdfA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_up[0],(NX)*(NY)*Q,MPI_DOUBLE,destup,tagufB,
                    &recvfB_down[0],(NX)*(NY)*Q,MPI_DOUBLE,destdown,tagufB,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_down[0],(NX)*(NY)*Q,MPI_DOUBLE,destdown,tagdfB,
                    &recvfB_up[0],(NX)*(NY)*Q,MPI_DOUBLE,destup,tagdfB,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
            for(l=0;l<Q;l++)
            {
                fApost[i][j][NZ-1][l]=recvfA_up[l+(i+j*(NX))*Q];
                fApost[i][j][0][l]=recvfA_down[l+(i+j*(NX))*Q];
                fBpost[i][j][NZ-1][l]=recvfB_up[l+(i+j*(NX))*Q];
                fBpost[i][j][0][l]=recvfB_down[l+(i+j*(NX))*Q];
            }
    MPI_Barrier(MPI_COMM_WORLD);
}

void Boundary()
{
    int ip,jp,kp;
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(bounce[i][j][k]==1.0)
                {
                    for(l=0;l<Q;l++)
                    {
                        ip=min(NX-1,max((i+ex[l]),0));
                        jp=min(NY-1,max((j+ey[l]),0));
                        kp=min(NZ-1,max((k+ez[l]),0));
                        fApost[i][j][k][l]=fApost[ip][jp][kp][op[l]];
                        fBpost[i][j][k][l]=fBpost[ip][jp][kp][op[l]];
                    }
                }

            }   
}

void Boundary2()
{
    if(rankx==0)
    {
        Inlet();
    }
    if(rankx==PX-1)
    {
        Outlet();
    }

    int ip,jp,kp;
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(bounce[i][j][k]==1.0)
                {
                    for(l=0;l<Q;l++)
                    {
                        ip=min(NX-1,max((i+ex[l]),0));
                        jp=min(NY-1,max((j+ey[l]),0));
                        kp=min(NZ-1,max((k+ez[l]),0));
                        fApost[i][j][k][l]=fApost[ip][jp][kp][op[l]];
                        fBpost[i][j][k][l]=fBpost[ip][jp][kp][op[l]];
                    }
                }

            }
}

void InletZH()
{
    if(rankx==0)
    {
        double uinlet, rhoinlet;
        double sigmafA, sigmafB;
        double NyfA, NyfB, NzfA, NzfB;
        uinlet=0.001;
        double rhouA[NY][NZ];
        double rhouB[NY][NZ];
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                i=1;
                sigmafA=fA[i][j][k][0]+fA[i][j][k][3]+fA[i][j][k][4]+fA[i][j][k][5]+fA[i][j][k][6]+fA[i][j][k][15]+fA[i][j][k][16]+fA[i][j][k][17]+fA[i][j][k][18]+2.0*(fA[i][j][k][2]+fA[i][j][k][8]+fA[i][j][k][10]+fA[i][j][k][12]+fA[i][j][k][14]);
                sigmafB=fB[i][j][k][0]+fB[i][j][k][3]+fB[i][j][k][4]+fB[i][j][k][5]+fB[i][j][k][6]+fB[i][j][k][15]+fB[i][j][k][16]+fB[i][j][k][17]+fB[i][j][k][18]+2.0*(fB[i][j][k][2]+fB[i][j][k][8]+fB[i][j][k][10]+fB[i][j][k][12]+fB[i][j][k][14]);

                rhouA[j][k]=rhoAin-sigmafA;
                rhouB[j][k]=rhoBin-sigmafB;

                NyfA=fA[i][j][k][3]+fA[i][j][k][15]+fA[i][j][k][17]-fA[i][j][k][4]-fA[i][j][k][16]-fA[i][j][k][18];
                NzfA=fA[i][j][k][5]+fA[i][j][k][15]+fA[i][j][k][16]-fA[i][j][k][6]-fA[i][j][k][17]-fA[i][j][k][18];
                NyfB=fB[i][j][k][3]+fB[i][j][k][15]+fB[i][j][k][17]-fB[i][j][k][4]-fB[i][j][k][16]-fB[i][j][k][18];
                NzfB=fB[i][j][k][5]+fB[i][j][k][15]+fB[i][j][k][16]-fB[i][j][k][6]-fB[i][j][k][17]-fB[i][j][k][18];

                fA[i][j][k][1]=fA[i][j][k][2]+(rhouA[j][k])/3.0;
                fA[i][j][k][7]=fA[i][j][k][10]+(rhouA[j][k])/6.0-0.5*NyfA;
                fA[i][j][k][9]=fA[i][j][k][8]+(rhouA[j][k])/6.0+0.5*NyfA;
                fA[i][j][k][11]=fA[i][j][k][14]+(rhouA[j][k])/6.0-0.5*NzfA;
                fA[i][j][k][13]=fA[i][j][k][12]+(rhouA[j][k])/6.0+0.5*NzfA;

                fB[i][j][k][1]=fB[i][j][k][2]+(rhouB[j][k])/3.0;
                fB[i][j][k][7]=fB[i][j][k][10]+(rhouB[j][k])/6.0-0.5*NyfB;
                fB[i][j][k][9]=fB[i][j][k][8]+(rhouB[j][k])/6.0+0.5*NyfB;
                fB[i][j][k][11]=fB[i][j][k][14]+(rhouB[j][k])/6.0-0.5*NzfB;
                fB[i][j][k][13]=fB[i][j][k][12]+(rhouB[j][k])/6.0+0.5*NzfB;

                }
    }
}


void Streaming()
{
    int iq, jq, kq;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]<1.0)
                {
                    for(l=0;l<Q;l++)
                    {
                        iq=i-ex[l];
                        jq=j-ey[l];
                        kq=k-ez[l];
                        fA[i][j][k][l]=fApost[iq][jq][kq][l];
                        fB[i][j][k][l]=fBpost[iq][jq][kq][l];
                    }
                }
            }
                
}

void Macro()
{
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]<1.0)
                {
                    rhoA0[i][j][k]=rhoA[i][j][k];
                    rhoB0[i][j][k]=rhoB[i][j][k];
                    ux0[i][j][k]=ux[i][j][k];
                    uy0[i][j][k]=uy[i][j][k];
                    uz0[i][j][k]=uz[i][j][k];

                    rhoA[i][j][k]=0.0;
                    rhoB[i][j][k]=0.0;
                    ux[i][j][k]=0.0;
                    uy[i][j][k]=0.0;
                    uz[i][j][k]=0.0;

                    for(l=0;l<Q;l++)
                    {
                        rhoA[i][j][k]+=fA[i][j][k][l];
                        rhoB[i][j][k]+=fB[i][j][k][l];
                        ux[i][j][k]+=ex[l]*fA[i][j][k][l];
                        ux[i][j][k]+=ex[l]*fB[i][j][k][l];
                        uy[i][j][k]+=ey[l]*fA[i][j][k][l];
                        uy[i][j][k]+=ey[l]*fB[i][j][k][l];
                        uz[i][j][k]+=ez[l]*fA[i][j][k][l];
                        uz[i][j][k]+=ez[l]*fB[i][j][k][l];
                    }

                    rho[i][j][k]=rhoA[i][j][k]+rhoB[i][j][k];
                    ux[i][j][k]=(ux[i][j][k]+fscAx[i][j][k]/2.0+fscBx[i][j][k]/2.0)/rho[i][j][k];
                    uy[i][j][k]=(uy[i][j][k]+fscAy[i][j][k]/2.0+fscBy[i][j][k]/2.0)/rho[i][j][k];
                    uz[i][j][k]=(uz[i][j][k]+fscAz[i][j][k]/2.0+fscBz[i][j][k]/2.0)/rho[i][j][k];
                }                
            }

}

void Infosendrecvmacro()
{
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            sendmacro_right[(j-1+(k-1)*(NY-2))*Nmacro]=rhoA[NX-2][j][k];
            sendmacro_left[(j-1+(k-1)*(NY-2))*Nmacro]=rhoA[1][j][k];
            sendmacro_right[(j-1+(k-1)*(NY-2))*Nmacro+1]=rhoB[NX-2][j][k];
            sendmacro_left[(j-1+(k-1)*(NY-2))*Nmacro+1]=rhoB[1][j][k];
        }
    MPI_Sendrecv(&sendmacro_right[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destright,tagrm,
                    &recvmacro_left[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destleft,tagrm,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_left[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destleft,taglm,
                    &recvmacro_right[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destright,taglm,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            rhoA[NX-1][j][k]=recvmacro_right[(j-1+(k-1)*(NY-2))*Nmacro];
            rhoA[0][j][k]=recvmacro_left[(j-1+(k-1)*(NY-2))*Nmacro];
            rhoB[NX-1][j][k]=recvmacro_right[(j-1+(k-1)*(NY-2))*Nmacro+1];
            rhoB[0][j][k]=recvmacro_left[(j-1+(k-1)*(NY-2))*Nmacro+1];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            sendmacro_front[(i+(k-1)*(NX))*Nmacro]=rhoA[i][NY-2][k];
            sendmacro_back[(i+(k-1)*(NX))*Nmacro]=rhoA[i][1][k];
            sendmacro_front[(i+(k-1)*(NX))*Nmacro+1]=rhoB[i][NY-2][k];
            sendmacro_back[(i+(k-1)*(NX))*Nmacro+1]=rhoB[i][1][k];
        }
    MPI_Sendrecv(&sendmacro_front[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destfront,tagfm,
                    &recvmacro_back[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destback,tagfm,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_back[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destback,tagbm,
                    &recvmacro_front[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destfront,tagbm,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            rhoA[i][NY-1][k]=recvmacro_front[(i+(k-1)*(NX))*Nmacro];
            rhoA[i][0][k]=recvmacro_back[(i+(k-1)*(NX))*Nmacro];
            rhoB[i][NY-1][k]=recvmacro_front[(i+(k-1)*(NX))*Nmacro+1];
            rhoB[i][0][k]=recvmacro_back[(i+(k-1)*(NX))*Nmacro+1];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            sendmacro_up[(i+j*(NX))*Nmacro]=rhoA[i][j][NZ-2];
            sendmacro_down[(i+j*(NX))*Nmacro]=rhoA[i][j][1];
            sendmacro_up[(i+j*(NX))*Nmacro+1]=rhoB[i][j][NZ-2];
            sendmacro_down[(i+j*(NX))*Nmacro+1]=rhoB[i][j][1];
        }
    MPI_Sendrecv(&sendmacro_up[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destup,tagum,
                    &recvmacro_down[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destdown,tagum,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_down[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destdown,tagdm,
                    &recvmacro_up[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destup,tagdm,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            rhoA[i][j][NZ-1]=recvmacro_up[(i+j*(NX))*Nmacro];
            rhoA[i][j][0]=recvmacro_down[(i+j*(NX))*Nmacro];
            rhoB[i][j][NZ-1]=recvmacro_up[(i+j*(NX))*Nmacro+1];
            rhoB[i][j][0]=recvmacro_down[(i+j*(NX))*Nmacro+1];
        }

}

void Getbounce()
{
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                bounce[i][j][k]=0.0;
                if(solid[i][j][k]>=1.0)
                {
                    for(l=0;l<Q;l++)
                    {
                        int ib, jb, kb;
                        ib=i+ex[l];
                        jb=j+ey[l];
                        kb=k+ez[l];
                        if(solid[ib][jb][kb]<1.0)
                        {
                            bounce[i][j][k]=1.0;
                            break;
                        }
                    }
                }
            }

    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            send_right[j-1+(k-1)*(NY-2)]=bounce[NX-2][j][k];
            send_left[j-1+(k-1)*(NY-2)]=bounce[1][j][k];
        }
    MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,6001,
                    &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,6001,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,6002,
                    &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,6002,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            bounce[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
            bounce[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            send_front[i+(k-1)*(NX)]=bounce[i][NY-2][k];
            send_back[i+(k-1)*(NX)]=bounce[i][1][k];
        }
    MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,6003,
                    &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,6003,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,6004,
                    &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,6004,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            bounce[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
            bounce[i][0][k]=recv_back[i+(k-1)*(NX)];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            send_up[i+j*(NX)]=bounce[i][j][NZ-2];
            send_down[i+j*(NX)]=bounce[i][j][1];
        }
    MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,6005,
                    &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,6005,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,6006,
                    &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,6006,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            bounce[i][j][NZ-1]=recv_up[i+j*(NX)];
            bounce[i][j][0]=recv_down[i+j*(NX)];
        }

    /*if(rankz==0)
    {
        for(i=0;i<NX;i++)
            for(j=0;j<NY;j++)
            {
                bounce[i][j][0]=1.0;
            }
    }
    if(rankz==PZ-1)
    {
        for(i=0;i<NX;i++)
            for(j=0;j<NY;j++)
            {
                bounce[i][j][NZ-1]=1.0;
            }
    }
    if(ranky==0)
    {
        for(i=0;i<NX;i++)
            for(k=0;k<NZ;k++)
            {
                bounce[i][0][k]=1.0;
            }
    }
    if(ranky==PY-1)
    {
        for(i=0;i<NX;i++)
            for(k=0;k<NZ;k++)
            {
                bounce[i][NY-1][k]=1.0;
            }
    }*/
    /*if(rankx==0)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                bounce[0][j][k]=0.0;
            }
    }
    if(rankx==PX-1)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                bounce[NX-1][j][k]=0.0;
            }
    }*/
}

void Phisolid()
{
    double aveSA, aveSB, aven;
    int ip, jp, kp;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]>=1.0)
                {
                    aveSA=0.0;
                    aveSB=0.0;
                    aven=0.0;
                    for(l=0;l<Q;l++)
                    {
                        ip=i+ex[l];
                        jp=j+ey[l];
                        kp=k+ez[l];
                        if(solid[ip][jp][kp]<1.0)
                        {
                            aveSA=aveSA+w[l]*rhoA[ip][jp][kp];
                            aveSB=aveSB+w[l]*rhoB[ip][jp][kp];
                            aven=aven+w[l];
                        }
                    }
                    if(aven>0.0)
                    {
                        rhoA[i][j][k]=max(0.01,min(rholA,lA*aveSA/aven));
                        rhoB[i][j][k]=max(0.001,min(rholB,lB*aveSB/aven));
                    }
                    if(aven==0.0)
                    {
                        rhoA[i][j][k]=rholA;
                        rhoB[i][j][k]=rhogB;
                    }
                }
            }
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            sendmacro_right[(j-1+(k-1)*(NY-2))*Nmacro]=rhoA[NX-2][j][k];
            sendmacro_left[(j-1+(k-1)*(NY-2))*Nmacro]=rhoA[1][j][k];
            sendmacro_right[(j-1+(k-1)*(NY-2))*Nmacro+1]=rhoB[NX-2][j][k];
            sendmacro_left[(j-1+(k-1)*(NY-2))*Nmacro+1]=rhoB[1][j][k];
        }
    MPI_Sendrecv(&sendmacro_right[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destright,7001,
                    &recvmacro_left[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destleft,7001,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_left[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destleft,7002,
                    &recvmacro_right[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destright,7002,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            rhoA[NX-1][j][k]=recvmacro_right[(j-1+(k-1)*(NY-2))*Nmacro];
            rhoA[0][j][k]=recvmacro_left[(j-1+(k-1)*(NY-2))*Nmacro];
            rhoB[NX-1][j][k]=recvmacro_right[(j-1+(k-1)*(NY-2))*Nmacro+1];
            rhoB[0][j][k]=recvmacro_left[(j-1+(k-1)*(NY-2))*Nmacro+1];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            sendmacro_front[(i+(k-1)*(NX))*Nmacro]=rhoA[i][NY-2][k];
            sendmacro_back[(i+(k-1)*(NX))*Nmacro]=rhoA[i][1][k];
            sendmacro_front[(i+(k-1)*(NX))*Nmacro+1]=rhoB[i][NY-2][k];
            sendmacro_back[(i+(k-1)*(NX))*Nmacro+1]=rhoB[i][1][k];
        }
    MPI_Sendrecv(&sendmacro_front[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destfront,7003,
                    &recvmacro_back[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destback,7003,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_back[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destback,7004,
                    &recvmacro_front[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destfront,7004,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            rhoA[i][NY-1][k]=recvmacro_front[(i+(k-1)*(NX))*Nmacro];
            rhoA[i][0][k]=recvmacro_back[(i+(k-1)*(NX))*Nmacro];
            rhoB[i][NY-1][k]=recvmacro_front[(i+(k-1)*(NX))*Nmacro+1];
            rhoB[i][0][k]=recvmacro_back[(i+(k-1)*(NX))*Nmacro+1];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            sendmacro_up[(i+j*(NX))*Nmacro]=rhoA[i][j][NZ-2];
            sendmacro_down[(i+j*(NX))*Nmacro]=rhoA[i][j][1];
            sendmacro_up[(i+j*(NX))*Nmacro+1]=rhoB[i][j][NZ-2];
            sendmacro_down[(i+j*(NX))*Nmacro+1]=rhoB[i][j][1];
        }
    MPI_Sendrecv(&sendmacro_up[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destup,7005,
                    &recvmacro_down[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destdown,7005,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_down[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destdown,7006,
                    &recvmacro_up[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destup,7006,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            rhoA[i][j][NZ-1]=recvmacro_up[(i+j*(NX))*Nmacro];
            rhoA[i][j][0]=recvmacro_down[(i+j*(NX))*Nmacro];
            rhoB[i][j][NZ-1]=recvmacro_up[(i+j*(NX))*Nmacro+1];
            rhoB[i][j][0]=recvmacro_down[(i+j*(NX))*Nmacro+1];
        }


    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                Phipsi();
            }
}

void Phisolid2()
{
    if(rankx==0)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                rhoA[0][j][k]=rhoAin;
                rhoB[0][j][k]=rhoBin;
            }
    }
    /*if(rankx==PX-1)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                rhoA[NX-1][j][k]=rhoAout;
                rhoB[NX-1][j][k]=rhoBout;
                i=NX-1;
                rhoA[i][j][k]=rhoA[i][j][k]-(1.5*rhoA[i][j][k]*ux[i][j][k]-2.0*rhoA[i-1][j][k]*ux[i-1][j][k]+0.5*rhoA[i-2][j][k]*ux[i-2][j][k]);
                rhoA[i][j][k]=max(0.01,min(rholA,rhoA[i][j][k]));
                rhoB[i][j][k]=rhoB[i][j][k]-(1.5*rhoB[i][j][k]*ux[i][j][k]-2.0*rhoB[i-1][j][k]*ux[i-1][j][k]+0.5*rhoB[i-2][j][k]*ux[i-2][j][k]);
                rhoB[i][j][k]=max(0.001,min(rholB,rhoB[i][j][k]));
            }
    }*/

    double aveSA, aveSB, aven;
    int ip, jp, kp;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]>=1.0)
                {
                    aveSA=0.0;
                    aveSB=0.0;
                    aven=0.0;
                    for(l=0;l<Q;l++)
                    {
                        ip=i+ex[l];
                        jp=j+ey[l];
                        kp=k+ez[l];
                        if(solid[ip][jp][kp]<1.0)
                        {
                            aveSA=aveSA+w[l]*rhoA[ip][jp][kp];
                            aveSB=aveSB+w[l]*rhoB[ip][jp][kp];
                            aven=aven+w[l];
                        }
                    }
                    if(aven>0.0)
                    {
                        rhoA[i][j][k]=max(0.01,min(rholA,lA*aveSA/aven));
                        rhoB[i][j][k]=max(0.001,min(rholB,lB*aveSB/aven));
                    }
                    if(aven==0.0)
                    {
                        rhoA[i][j][k]=rholA;
                        rhoB[i][j][k]=rhogB;
                    }
                }
            }
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            sendmacro_right[(j-1+(k-1)*(NY-2))*Nmacro]=rhoA[NX-2][j][k];
            sendmacro_left[(j-1+(k-1)*(NY-2))*Nmacro]=rhoA[1][j][k];
            sendmacro_right[(j-1+(k-1)*(NY-2))*Nmacro+1]=rhoB[NX-2][j][k];
            sendmacro_left[(j-1+(k-1)*(NY-2))*Nmacro+1]=rhoB[1][j][k];
        }
    MPI_Sendrecv(&sendmacro_right[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destright,7001,
                    &recvmacro_left[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destleft,7001,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_left[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destleft,7002,
                    &recvmacro_right[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destright,7002,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            rhoA[NX-1][j][k]=recvmacro_right[(j-1+(k-1)*(NY-2))*Nmacro];
            rhoA[0][j][k]=recvmacro_left[(j-1+(k-1)*(NY-2))*Nmacro];
            rhoB[NX-1][j][k]=recvmacro_right[(j-1+(k-1)*(NY-2))*Nmacro+1];
            rhoB[0][j][k]=recvmacro_left[(j-1+(k-1)*(NY-2))*Nmacro+1];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            sendmacro_front[(i+(k-1)*(NX))*Nmacro]=rhoA[i][NY-2][k];
            sendmacro_back[(i+(k-1)*(NX))*Nmacro]=rhoA[i][1][k];
            sendmacro_front[(i+(k-1)*(NX))*Nmacro+1]=rhoB[i][NY-2][k];
            sendmacro_back[(i+(k-1)*(NX))*Nmacro+1]=rhoB[i][1][k];
        }
    MPI_Sendrecv(&sendmacro_front[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destfront,7003,
                    &recvmacro_back[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destback,7003,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_back[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destback,7004,
                    &recvmacro_front[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destfront,7004,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            rhoA[i][NY-1][k]=recvmacro_front[(i+(k-1)*(NX))*Nmacro];
            rhoA[i][0][k]=recvmacro_back[(i+(k-1)*(NX))*Nmacro];
            rhoB[i][NY-1][k]=recvmacro_front[(i+(k-1)*(NX))*Nmacro+1];
            rhoB[i][0][k]=recvmacro_back[(i+(k-1)*(NX))*Nmacro+1];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            sendmacro_up[(i+j*(NX))*Nmacro]=rhoA[i][j][NZ-2];
            sendmacro_down[(i+j*(NX))*Nmacro]=rhoA[i][j][1];
            sendmacro_up[(i+j*(NX))*Nmacro+1]=rhoB[i][j][NZ-2];
            sendmacro_down[(i+j*(NX))*Nmacro+1]=rhoB[i][j][1];
        }
    MPI_Sendrecv(&sendmacro_up[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destup,7005,
                    &recvmacro_down[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destdown,7005,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_down[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destdown,7006,
                    &recvmacro_up[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destup,7006,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            rhoA[i][j][NZ-1]=recvmacro_up[(i+j*(NX))*Nmacro];
            rhoA[i][j][0]=recvmacro_down[(i+j*(NX))*Nmacro];
            rhoB[i][j][NZ-1]=recvmacro_up[(i+j*(NX))*Nmacro+1];
            rhoB[i][j][0]=recvmacro_down[(i+j*(NX))*Nmacro+1];
        }

    if(rankx==0)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                rhoA[0][j][k]=rhoAin;
                rhoB[0][j][k]=rhoBin;
            }
    }
    /*if(rankx==PX-1)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                rhoA[NX-1][j][k]=rhoAout;
                rhoB[NX-1][j][k]=rhoBout;
                i=NX-1;
                rhoA[i][j][k]=rhoA[i][j][k]-(1.5*rhoA[i][j][k]*ux[i][j][k]-2.0*rhoA[i-1][j][k]*ux[i-1][j][k]+0.5*rhoA[i-2][j][k]*ux[i-2][j][k]);
                rhoA[i][j][k]=max(0.01,min(rholA,rhoA[i][j][k]));
                rhoB[i][j][k]=rhoB[i][j][k]-(1.5*rhoB[i][j][k]*ux[i][j][k]-2.0*rhoB[i-1][j][k]*ux[i-1][j][k]+0.5*rhoB[i-2][j][k]*ux[i-2][j][k]);
                rhoB[i][j][k]=max(0.01,min(rholB,rhoB[i][j][k]));
            }
    }*/
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                Phipsi();
            }
}

void Vof()
{
    double vofmax=1.0;
    double vofmin=0.0;
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(phiA[i][j][k]!=0.0&&phiB[i][j][k]!=0.0)
                {
                    vof[i][j][k]=phiA[i][j][k]/(phiA[i][j][k]+phiB[i][j][k]);
                }
                else
                {
                    vof[i][j][k]=1.0;
                }

                vof[i][j][k]=(vof[i][j][k]-vofmin)/(vofmax-vofmin);
                if(vof[i][j][k]>1.0)
			    {
				    vof[i][j][k]=1.0;
			    }
			    if(vof[i][j][k]<0.0)
			    {
				    vof[i][j][k]=0.0;
			    }
            }
}

void Input()
{
    FILE *fp;
    char filenamedata[20];
    sprintf(filenamedata,"./data/%s%.4d%s","data",mpirank,".dat");
    fp=fopen(filenamedata,"r");

    if(NULL==fp)
    {
        cout<<mpirank<<"data open error"<<endl;
    }

    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                fscanf(fp,"%i\n",&data[i][j][k]);
            }
    fclose(fp);
}

void Output(int m)
{
    ostringstream command;
    command<<"mkdir -p "<<"./LBM_"<<m;
    system(command.str().c_str());

    ostringstream name;
	name<<"./LBM_"<<m<<"/Proc_"<<mpirank<<".dat";
	ofstream out(name.str().c_str());

	for(k=1;k<NZ-1;k++)
		for(j=1;j<NY-1;j++)
			for(i=1;i<NX-1;i++)
			{
                out<<ux[i][j][k]<<" "<<uy[i][j][k]<<" "<<uz[i][j][k]<<" "<<rhoA[i][j][k]<<" "<<rhoB[i][j][k]<<" "<<vof[i][j][k]<<" "<<Pcr[i][j][k]<<" "<<solid[i][j][k]<<" "<<C[i][j][k]<<" "<<CA[i][j][k]<<" "<<CB[i][j][k]<<" "<<bounce[i][j][k]<<" "<<Pnu[i][j][k]<<endl;
                //out<<ux[i][j][k]<<" "<<uy[i][j][k]<<" "<<uz[i][j][k]<<" "<<rhoA[i][j][k]<<" "<<rhoB[i][j][k]<<" "<<vof[i][j][k]<<" "<<p[i][j][k]<<" "<<solid[i][j][k]<<endl;
			}
}

void Errorcheck()
{
    double temp1, temp2;
    temp1=0.0;
    temp2=0.0;
    for(i=1;i<NX-1;i++)
		for(j=1;j<NY-1;j++)
			for(k=1;k<NZ-1;k++)
			{
				if(solid[i][j][k]<1.0)
				{
					//temp1+=((ux[i][j][k]-ux0[i][j][k])*(ux[i][j][k]-ux0[i][j][k])+(uy[i][j][k]-uy0[i][j][k])*(uy[i][j][k]-uy0[i][j][k])+(uz[i][j][k]-uz0[i][j][k])*(uz[i][j][k]-uz0[i][j][k]));
  		            //temp2+=(ux[i][j][k]*ux[i][j][k]+uy[i][j][k]*uy[i][j][k]+uz[i][j][k]*uz[i][j][k]);

                    temp1+=(rhoA0[i][j][k]-rhoA[i][j][k])*(rhoA0[i][j][k]-rhoA[i][j][k]);
                    temp2+=rhoA0[i][j][k]*rhoA0[i][j][k];
				}
			}
			temp1=sqrt(temp1);
			temp2=sqrt(temp2);
			temp1=temp1/(temp2+1E-30);
    MPI_Reduce(&temp1,&error,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(mpirank==0)
    {
        cout<<"The"<<n<<"th computation result:"<<endl;
        cout<<"The max relative error of rhoA is:"<<setiosflags(ios::scientific)<<error<<endl;
    }
}

void Parameterg()
{
    Cinitial=pow(Ksp,1.0/2);
    Cinlet=1.0;
    Coutlet=1.0;

    DA=0.005;
    DB=0.005;
    H=100.0;

    kr0=0.00005;
    Ceq0=0.0;
    Vm=0.124;

    upd=0;

    C=new double**[NX];
    C0=new double**[NX];
    D=new double**[NX];
    kr=new double**[NX];
    Ceq=new double**[NX];
    CA=new double**[NX];
    CB=new double**[NX];
    CA0=new double**[NX];
    CB0=new double**[NX];
    SmC=new double**[NX];

    for (i=0;i<NX;i++)
    {
        C[i]=new double*[NY];
        C0[i]=new double*[NY];
        D[i]=new double*[NY];
        kr[i]=new double*[NY];
        Ceq[i]=new double*[NY];
        CA[i]=new double*[NY];
        CB[i]=new double*[NY];
        CA0[i]=new double*[NY];
        CB0[i]=new double*[NY];
        SmC[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            C[i][j]=new double[NZ];
            C0[i][j]=new double[NZ];
            D[i][j]=new double[NZ];
            kr[i][j]=new double[NZ];
            Ceq[i][j]=new double[NZ];
            CA[i][j]=new double[NZ];
            CB[i][j]=new double[NZ];
            CA0[i][j]=new double[NZ];
            CB0[i][j]=new double[NZ];
            SmC[i][j]=new double[NZ];
        }

    g=new double***[NX];
    gpost=new double***[NX];
    sg=new double***[NX];
    Scst=new double***[NX];

    for (i=0;i<NX;i++)
    {
        g[i]=new double**[NY];
        gpost[i]=new double**[NY];
        sg[i]=new double**[NY];
        Scst[i]=new double**[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            g[i][j]=new double*[NZ];
            gpost[i][j]=new double*[NZ];
            sg[i][j]=new double*[NZ];
            Scst[i][j]=new double*[NZ];
        }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
            for (k=0;k<NZ;k++)
            {
                g[i][j][k]=new double[QG];
                gpost[i][j][k]=new double[QG];
                sg[i][j][k]=new double[QG];
                Scst[i][j][k]=new double[QG];
            }

    sendg_right=new double[(NY-2)*(NZ-2)*QG];
    sendg_left=new double[(NY-2)*(NZ-2)*QG];
    sendg_front=new double[NX*(NZ-2)*QG];
    sendg_back=new double[NX*(NZ-2)*QG];
    sendg_up=new double[NX*NY*QG];
    sendg_down=new double[NX*NY*QG];
    recvg_right=new double[(NY-2)*(NZ-2)*QG];
    recvg_left=new double[(NY-2)*(NZ-2)*QG];
    recvg_front=new double[NX*(NZ-2)*QG];
    recvg_back=new double[NX*(NZ-2)*QG];
    recvg_up=new double[NX*NY*QG];
    recvg_down=new double[NX*NY*QG];
}

double geq(int l,double C,double ux,double uy,double uz)
{
    double eug, geq;
    eug=ex[l]*ux+ey[l]*uy+ez[l]*uz;
    geq=C*(J[l]+0.5*eug);
    return geq;
}

double mgeq(int l,double C,double ux,double uy,double uz)
{
    double mgeq;
    switch(l)
    {
        case 0:{mgeq=C;break;}
        case 1:{mgeq=C*ux;break;}
        case 2:{mgeq=C*uy;break;}
        case 3:{mgeq=C*uz;break;}
        case 4:{mgeq=3.0/4.0*C;break;}
        case 5:{mgeq=0.0;break;}
        case 6:{mgeq=0.0;break;}
        default:mgeq=0.0;
    }
    return mgeq;
}

void Initialg()
{
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(solid[i][j][k]<1.0)
                {
                    CA[i][j][k]=Cinitial;
                    CB[i][j][k]=Cinitial/H;
                    C[i][j][k]=vof[i][j][k]*CA[i][j][k]+(1.0-vof[i][j][k])*CB[i][j][k];
                    kr[i][j][k]=kr0;
                    Ceq[i][j][k]=Ceq0;
                    for(l=0;l<QG;l++)
                    {
                        g[i][j][k][l]=geq(l,C[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]);
                        Scst[i][j][k][l]=0.0;
                    }
                }
                else
                {
                    CA[i][j][k]=0.0;
                    CB[i][j][k]=0.0;
                    C[i][j][k]=vof[i][j][k]*CA[i][j][k]+(1.0-vof[i][j][k])*CB[i][j][k];
                    kr[i][j][k]=kr0;
                    Ceq[i][j][k]=Ceq0;
                    for(l=0;l<QG;l++)
                    {
                        g[i][j][k][l]=geq(l,C[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]);
                        Scst[i][j][k][l]=0.0;
                    }
                }
            }
    Dproperty();
}

void Dproperty()
{
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                D[i][j][k]=DA*DB/(vof[i][j][k]*DB+(1.0-vof[i][j][k])*DA);
                sg[i][j][k][0]=1.0;
                sg[i][j][k][1]=1.0/(4.0*D[i][j][k]+0.5);
                sg[i][j][k][2]=1.0/(4.0*D[i][j][k]+0.5);
                sg[i][j][k][3]=1.0/(4.0*D[i][j][k]+0.5);
                sg[i][j][k][4]=1.0;
                sg[i][j][k][5]=1.0;
                sg[i][j][k][6]=1.0;                
            }

    int nsnu;
    int ip,jp,kp;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                nsnu=0;
                if(solid[i][j][k]<1.0)
                {
                    for(l=1;l<QG;l++)
                    {
                        ip=i+ex[l];
                        jp=j+ey[l];
                        kp=k+ez[l];
                        if(solid[ip][jp][kp]<1.0)
                        {
                            nsnu=nsnu+1;
                        }
                        if(solid[ip][jp][kp]==1.0)
                        {
                            nsnu=0;
                            break;
                        }
                    }
                    chi[i][j][k]=nsnu*pow(solid[i][j][k],2.0/3);
                }
                else
                {
                    chi[i][j][k]=0.0;
                }
            }
}

void CSTsource()
{
    double coeff;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                dvofx[i][j][k]=0.0;
                dvofy[i][j][k]=0.0;
                dvofz[i][j][k]=0.0;
                for(l=0;l<Q;l++)
                {
                    dvofx[i][j][k]+=3.0*w[l]*ex[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                    dvofy[i][j][k]+=3.0*w[l]*ey[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                    dvofz[i][j][k]+=3.0*w[l]*ez[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                }
                coeff=C[i][j][k]*(H-1.0)/(H*vof[i][j][k]+(1.0-vof[i][j][k]));
                Scst[i][j][k][0]=0.0;
                Scst[i][j][k][1]=(1.0-0.5*sg[i][j][k][1])/4.0*coeff*dvofx[i][j][k];
                Scst[i][j][k][2]=(1.0-0.5*sg[i][j][k][2])/4.0*coeff*dvofy[i][j][k];
                Scst[i][j][k][3]=(1.0-0.5*sg[i][j][k][3])/4.0*coeff*dvofz[i][j][k];
                Scst[i][j][k][4]=0.0;
                Scst[i][j][k][5]=0.0;
                Scst[i][j][k][6]=0.0;
            }
}

void Collisiong()
{
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]<1.0)
                {
                    mg[0]=g[i][j][k][0]+g[i][j][k][1]+g[i][j][k][2]+g[i][j][k][3]+g[i][j][k][4]+g[i][j][k][5]+g[i][j][k][6];
                    mg[1]=g[i][j][k][1]-g[i][j][k][2];
                    mg[2]=g[i][j][k][3]-g[i][j][k][4];
                    mg[3]=g[i][j][k][5]-g[i][j][k][6];
                    mg[4]=6.0*g[i][j][k][0]-g[i][j][k][1]-g[i][j][k][2]-g[i][j][k][3]-g[i][j][k][4]-g[i][j][k][5]-g[i][j][k][6];
                    mg[5]=2.0*g[i][j][k][1]+2.0*g[i][j][k][2]-g[i][j][k][3]-g[i][j][k][4]-g[i][j][k][5]-g[i][j][k][6];
                    mg[6]=g[i][j][k][3]+g[i][j][k][4]-g[i][j][k][5]-g[i][j][k][6];
                    for(l=0;l<QG;l++)
                    {
                        Mg[l]=mg[l]-sg[i][j][k][l]*(mg[l]-mgeq(l,C[i][j][k],ux[i][j][k]*0.0,uy[i][j][k]*0.0,uz[i][j][k]*0.0))+Scst[i][j][k][l];
                    }
                    gpost[i][j][k][0]=Mg[0]/7.0+Mg[4]/7.0;
                    gpost[i][j][k][1]=Mg[0]/7.0+Mg[1]/2.0-Mg[4]/42.0+Mg[5]/6.0;
                    gpost[i][j][k][2]=Mg[0]/7.0-Mg[1]/2.0-Mg[4]/42.0+Mg[5]/6.0;
                    gpost[i][j][k][3]=Mg[0]/7.0+Mg[2]/2.0-Mg[4]/42.0-Mg[5]/12.0+Mg[6]/4.0;
                    gpost[i][j][k][4]=Mg[0]/7.0-Mg[2]/2.0-Mg[4]/42.0-Mg[5]/12.0+Mg[6]/4.0;
                    gpost[i][j][k][5]=Mg[0]/7.0+Mg[3]/2.0-Mg[4]/42.0-Mg[5]/12.0-Mg[6]/4.0;
                    gpost[i][j][k][6]=Mg[0]/7.0-Mg[3]/2.0-Mg[4]/42.0-Mg[5]/12.0-Mg[6]/4.0;
                }
            }
}

void Infosendrecvg()
{
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
            for(l=0;l<QG;l++)
            {
                sendg_right[l+(j-1+(k-1)*(NY-2))*QG]=gpost[NX-2][j][k][l];
                sendg_left[l+(j-1+(k-1)*(NY-2))*QG]=gpost[1][j][k][l];
            }
    MPI_Sendrecv(&sendg_right[0],(NY-2)*(NZ-2)*QG,MPI_DOUBLE,destright,tagrg,
                    &recvg_left[0],(NY-2)*(NZ-2)*QG,MPI_DOUBLE,destleft,tagrg,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendg_left[0],(NY-2)*(NZ-2)*QG,MPI_DOUBLE,destleft,taglg,
                    &recvg_right[0],(NY-2)*(NZ-2)*QG,MPI_DOUBLE,destright,taglg,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
            for(l=0;l<QG;l++)
            {
                gpost[NX-1][j][k][l]=recvg_right[l+(j-1+(k-1)*(NY-2))*QG];
                gpost[0][j][k][l]=recvg_left[l+(j-1+(k-1)*(NY-2))*QG];
            }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
            for(l=0;l<QG;l++)
            {
                sendg_front[l+(i+(k-1)*(NX))*QG]=gpost[i][NY-2][k][l];
                sendg_back[l+(i+(k-1)*(NX))*QG]=gpost[i][1][k][l];
            }
    MPI_Sendrecv(&sendg_front[0],(NX)*(NZ-2)*QG,MPI_DOUBLE,destfront,tagfg,
                    &recvg_back[0],(NX)*(NZ-2)*QG,MPI_DOUBLE,destback,tagfg,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendg_back[0],(NX)*(NZ-2)*QG,MPI_DOUBLE,destback,tagbg,
                    &recvg_front[0],(NX)*(NZ-2)*QG,MPI_DOUBLE,destfront,tagbg,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
            for(l=0;l<QG;l++)
            {
                gpost[i][NY-1][k][l]=recvg_front[l+(i+(k-1)*(NX))*QG];
                gpost[i][0][k][l]=recvg_back[l+(i+(k-1)*(NX))*QG];
            }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
            for(l=0;l<QG;l++)
            {
                sendg_up[l+(i+j*(NX))*QG]=gpost[i][j][NZ-2][l];
                sendg_down[l+(i+j*(NX))*QG]=gpost[i][j][1][l];
            }
    MPI_Sendrecv(&sendg_up[0],(NX)*(NY)*QG,MPI_DOUBLE,destup,tagug,
                    &recvg_down[0],(NX)*(NY)*QG,MPI_DOUBLE,destdown,tagug,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendg_down[0],(NX)*(NY)*QG,MPI_DOUBLE,destdown,tagdg,
                    &recvg_up[0],(NX)*(NY)*QG,MPI_DOUBLE,destup,tagdg,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
            for(l=0;l<QG;l++)
            {
                gpost[i][j][NZ-1][l]=recvg_up[l+(i+j*(NX))*QG];
                gpost[i][j][0][l]=recvg_down[l+(i+j*(NX))*QG];
            }
    MPI_Barrier(MPI_COMM_WORLD);
}

void Boundaryg()
{
    int ip,jp,kp;
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(bounce[i][j][k]==1.0&&solid[i][j][k]==1.0)
                {
                    for(l=0;l<QG;l++)
                    {
                        ip=min(NX-1,max((i+ex[l]),0));
                        jp=min(NY-1,max((j+ey[l]),0));
                        kp=min(NZ-1,max((k+ez[l]),0));
                        gpost[i][j][k][l]=gpost[ip][jp][kp][op[l]]-max(0.0,vof[i][j][k]*Krnu[i][j][k]*(C[i][j][k]*C[i][j][k]/Ksp-1.0));
                    }
                }
                if(bounce[i][j][k]==1.0&&solid[i][j][k]>1.0)
                {
                    for(l=0;l<QG;l++)
                    {
                        ip=min(NX-1,max((i+ex[l]),0));
                        jp=min(NY-1,max((j+ey[l]),0));
                        kp=min(NZ-1,max((k+ez[l]),0));
                        gpost[i][j][k][l]=gpost[ip][jp][kp][op[l]]-max(0.0,vof[i][j][k]*chi[i][j][k]*Krnu[i][j][k]*(C[i][j][k]*C[i][j][k]/Ksp-1.0));
                    }
                }

            }
}

void Boundaryg2()
{
    if(rankx==0)
    {
        Inletg();
    }
    /*if(rankx==PX-1)
    {
        Outletg();
    }*/

    int ip,jp,kp;
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(bounce[i][j][k]==1.0&&solid[i][j][k]==1.0)
                {
                    for(l=0;l<QG;l++)
                    {
                        ip=min(NX-1,max((i+ex[l]),0));
                        jp=min(NY-1,max((j+ey[l]),0));
                        kp=min(NZ-1,max((k+ez[l]),0));
                        gpost[i][j][k][l]=gpost[ip][jp][kp][op[l]]-max(0.0,vof[ip][jp][kp]*Krnu[ip][jp][kp]*(max(0.0,C[ip][jp][kp])*max(0.0,C[ip][jp][kp])/Ksp-1.0));
                    }
                }
                if(bounce[i][j][k]==1.0&&solid[i][j][k]>1.0)
                {
                    for(l=0;l<QG;l++)
                    {
                        ip=min(NX-1,max((i+ex[l]),0));
                        jp=min(NY-1,max((j+ey[l]),0));
                        kp=min(NZ-1,max((k+ez[l]),0));
                        gpost[i][j][k][l]=gpost[ip][jp][kp][op[l]]-max(0.0,vof[ip][jp][kp]*chi[ip][jp][kp]*Krnu[ip][jp][kp]*(max(0.0,C[ip][jp][kp])*max(0.0,C[ip][jp][kp])/Ksp-1.0));
                    }
                }

            }
}

void Inletg()
{
    int NXC=10;
    double Cw;
    i=NXC;
    for(j=0;j<NY;j++)
        for(k=0;k<NZ;k++)
            {                
                gpost[i][j][k][1]=gpost[i+1][j][k][2]+C[i+1][j][k]*ux[i+1][j][k]*0.0;
            }
}

void Outletg()
{
    int NXC=10;
    double Cw;
    i=NX-NXC-1;
    for(j=0;j<NY;j++)
        for(k=0;k<NZ;k++)
            {                
                Cw=C[i-1][j][k];
                gpost[i][j][k][2]=Cw/4.0-gpost[i-2][j][k][1];
            }
}

void Streamingg()
{
    int iq, jq, kq;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]<1.0)
                {
                    for(l=0;l<QG;l++)
                    {
                        iq=i-ex[l];
                        jq=j-ey[l];
                        kq=k-ez[l];
                        g[i][j][k][l]=gpost[iq][jq][kq][l];
                    }
                }
            }
}

void Macrog()//i=0:NX-1
{
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(solid[i][j][k]<1.0)
                {
                    C0[i][j][k]=C[i][j][k];
                    C[i][j][k]=0.0;
                    for(l=0;l<QG;l++)
                    {
                        C[i][j][k]+=g[i][j][k][l];
                    }
                }
            }
}

void InfosendrecvC()
{
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            send_right[j-1+(k-1)*(NY-2)]=C[NX-2][j][k];
            send_left[j-1+(k-1)*(NY-2)]=C[1][j][k];
        }
    MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrC,
                    &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrC,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,taglC,
                    &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,taglC,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            C[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
            C[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            send_front[i+(k-1)*(NX)]=C[i][NY-2][k];
            send_back[i+(k-1)*(NX)]=C[i][1][k];
        }
    MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfC,
                    &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfC,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbC,
                    &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbC,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            C[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
            C[i][0][k]=recv_back[i+(k-1)*(NX)];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            send_up[i+j*(NX)]=C[i][j][NZ-2];
            send_down[i+j*(NX)]=C[i][j][1];
        }
    MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,taguC,
                    &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,taguC,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagdC,
                    &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagdC,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            C[i][j][NZ-1]=recv_up[i+j*(NX)];
            C[i][j][0]=recv_down[i+j*(NX)];
        }
    MPI_Barrier(MPI_COMM_WORLD);
}


void Infosendrecvsolid()
{
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            send_right[j-1+(k-1)*(NY-2)]=solid[NX-2][j][k];
            send_left[j-1+(k-1)*(NY-2)]=solid[1][j][k];
        }
    MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrS,
                    &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrS,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,taglS,
                    &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,taglS,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            solid[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
            solid[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            send_front[i+(k-1)*(NX)]=solid[i][NY-2][k];
            send_back[i+(k-1)*(NX)]=solid[i][1][k];
        }
    MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfS,
                    &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfS,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbS,
                    &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbS,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            solid[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
            solid[i][0][k]=recv_back[i+(k-1)*(NX)];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            send_up[i+j*(NX)]=solid[i][j][NZ-2];
            send_down[i+j*(NX)]=solid[i][j][1];
        }
    MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,taguS,
                    &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,taguS,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagdS,
                    &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagdS,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            solid[i][j][NZ-1]=recv_up[i+j*(NX)];
            solid[i][j][0]=recv_down[i+j*(NX)];
        }

    if(rankx==PX-1)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                solid[NX-1][j][k]=2.0;
            }
    }
    if(rankx==0)
    {
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                solid[0][j][k]=solid[1][j][k];
            }
    }
}

void Kinetic()
{}

void Getcacb()
{
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                CA[i][j][k]=H*C[i][j][k]/(H*vof[i][j][k]+1.0-vof[i][j][k]);
                CB[i][j][k]=C[i][j][k]/(H*vof[i][j][k]+1.0-vof[i][j][k]);
            }
}

void Errorcheckg()
{
    double temp1, temp2;
    temp1=0.0;
    temp2=0.0;
    for(i=1;i<NX-1;i++)
		for(j=1;j<NY-1;j++)
			for(k=1;k<NZ-1;k++)
			{
				if(solid[i][j][k]<1.0)
				{
					//temp1+=((ux[i][j][k]-ux0[i][j][k])*(ux[i][j][k]-ux0[i][j][k])+(uy[i][j][k]-uy0[i][j][k])*(uy[i][j][k]-uy0[i][j][k])+(uz[i][j][k]-uz0[i][j][k])*(uz[i][j][k]-uz0[i][j][k]));
  		            //temp2+=(ux[i][j][k]*ux[i][j][k]+uy[i][j][k]*uy[i][j][k]+uz[i][j][k]*uz[i][j][k]);

                    temp1+=(C0[i][j][k]-C[i][j][k])*(C0[i][j][k]-C[i][j][k]);
                    temp2+=C0[i][j][k]*C0[i][j][k];
				}
			}
			temp1=sqrt(temp1);
			temp2=sqrt(temp2);
			temp1=temp1/(temp2+1E-30);
    MPI_Reduce(&temp1,&errorC,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(mpirank==0)
    {
        cout<<"The max relative error of C is:"<<setiosflags(ios::scientific)<<errorC<<endl;
    }
}

void Parameternu()
{
    vmnu=3.69e-5;
    gammanu=0.04;
    kBnu=1.38e-23;
    NAnu=6.02e23;
    Tnu=298.15;
    J0nu=1.0e8;
    tlim=100;
    rnu=0.01;
    Ksp=3.31e1;
    //double ***Pnu, ***Pcr;
    //double ***chi, ***Ceqnu, ***Krnu;
    Pnu=new double**[NX];
    Pcr=new double**[NX];
    chi=new double**[NX];
    Ceqnu=new double**[NX];
    Krnu=new double**[NX];
    SmA=new double**[NX];
    SmB=new double**[NX];

    for (i=0;i<NX;i++)
    {
        Pnu[i]=new double*[NY];
        Pcr[i]=new double*[NY];
        chi[i]=new double*[NY];
        Ceqnu[i]=new double*[NY];
        Krnu[i]=new double*[NY];
        SmA[i]=new double*[NY];
        SmB[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            Pnu[i][j]=new double[NZ];
            Pcr[i][j]=new double[NZ];
            chi[i][j]=new double[NZ];
            Ceqnu[i][j]=new double[NZ];
            Krnu[i][j]=new double[NZ];
            SmA[i][j]=new double[NZ];
            SmB[i][j]=new double[NZ];
        }
}

void Initialnu()
{
    double pmax=1.5;
    double pmin=0.5;
    srand(time(NULL));
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                Pnu[i][j][k]=0.0;
                Pcr[i][j][k]=(rand()%1000/1000.0)*(pmax-pmin)+pmin;
                chi[i][j][k]=0.0;
                Ceqnu[i][j][k]=Ksp;
                Krnu[i][j][k]=3.81e-5;
                SmA[i][j][k]=0.0;
                SmB[i][j][k]=0.0;
            }
}

void Nuclear()
{
    int ip, jp, kp;
    tnu=n%tlim;
    if(tnu==0)
    {
        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    if(solid[i][j][k]==0.0&&Pnu[i][j][k]>Pcr[i][j][k])
                    {
                        solid[i][j][k]=rnu*rnu*rnu;
                    }
                    Pnu[i][j][k]=0.0;
                }
    }
    else
    {
        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    if(solid[i][j][k]==0.0)
                    {
                        for(l=1;l<QG;l++)
                        {
                            ip=i+ex[l];
                            jp=j+ey[l];
                            kp=k+ez[l];
                            if(solid[ip][jp][kp]==2.0)
                            {
                                if(C[i][j][k]*C[i][j][k]>Ksp)
                                {
                                    Jnu=vof[i][j][k]*J0nu*exp(-16.0*M_PI*vmnu*vmnu*gammanu*gammanu*gammanu/3.0/NAnu/NAnu/kBnu/kBnu/kBnu/Tnu/Tnu/Tnu/log(C[i][j][k]*C[i][j][k]/Ksp)/log(C[i][j][k]*C[i][j][k]/Ksp));
                                }
                                else
                                {
                                    Jnu=0.0;
                                }
                                Pnu[i][j][k]=Pnu[i][j][k]+Jnu*exp(-Jnu*(tnu-1));
                            }
                        }
                    }
                }
    }
}

void Nuclearevolution()
{
    int ip,jp,kp;
    int temp=0;
    int temp2=0,temp3=0;
    MPI_Status status;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]<1.0&&vof[i][j][k]>0.5)
                {
                    for(l=1;l<QG;l++)
                    {
                        ip=i+ex[l];
                        jp=j+ey[l];
                        kp=k+ez[l];
                        if(solid[ip][jp][kp]==2.0)
                        {
                            solid[i][j][k]=solid[i][j][k]+max(0.0,vof[i][j][k]*chi[i][j][k]*Krnu[i][j][k]*(max(0.0,C[i][j][k])*max(0.0,C[i][j][k])/Ksp-1.0))*vmnu*1000.0;
                            SmC[i][j][k]=max(0.0,vof[i][j][k]*chi[i][j][k]*Krnu[i][j][k]*(max(0.0,C[i][j][k])*max(0.0,C[i][j][k])/Ksp-1.0));
                        }
                        if(solid[ip][jp][kp]==1.0)
                        {
                            solid[i][j][k]=solid[i][j][k]+max(0.0,vof[i][j][k]*Krnu[i][j][k]*(max(0.0,C[i][j][k])*max(0.0,C[i][j][k])/Ksp-1.0))*vmnu*1000.0;
                        }
                    }
                    if(solid[i][j][k]>=1.0)
                    {
                        temp=1;
                    }
                }
                if(solid[i][j][k]<0.0)
                {
                    solid[i][j][k]=0.0;
                }
                if(solid[i][j][k]>2.0)
                {
                    solid[i][j][k]=1.0;
                }  
            }

            MPI_Reduce(&temp,&temp2,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);

            if(mpirank==0)
            {
                for(i=1;i<mpisize;i++)
                {
                    MPI_Send(&temp2,1,MPI_INT,i,8881,MPI_COMM_WORLD);
                }
                temp3=temp2;
            }
            else
            {
                MPI_Recv(&temp3,1,MPI_INT,0,8881,MPI_COMM_WORLD,&status);
            }
                    upd=temp3;
                    
}

void Nuclearupdate()
{
    double averhoA, averhoB, aveC;
    int aven;
    int ip,jp,kp;

    for(i=0;i<=NX-1;i++)
        for(j=0;j<=NY-1;j++)
            for(k=0;k<=NZ-1;k++)
                {
                    SmA[i][j][k]=0.0;
                    SmB[i][j][k]=0.0;
                }

    if(upd==1)
    {
        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    if(solid[i][j][k]>=1.0&&solid[i][j][k]<2.0)
                    {
                        solid[i][j][k]=1.0;
                        ux[i][j][k]=0.0;
                        uy[i][j][k]=0.0;
                        uz[i][j][k]=0.0;
                        C[i][j][k]=0.0;
                        aven=0;

                        for(l=1;l<Q;l++)
                        {
                            ip=i+ex[l];
                            jp=j+ey[l];
                            kp=k+ez[l];
                            aven=0;

                            if(solid[ip][jp][kp]<1.0&&ip>0&&ip<NX-1&&jp>0&&jp<NY-1&&kp>0&&kp<NZ-1)
                            {
                                aven=aven+w[l];
                            }
                        }

                        if(aven>0)
                        {
                        for(l=1;l<Q;l++)
                        {
                            ip=i+ex[l];
                            jp=j+ey[l];
                            kp=k+ez[l];

                            if(solid[ip][jp][kp]<1.0&&ip>0&&ip<NX-1&&jp>0&&jp<NY-1&&kp>0&&kp<NZ-1)
                            {
                                SmA[ip][jp][kp]=SmA[ip][jp][kp]+w[l]/aven*rhoA[i][j][k];
                                SmB[ip][jp][kp]=SmB[ip][jp][kp]+w[l]/aven*rhoB[i][j][k];
                            }
                        }
                        }
                    }
                }
                //Infosendrecvmacro();
                InfosendrecvC();
                Infosendrecvsolid();
                /*if(rankx==0)
                {
                    Inlet();
                }
                if(rankx==PX-1)
                {
                    Outlet();
                }*/
                Getbounce();
                Phisolid2();                
    }
    upd=0;
}