#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void input(double t[], double RV[], FILE *fp, int row) {
    char line[256];  // 一行の最大長（固定推奨）
    int i = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (line[0] == '#') continue;
        int modori = sscanf(line, " %lf %lf", &t[i], &RV[i]);
        if (modori != 2) {
            fprintf(stderr, "Error! Failed to read line: %s\n", line);
            continue;  // この行は飛ばして次へ
        }
        i++;
        if (i >= row) break;  // 念のため配列越えを防ぐ
    }
}

double mean_anomaly(double P, double t, double T0) {
    return fmod(2*M_PI/P * (t-T0), 2.0 * M_PI);
}

double solve_kepler(double M, double e) {
    double E = M;
    for (int i = 0; i < 100; ++i) {
        double f = E - e * sin(E) - M;
        double f_prime = 1 - e * cos(E);
        E = E - f / f_prime;
    }
    return E;
}

double true_anomaly(double E, double e) {
    double sqrt_arg = sqrt((1 + e) / (1 - e));
    double tan_half_nu = sqrt_arg * tan(E / 2.0);
    return 2.0 * atan(tan_half_nu);
}



// double RV_equation(double Vsys, double Kp, double chi,double t,double omega, double e){
//     return Vsys+Kp*(cos(chi*t+omega)+e*cos(omega));
// }
double RV_equation(double Vsys, double Kp, double P, double t, double omega, double e, double T0) {
    double M = mean_anomaly(P, t, T0);
    double E = solve_kepler(M, e);
    double f = true_anomaly(E, e);
    return Vsys + Kp * (cos(f + omega) + e * cos(omega));
}

double log_Yudo(double Vsys, double Kp, double P, double omega, double e, double t[], double RV[], int row, double sigma, double T0, double b){
    double logL = 0.0;
    for (int i = 0; i < row; ++i) {
        double model = RV_equation(Vsys,Kp,P,t[i],omega,e,T0);
        double resid = RV[i] - model;
        logL += 0.5 * (resid * resid) / (sigma * sigma) * b - log(b)*0.5;
    }
    logL += log(P)+log(Kp)+log(b);//Jeffreysの事前分布
    return logL;
}

void vorbereiten(double *A, double *backup_A, double *dA, double step_size_A){
    *backup_A = *A;
    *dA = (double)rand()/RAND_MAX;
    *dA=(*dA-0.5e0)*step_size_A*2e0;
    *A=*A+*dA;
    return;
}

int main(void){
    int row=30;
    double *t,*RV;
    t=malloc(sizeof(double)*row);
    RV=malloc(sizeof(double)*row);

    int niter=100000;
    double step_size_Vsys=0.2e0;
    double step_size_Kp=0.5e0;
    double step_size_P=10.0e0;
    double step_size_omega=0.02e0;
    double step_size_e=0.02e0;
    double step_size_b=0.2e0;
    double step_size_T0=0.5e0;
    double sigma=1.0e0;

    FILE *fp,*fq;
    fp=fopen("RV_data.txt","r");
    fq=fopen("RV.txt","w");
    srand((unsigned)time(NULL));
    input(t,RV,fp,row);

    // 初期値
    double Vsys=-0.2;
    double Kp=11.3;
    double P=365.0;
    double omega=0.34;
    double e=0.47;
    double T0=-2.4;
    double b=1.0;//noise scale parameter
    int naccept=0;//更新提案の受理回数

    // 乱数を加える
    for(int iter=1;iter<niter+1;iter++){
        double backup_Vsys ,backup_Kp,backup_P, backup_omega, backup_e,backup_b,backup_T0;
        double dVsys ,dKp,dP, domega, de,db,dT0;
        double action_init,action_fin;

        //Vsys
        action_init=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        vorbereiten(&Vsys,&backup_Vsys,&dVsys,step_size_Vsys);
        action_fin=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        // メトロポリス
        double metropolis_Vsys = (double)rand()/RAND_MAX;
        if(exp(action_init-action_fin)>metropolis_Vsys){
            naccept=naccept+1;//受理
        }else{
            Vsys=backup_Vsys;//棄却
        }

        //Kp
        action_init=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        vorbereiten(&Kp,&backup_Kp,&dKp,step_size_Kp);
        action_fin=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        // メトロポリス
        double metropolis_Kp = (double)rand()/RAND_MAX;
        if(exp(action_init-action_fin)>metropolis_Kp){
            naccept=naccept+1;//受理
        }else{
            Kp=backup_Kp;//棄却
        }

        //P
        action_init=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        vorbereiten(&P,&backup_P,&dP,step_size_P);
        action_fin=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        // メトロポリス
        double metropolis_P = (double)rand()/RAND_MAX;
        if(exp(action_init-action_fin)>metropolis_P){
            naccept=naccept+1;//受理
        }else{
            P=backup_P;//棄却
        }        

        //omega
        action_init=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        vorbereiten(&omega,&backup_omega,&domega,step_size_omega);
        action_fin=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        // メトロポリス
        double metropolis_omega = (double)rand()/RAND_MAX;
        if(exp(action_init-action_fin)>metropolis_omega){
            naccept=naccept+1;//受理
        }else{
            omega=backup_omega;//棄却
        }

        //e
        action_init=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        vorbereiten(&e,&backup_e,&de,step_size_e);
        // メトロポリス
        double metropolis_e = (double)rand()/RAND_MAX;
        if (e <= 0.0 || e >= 1.0) {
            e = backup_e; // 棄却
        } else {
            action_fin=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
            if(exp(action_init-action_fin)>metropolis_e){
                naccept=naccept+1;//受理
            }else{
                e=backup_e;//棄却
            }
        }

        //b
        action_init=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        vorbereiten(&b,&backup_b,&db,step_size_b);
        // メトロポリス
        double metropolis_b = (double)rand()/RAND_MAX;
        if (b <= 0.0) {
            b = backup_b; // 棄却
        } else {
            action_fin=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
            if(exp(action_init-action_fin)>metropolis_b){
                naccept=naccept+1;//受理
            }else{
                b=backup_b;//棄却
            }
        }

        //T0
        action_init=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        vorbereiten(&T0,&backup_T0,&dT0,step_size_T0);
        // メトロポリス
        double metropolis_T0 = (double)rand()/RAND_MAX;
        action_fin=log_Yudo(Vsys,Kp,P,omega,e,t,RV,row,sigma,T0,b);
        if(exp(action_init-action_fin)>metropolis_T0){
            naccept=naccept+1;//受理
        }else{
            T0=backup_T0;//棄却
        }

        if(iter%10000==0)printf("%d\n",iter);
        // 出力
        fprintf(fq,"%.10f %.10f %.10f %.10f %.10f %.10f  %.10f %f %d\n",Vsys,Kp,P,omega,e,b,T0,(double)naccept/iter,iter);
    }
    fclose(fp);
    fclose(fq);
    return 0;
}