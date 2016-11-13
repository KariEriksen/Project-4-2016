#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <armadillo>
#include <string>
#include "mpi.h"
using namespace  std;
using namespace arma;


void InitializeGrid(int N, mat& Lattice);
void MeasureEnergy (double& E, mat& Lattice, int N);
void Metropolis(double T,int N, int MCC);
void WriteToFile(double Values1, double Values2, double Values3, double Values4);
void MeasureMagnetization(double& sumM, mat& Lattice);

int main(int argc, char* argv[]) {

    fstream outFile;
    outFile.open("dataExp40.dat", ios::out);
    outFile.close();

    int N = 20;      //size of lattice NxN
    int MCC = 1e4;   //number of Monte Carlo cycles
    int M = 7;       //number of different temperatures
    double T = 2.05; //initial temperature
    int numprocs, my_rank;

    //Initialization of MPI
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);


    // broadcast to all nodes common variables
    MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast (&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&M, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Bcast (&temp_step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double TimeStart, TimeEnd, TotalTime;
    TimeStart = MPI_Wtime();

    for (int i=0; i < M; ++i){

        //T = 2.4;
        T += 0.05;
        Metropolis(T, N, MCC);
    }
    TimeEnd = MPI_Wtime();
    TotalTime = TimeEnd-TimeStart;
    cout << TotalTime << endl;
    MPI_Finalize ();
}

void Metropolis(double T,int N, int MCC){

    mat Lattice = zeros(N, N);
    double Ecurrent = 0;
    double Enew = 0;
    double Mcurrent = 0;
    double sumE = 0;
    double sumEsq = 0;
    double sumM = 0;
    double sumMsq = 0;
    int counter;

    InitializeGrid(N, Lattice);
    //First measure Energy before cycle
    MeasureEnergy(Ecurrent, Lattice, N);
    sumE += Ecurrent;
    sumEsq += Ecurrent*Ecurrent;
    //initial value for magnatization
    MeasureMagnetization(Mcurrent, Lattice);

    //choose random number
    std::random_device rd;
    std::mt19937_64 gen(rd());
    //gen.seed(190);
    std::uniform_real_distribution<double> RanNum(0,1);

    for (int k=1; k < MCC; ++k){
        for (int l=0; l < N; ++l){
            for (int m=0; m < N; ++m){

                //Flip random spin
                int spin_x = floor(RanNum(gen)*N);
                int spin_y = floor(RanNum(gen)*N);
                Lattice(spin_x, spin_y) *= -1;

                MeasureEnergy(Enew, Lattice, N);
                MeasureMagnetization(Mcurrent, Lattice);
                double deltaE = Enew - Ecurrent;

                if (deltaE < 0){

                    // Accept new configuration
                    Ecurrent = Enew;
                    counter += 1;
                }

                else {

                    //k = 1;
                    double omega = exp(-deltaE/T);

                    double r = RanNum(gen);
                    if(r <= omega){
                        // Accept new configuration
                        Ecurrent = Enew;
                        counter += 1;
                    }

                    else {
                        // Reject new config, flip spin back again
                        Lattice(spin_x, spin_y) *= -1;
                        MeasureMagnetization(Mcurrent, Lattice);
                    }
                }
            }
        }
        sumM += Mcurrent;
        sumMsq += Mcurrent*Mcurrent;

        sumE += Ecurrent;
        sumEsq += Ecurrent*Ecurrent;
        //WriteToFile(Mcurrent, Ecurrent);
        //WriteToFile(k, counter);

    }
    MPI_Reduce(&sumE, &sumM, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    sumE = sumE/((double)MCC);
    sumM = sumM/((double)MCC);
    sumEsq = sumEsq/((double)MCC);
    sumMsq = sumMsq/((double)MCC);
    double varE = sumEsq - sumE*sumE;
    double varM = sumMsq - sumM*sumM;
    WriteToFile(sumE, sumM, varE, varM);
    //WriteToFile(varE, varM);
    //cout << varE << endl;
    //cout << T << ' ' << varE << ' ' << varM << endl;
}

void InitializeGrid(int N, mat& Lattice){

    //choose random number
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> RanNum(0,1);

    for (int k=0; k < N; ++k){
        for (int l=0; l < N; ++l){
            //Lattice(k,l) = 1;
            // A = ( if test ) ? (hvis if test er ja, sett A til dette) : (hvis if test er nei, sett A til dette);
            Lattice(k,l) = (RanNum(gen) > 0.5) ? -1 : 1;
        }
    }
    //cout << Lattice << endl;
}

void MeasureEnergy(double& E, mat& matrix, int N){

    //double J = -1;
    E = 0;
    for (int i=0; i < N; ++i){
        for (int j=0; j < N; ++j){

            int iPrev = i-1;
            int jPrev = j-1;

            if (iPrev == -1) iPrev = N-1;
            if (jPrev == -1) jPrev = N-1;
            //cout << "HEI " << matrix(i,j) << "; " << matrix(iNext, j) << ";" << matrix(iPrev,j) << "," << matrix(i,jNext) << ";" << matrix(i,jPrev) << endl;
            E -= matrix(i,j) * (matrix(iPrev,j) + matrix(i,jPrev));
        }
    }
    //cout << matrix << endl;
}

void MeasureMagnetization(double& sumM, mat& Lattice){

    sumM = abs(accu(Lattice));
    //sumM = accu(Lattice);
}

void WriteToFile(double Values1, double Values2, double Values3, double Values4){

    fstream outFile;
    outFile.open("dataExp40.dat", ios::app);

    outFile << Values1 << ' ' << Values2 << ' ' << Values3 << ' ' << Values4 <<endl;

    outFile.close();
    //return 0;
}

