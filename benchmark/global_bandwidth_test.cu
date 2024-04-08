#include "cuda.h"
#include "cuda_runtime.h"
#include "chrono"
#include <cstdlib>
#include <iostream>
using namespace std;
using namespace chrono;

extern "C" __global__ void global_bandwidth_test(double* data, double* cp, int n){
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int range = blockDim.x*gridDim.x;
    for(int i=x;i<n;i+=range){
        cp[i] = data[i];
    }
}

extern "C" __global__ void global_bandwidth_extra_cost_test(double* data, double* cp, int n){
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int range = blockDim.x*gridDim.x;
    for(int i=x;i<n;i+=range){
        
    }
}

int main(){
    // double* h_data = (double*)malloc(sizeof(double)*4096*4096);
    double* d_data, *d_cp;
    cudaMalloc(&d_data, sizeof(double)*4096*4096);
    cudaMalloc(&d_cp, sizeof(double)*4096*4096);
    cudaDeviceSynchronize();
    auto start = system_clock::now();
    global_bandwidth_test<<<4096*4096/64/256, 256>>>(d_data, d_cp, 4096*4096);
    cudaDeviceSynchronize();
    // cudaError_t error = cudaGetLastError();
    // printf("%s\n", cudaGetErrorString(error));
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    start = system_clock::now();
    global_bandwidth_extra_cost_test<<<4096*4096/128/256, 256>>>(d_data, d_cp, 4096*4096);
    cudaDeviceSynchronize();
    end = system_clock::now();
    auto duration_extra_cost = duration_cast<microseconds>(end - start);
    cout<<duration.count()<<endl;
    cout<<duration_extra_cost.count()<<endl;
    double bandwidth = 4096*4096*2*8/(double(duration.count())-double(duration_extra_cost.count()))/1000;
    cout <<  "global bandwidth: " 
     << bandwidth
     << "GB/s" << endl;
}