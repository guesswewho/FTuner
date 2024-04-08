#include "cuda.h"
#include "cuda_runtime.h"
#include "chrono"
#include <cstdlib>
#include <iostream>
using namespace std;
using namespace chrono;

extern "C" __global__ void shared_bandwidth_test(double* data, double* cp, int n){
    __shared__ double cache[32];
    __shared__ double cache_cp[32];
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int range = blockDim.x*gridDim.x;
    int cache_r = threadIdx.x%32;
    int y;
    for(int i=x;i<n;i+=range){
        cache[cache_r] = data[i];
        for(int j=0;j<1000;j++){
          cache_r = (cache_r+1) % 32;
          cache[cache_r] = cache_cp[cache_r];
          cache_r = (cache_r+1) % 32;
          __syncthreads();
          cache[cache_r] = cache_cp[cache_r];
          // __syncthreads();
          // cache_r = (cache_r+1) % 32;
          // cache[cache_r] = cache_cp[cache_r];
          // __syncthreads();
          // cache_r = (cache_r+1) % 32;
          // cache[cache_r] = cache_cp[cache_r];
        }
        cp[i] = cache[cache_r];
        cp[i] = y;
    }

}

extern "C" __global__ void shared_bandwidth_extra_cost_test(double* data, double* cp, int n){
    __shared__ double cache[32];
    __shared__ double cache_cp[32];
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int range = blockDim.x*gridDim.x;
    int cache_r = threadIdx.x%32;
    int y;
    for(int i=x;i<n;i+=range){
        cache[cache_r] = data[i];
        for(int j=0;j<1000;j++){
          cache_r = (cache_r+1) % 32;
          cache[cache_r] = cache_cp[cache_r];
          cache_r = (cache_r+1) % 32;
          __syncthreads();
          // cache[cache_r] = cache_cp[cache_r];
          // __syncthreads();
          // cache_r = (cache_r+1) % 32;
          // cache[cache_r] = cache_cp[cache_r];
          // __syncthreads();
          // cache_r = (cache_r+1) % 32;
          // cache[cache_r] = cache_cp[cache_r];
        }
        cp[i] = cache[cache_r];
        cp[i] = y;
    }

}

int main(){
    // double* h_data = (double*)malloc(sizeof(double)*4096*4096);
    double* d_data, *d_cp;
    cudaMalloc(&d_data, sizeof(double)*4096*4096);
    cudaMalloc(&d_cp, sizeof(double)*4096*4096);
    cudaDeviceSynchronize();
    auto start = system_clock::now();
    shared_bandwidth_test<<<4096*4096/128/256, 256>>>(d_data, d_cp, 4096*4096);
    cudaDeviceSynchronize();
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    start = system_clock::now();
    shared_bandwidth_extra_cost_test<<<4096*4096/128/256, 256>>>(d_data, d_cp, 4096*4096);
    cudaDeviceSynchronize();
    end = system_clock::now();
    auto duration_extra_cost = duration_cast<microseconds>(end - start);
    cout<<double(duration.count())<<endl;
    cout<<double(duration_extra_cost.count())<<endl;
    double bandwidth = 4096*4096*2*8/(double(duration.count())-double(duration_extra_cost.count()));
    cout <<  "shared memory bandwidth: " 
     << bandwidth
     << "GB/s" << endl;
}