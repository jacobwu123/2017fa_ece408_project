
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 1024
#define SHARE_MEM_BLOCK 512
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{




__global__ void unroll_Kernel(int sampleId, int C, int H, int W, int K, float* x, float* X_unroll) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out*W_out;
   // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
   // (void)W_out; // silence declared but never referenced warning. remove this line when you start working
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
    int t = blockIdx.x*CUDA_MAX_NUM_THREADS + threadIdx.x;
    if(t < C*W_unroll)
    {
       int c = t / W_unroll; //y
       int s = t % W_unroll; // x
       int h_out = s / W_out;
       int w_out = s % W_out;
       int h_unroll = h_out * W_out + w_out;
       int w_base = c * K * K;
       for(int p = 0; p < K; p++){
            for(int q = 0; q < K; q++) {
                int w_unroll = w_base + p * K + q; 
                X_unroll[w_unroll*W_unroll + h_unroll] = x4d(sampleId, c, h_out + p, w_out + q);
            }
        }
    }

    #undef x4d
    #undef k4d
}

__global__ void multiplication(int sampleId, int M, int C, int K, int H_out, int W_out, float*k, float*X_unroll, float*y){
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    const int filterWidth = C*K*K;
    const int yWidth = H_out*W_out;
    __shared__ float shareK[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shareX[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;
    float Cvalue = 0;
    for(int ph = 0; ph < ceil(filterWidth/(float)TILE_WIDTH);ph++){
        if((Row < M) && (ph*TILE_WIDTH + tx < filterWidth))
            shareK[ty][tx] = k[Row*filterWidth + ph*TILE_WIDTH + tx];
        else
            shareK[ty][tx] = 0;
        if((Col < yWidth) && ( ph*TILE_WIDTH + ty < filterWidth))
            shareX[ty][tx] = X_unroll[(ph*TILE_WIDTH + ty)*yWidth + Col];
        else
            shareX[ty][tx] = 0;
        __syncthreads();
        if(Row < M && Col < yWidth){
            for(int i = 0; i < TILE_WIDTH; i++)
                Cvalue += shareK[ty][i]*shareX[i][tx];
        }
        __syncthreads();
    }

    
    if(Row < M && Col < yWidth)
        y[sampleId*(M*H_out*W_out) + Row*yWidth + Col] = Cvalue;

    #undef y4d
    #undef k4d
}


/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
  //  CHECK_EQ(0, 1) << "Reach line 62!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    //cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0]; // number of inputs
    const int M = y.shape_[1]; // number of output feature maps
    const int C = x.shape_[1]; // number of input feature maps
    const int H = x.shape_[2]; // height of each input feature map
    const int W = x.shape_[3]; // width of each input feature map
    const int K = w.shape_[3]; // width of filter
    const int H_out = H - K + 1; // height of each output feature map
    const int W_out = W - K + 1; // width of each output feature map
    cudaStream_t s[32];
    for(int i = 0; i < 32; i++){
        cudaStreamCreate(&s[i]);
    }
    //Set the kernel dimensions
    int num_threads = C * H_out * W_out;
    int num_blocks = ceil((num_threads * 1.0) / CUDA_MAX_NUM_THREADS);
    float* X_unroll;

    dim3 gridDim(ceil(C*H_out*W_out*1.0/TILE_WIDTH),ceil(M*1.0/TILE_WIDTH),1);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);

    cudaMalloc((void**)&X_unroll, sizeof(float)*C*K*K*H_out*W_out);

    for(int sampleId = 0; sampleId < B; sampleId++){
        unroll_Kernel<<<num_blocks, CUDA_MAX_NUM_THREADS,0,s[B%32]>>>(sampleId, C, H, W, K, x.dptr_, X_unroll);
        multiplication<<<gridDim,blockDim,0,s[B%32]>>>(sampleId, M, C, K, H_out, W_out, w.dptr_, X_unroll, y.dptr_);
    }

    cudaFree(X_unroll);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif