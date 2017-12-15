
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 32
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

//__constant__ float MASK[M][C][K][K];


__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, 
                                         const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(W_out*1.0/TILE_WIDTH); // width of gird
   // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
   // (void)W_out; // silence declared but never referenced warning. remove this line when you start working
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
    int n = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.x;
    int w0 = threadIdx.y;

    int h_base = (blockIdx.z/W_grid) * TILE_WIDTH; // elementwize starting height index
    int w_base = (blockIdx.z%W_grid) * TILE_WIDTH; // elementwise starting width index
    int h = h_base + threadIdx.y; // actual height index 
    int w = w_base + threadIdx.x; //  actual width index
    
    int x_tile_width = TILE_WIDTH + K - 1; // input tile size
    //declare shared memory
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[x_tile_width*x_tile_width]; // starting address of shared weights

    float acc = 0.0;
    for(int c = 0; c < C; c++){ // number of feature maps
        if(h < H_out && w < W_out){
            if(h0 < K && w0 < K){
                W_shared[w0*K+h0] = k4d(m,c, w0, h0);
            }
        }
        if(h < H_out && w < W_out){
            for(int i = h;i < h_base + x_tile_width; i += TILE_WIDTH){
                for(int j = w; j < w_base + x_tile_width; j += TILE_WIDTH){
                    if(i < H && j < W)
                    {
                        X_shared[(i-h_base)*x_tile_width + j-w_base] = x4d(n,c,i,j);
                    }
                }
            }
        }
        __syncthreads();
        if(h < H_out && w < W_out)
        {
            for(int p = 0; p < K; p++){
                for(int q = 0; q < K; q++)
                {
                    acc += X_shared[(w0 + p)*x_tile_width + h0 + q] * W_shared[p*K + q];
                }
                
            }   
        }
        __syncthreads();
    }    
    if(h < H_out && w < W_out)
        y4d(n,m,h,w) = acc; 

    #undef y4d
    #undef x4d
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
    cudaStream_t s = y.stream_->stream_;

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
    const int W_grid = ceil(W_out*1.0/(float)TILE_WIDTH);
    const int H_grid = ceil(H_out*1.0/(float)TILE_WIDTH);
    const int Z = H_grid*W_grid;
    // Set the kernel dimensions
  //  cudaMemcpyToSymbol(MASK,w.dptr_, K*K*M*C*sizeof(float));

    dim3 gridDim(B,M,Z);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    size_t shmem_size = sizeof(float*)*((TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1) + K*K);
    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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