#include "matrix.h"

namespace GPU {

namespace mblas {

#ifdef __APPLE__
boost::thread_specific_ptr<cublasHandle_t> CublasHandler::handle_;
#else
thread_local cublasHandle_t* CublasHandler::handle_ = nullptr;
#endif

void Swap(Matrix& Out, Matrix& In) {
  size_t iRows = In.GetShape().rows;
  size_t iCols = In.GetShape().cols;
  size_t oRows = Out.GetShape().rows;
  size_t oCols = Out.GetShape().cols;

  Out.GetShape().Resize(iRows, iCols);
  In.GetShape().Resize(oRows, oCols);

  In.GetVec().swap(Out.GetVec());
}

void Mean(Matrix& Out, const Matrix& In) {
  size_t m = In.GetShape().rows;
  size_t n = In.GetShape().cols;

  Out.Resize(1, n, 0.f);
  Matrix Ones(1, m, 1, 1.f);

  float alpha = 1.0 / m;
  float beta  = 0.0;
  cublasSgemv(CublasHandler::GetHandle(), CUBLAS_OP_N, n, m, &alpha, In.data(), n,
              Ones.data(), 1, &beta, Out.data(), 1);
}

void Transpose(Matrix& Out, const Matrix& In) {
  size_t m = In.GetShape().rows;
  size_t n = In.GetShape().cols;

  Out.Resize(n, m);

  float alpha = 1.0;
  float beta  = 0.0;

  cublasSgeam(CublasHandler::GetHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, In.data(), n,
              &beta, In.data(), n, Out.data(), m);

}

void Transpose(Matrix& Out) {
  Matrix Temp;
  Transpose(Temp, Out);
  Swap(Out, Temp);
}

Matrix& Concat(Matrix& Out, const Matrix& In) {
  size_t oldSize = Out.size();
  Out.Resize(Out.GetShape().rows + In.GetShape().rows, Out.GetShape().cols);
  lib::copy(In.begin(), In.end(), Out.begin() + oldSize);
  return Out;
}

void Copy(Matrix& Out, const Matrix& In) {
  Out.Resize(In.GetShape().rows, In.GetShape().cols);
  lib::copy(In.begin(), In.end(), Out.begin());
}

void PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r, const size_t c) {
  size_t start = r * Out.GetShape().cols + c;
  lib::copy(In.begin(), In.end(), Out.begin() + start);
}

void CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r, const size_t c) {
  size_t length = In.GetShape().cols - c;
  Out.Resize(1, length);
  size_t start = r * In.GetShape().cols + c;
  size_t end   = start + length;
  lib::copy(In.begin() + start, In.begin() + end, Out.begin());
}

__global__ void gCopyRows(float* out, const float* in, size_t cols,
                          const RowPair* devPairs, size_t numPairs) {
  for(int bid = 0; bid < numPairs; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < numPairs) {
      size_t dstId = devPairs[j].first;
      size_t srcId = devPairs[j].second;

      float* rowOut = out + dstId * cols;
      const float* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

void CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPair* devPairs,
                 size_t numPairs) {
  float* d_out = Out.data();
  const float* d_in = In.data();

  int threads = std::min(MAX_THREADS, (int)In.GetShape().cols);
  int blocks = std::min(MAX_BLOCKS, (int)numPairs);;
  gCopyRows<<<blocks, threads>>>(d_out, d_in, In.GetShape().cols, devPairs, numPairs);
  cudaStreamSynchronize(0);
}

void CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPairs& pairs) {
  thrust::device_vector<RowPair> devPairs = pairs;
  CopyRows(Out, In, thrust::raw_pointer_cast(devPairs.data()), devPairs.size());
}

void Assemble(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces) {
  RowPairs rowPairs;
  for(size_t i = 0; i < indeces.size(); i++)
    rowPairs.emplace_back(i, indeces[i]);
  Out.Resize(rowPairs.size(), In.GetShape().cols);
  CopyRows(Out, In, rowPairs);
}

__global__ void gSlice(float* out, const float* in,
                       size_t n, size_t dim,
                       size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * dim;
      const float* rowIn = in + j * cols + n * dim;

      for(int tid = 0; tid < dim; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < dim)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

void Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim) {

  Out.Resize(In.GetShape().rows, dim);

  float* d_out = Out.data();
  const float* d_in = In.data();

  int threads = std::min(MAX_THREADS, (int)dim);
  int blocks = std::min(MAX_BLOCKS, (int)In.GetShape().rows);
  gSlice<<<blocks, threads>>>(d_out, d_in, n, dim, In.GetShape().rows, In.GetShape().cols);
  cudaStreamSynchronize(0);
}

void Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB) {
  Matrix::value_type alpha = 1.0;
  Matrix::value_type beta = 0.0;

  //size_t m = A.GetShape().rows;
  //size_t k = A.GetShape().cols;
  ////if(transA)
  ////  std::swap(m, k);
  //
  //size_t l = B.GetShape().rows;
  //size_t n = B.GetShape().cols;
  ////if(transB)
  ////  std::swap(l, n);
  //
  //C.Resize(m, n);
  //
  //size_t lda = A.GetShape().cols;
  //size_t ldb = B.GetShape().cols;
  //size_t ldc = C.GetShape().cols;
  //
  //nervana_sgemm(const_cast<float*>(A.data()),
  //              const_cast<float*>(B.data()),
  //              C.data(),
  //              transA, transB,
  //              m, n, k,
  //              lda, ldb, ldc,
  //              alpha, beta,
  //              0, false, false, 0);

  size_t m = A.GetShape().rows;
  size_t k = A.GetShape().cols;
  if(transA)
    std::swap(m, k);

  size_t l = B.GetShape().rows;
  size_t n = B.GetShape().cols;
  if(transB)
    std::swap(l, n);

  size_t lda = A.GetShape().cols;
  size_t ldb = B.GetShape().cols;
  size_t ldc = B.GetShape().cols;

  if(transB)
    ldc = B.GetShape().rows;

  C.Resize(m, n);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemm(handle, opB, opA,
              n, m, k, &alpha, B.data(), ldb, A.data(), lda, &beta, C.data(), ldc);
}

void Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB) {
 Prod(CublasHandler::GetHandle(), C, A, B, transA, transB);
}

__global__ void gSoftMax(float* softMaxP, size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;
      float* sp = softMaxP + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sp[id] = __expf(sp[id]);
          _sum[threadIdx.x] += sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x){
        int id = tid + threadIdx.x;
        if(id < cols)
          sp[id] /= _sum[0];
      }
    }
  }
}

void Softmax(Matrix& Out) {
  int blocks = std::min(MAX_BLOCKS, (int)Out.GetShape().rows);
  int threads = std::min(MAX_THREADS, (int)Out.GetShape().cols);
  int shared = sizeof(float) * threads * 2;
  gSoftMax<<<blocks, threads, shared>>>(Out.data(), Out.GetShape().rows, Out.GetShape().cols);
  cudaStreamSynchronize(0);
}

}

}

