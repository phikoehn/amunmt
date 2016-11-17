#include "gpu/mblas/matrix_functions.h"

#include "gpu/mblas/handles.h"

namespace GPU {
namespace mblas {

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
          fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
          if (abort) exit(code);
      }
}

#ifdef __APPLE__
boost::thread_specific_ptr<cublasHandle_t> CublasHandler::handle_;
#else
thread_local cublasHandle_t* CublasHandler::handle_ = nullptr;
thread_local CudaStreamHandler* CudaStreamHandler::instance_ = nullptr;;
#endif

Matrix& Swap(Matrix& Out, Matrix& In) {
  Out.swap(In);
  return Out;
}

__global__ void gMean(float* d_out, const float* d_in, const int* mapping,
                      int batchNum, int senLen, int stateLength) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < stateLength) {
    float sum = 0.0f;
    int counter = 0;

    for (int i = 0; i < batchNum * senLen; ++i) {
      sum += mapping[i] * d_in[i * stateLength + id];
      counter += mapping[i];

      if ((i + 1) % senLen == 0) {
        sum /= counter;
        d_out[(i / senLen) * stateLength + id] = sum;
        sum = 0.0f;
        counter = 0;
      }
    }
  }
}

void Mean(Matrix& Out, const Matrix& In, const DeviceVector<int>& mapping) {
  int batchNum = Out.shape(0);
  int stateLength = Out.shape(1);
  int sentenceLength = In.shape(0) / batchNum;

  int nThreads = 512;
  int nBlocks =  (stateLength / 512) + ((stateLength % 512 == 0) ?  0 : 1);

  gMean<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (Out.data(), In.data(), thrust::raw_pointer_cast(mapping.data()),
     batchNum, sentenceLength, stateLength);
}

Matrix& Transpose(Matrix& Out, const Matrix& In) {
  size_t m = In.shape(0);
  size_t n = In.shape(1);

  Out.Resize(n, m, 1);

  float alpha = 1.0;
  float beta  = 0.0;

  cublasSgeam(CublasHandler::GetHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, In.data(), n,
              &beta, In.data(), n, Out.data(), m);

  return Out;
}

Matrix& Transpose(Matrix& Out) {
  Matrix Temp;
  Transpose(Temp, Out);
  Swap(Out, Temp);
  return Out;
}

Matrix& Concat(Matrix& Out, const Matrix& In) {
  size_t oldSize = Out.size();
  Out.Resize(Out.shape(0) + In.shape(0), Out.shape(1), 1);

  Out.copy(In, oldSize);
  return Out;
}

Matrix& Copy(Matrix& Out, const Matrix& In) {
  Out.Resize(In.shape(0), In.shape(1), 1);

  Out.copy(In);
  return Out;
}

__global__ void gPasteRows(float* d_out, int outRows, int outCols, const float* d_in, int inRows, int inCols, int colNo, int sparse) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < inRows * inCols) {
    int inRow = id / inCols;
    int inCol = id % inCols;
    int outID = (outRows + sparse * inRow) * outCols + inCol + colNo;
    d_out[outID] = d_in[id];
  }
}
void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo, size_t sparse) {
  int nColumns = In.shape(1);
  int nRows = In.shape(0);
  int nThreads = 512;
  int nBlocks =  (In.size() / 512) + ((In.size() % 512 == 0) ?  0 : 1);


  gPasteRows<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (Out.data(), rowNo, Out.shape(1), In.data(), In.shape(0), In.shape(1), colNo, sparse);
}

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r, const size_t c) {
  size_t start = r * Out.shape(1) + c;
  Out.copy(In, start);

  return Out;
}

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r, const size_t c) {
  size_t length = In.shape(1) - c;
  Out.Resize(1, length, 1);
  size_t start = r * In.shape(1) + c;

  Out.copy(In, start, length);

  return Out;
}

__global__ void gCopyRows(float* out, const float* in, size_t cols,
                          const size_t* targetRowIdx, size_t numPairs) {
  for(int bid = 0; bid < numPairs; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < numPairs) {
      size_t dstId = j;
      size_t srcId = targetRowIdx[j];

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

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const size_t* dev,
                 size_t numPairs) {
  float* d_out = Out.data();
  const float* d_in = In.data();

  int threads = std::min(MAX_THREADS, (int)In.shape(1));
  int blocks = std::min(MAX_BLOCKS, (int)numPairs);;

  gCopyRows<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (d_out, d_in, In.shape(1), dev, numPairs);

  return Out;
}


Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<size_t>& indeces) {
  Out.Resize(indeces.size(), In.shape(1), 1);
  CopyRows(Out, In, thrust::raw_pointer_cast(indeces.data()), indeces.size());
  return Out;
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

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim) {

  Out.Resize(In.shape(0), dim, 1);

  float* d_out = Out.data();
  const float* d_in = In.data();

  int threads = std::min(MAX_THREADS, (int)dim);
  int blocks = std::min(MAX_BLOCKS, (int)In.shape(0));

  gSlice<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (d_out, d_in, n, dim, In.shape(0), In.shape(1));
  return Out;
}

Matrix& Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB) {
  Matrix::value_type alpha = 1.0;
  Matrix::value_type beta = 0.0;

  size_t m = A.shape(0);
  size_t k = A.shape(1);
  if(transA)
    std::swap(m, k);

  size_t l = B.shape(0);
  size_t n = B.shape(1);
  if(transB)
    std::swap(l, n);

  size_t lda = A.shape(1);
  size_t ldb = B.shape(1);
  size_t ldc = B.shape(1);

  if(transB)
    ldc = B.shape(0);

  C.Resize(m, n, 1);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemm(handle, opB, opA,
              n, m, k, &alpha, B.data(), ldb, A.data(), lda, &beta, C.data(), ldc);
  return C;
}

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB) {
  return Prod(CublasHandler::GetHandle(), C, A, B, transA, transB);
}

__global__ void gSoftMax(float* softMaxP, size_t rows, size_t cols) {
  for (int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if (j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;
      float* sp = softMaxP + j * cols;
      _sum[threadIdx.x] = 0.0;
      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          sp[id] = __expf(sp[id]);
          _sum[threadIdx.x] += sp[id];
        }
      }

      __syncthreads();

      int len = blockDim.x;
      while (len != 1) {
        __syncthreads();

        int skip = (len + 1) >> 1;
        if (threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }

      __syncthreads();

      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          sp[id] /= _sum[0];
        }
      }
    }
  }
}

Matrix& Softmax(Matrix& Out) {
  int blocks = std::min(MAX_BLOCKS, (int)Out.shape(0));
  int threads = std::min(MAX_THREADS, (int)Out.shape(1));
  int shared = sizeof(float) * threads * 2;

  gSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (Out.data(), Out.shape(0), Out.shape(1));
  return Out;
}

__global__ void gLogSoftMax(float* softMaxP, size_t rows, size_t cols) {
  for (int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if (j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;
      float* sp = softMaxP + j * cols;
      _sum[threadIdx.x] = 0.0;
      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          sp[id] = __expf(sp[id]);
          _sum[threadIdx.x] += sp[id];
        }
      }

      int len = blockDim.x;
      while (len != 1) {
        __syncthreads();

        int skip = (len + 1) >> 1;
        if (threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }

      __syncthreads();

      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          sp[id] = __logf(sp[id]/_sum[0]);
        }
      }
    }
  }
}


Matrix& LogSoftmax(Matrix& Out) {
  int blocks = std::min(MAX_BLOCKS, (int)Out.shape(0));
  int threads = std::min(MAX_THREADS, (int)Out.shape(1));
  int shared = sizeof(float) * threads * 2;

  gLogSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (Out.data(), Out.shape(0), Out.shape(1));

  return Out;
}

__global__ void gSetColumn(float* d_in, int n_columns, int n_rows, int noColumn, float value) {
  int rowNumber = threadIdx.x  + blockDim.x * blockIdx.x;
  int index = noColumn + rowNumber * n_columns;

  if (index < n_columns * n_rows) {
    d_in[index] = value;
  }
}

void SetColumn(Matrix& In, int noColumn, float value) {
  int nColumns = In.shape(1);
  int nRows = In.shape(0);
  int nBlocks = nRows / 512 + (nRows % 512 == 0) ?  0 : 1;
  int nThreads = std::min(512, nRows);

  gSetColumn<<<nBlocks, nThreads, 0, mblas::CudaStreamHandler::GetStream()>>>
    (In.data(), nColumns, nRows, noColumn, value);
}

__global__ void gFill(float* d_in, int size, float val) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < size) {
    d_in[index] = val;
  }
}

void Fill(Matrix& In, float value) {
  size_t size = In.size();
  int nThreads = std::min(512, (int)size);
  int nBlocks = (size / nThreads) + ((size % nThreads == 0) ? 0 : 1);

  gFill<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (In.data(), size, value);
}

}  // namespace mblas
}  // namespace GPU

