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
  int batchNum = Out.Rows();
  int stateLength = Out.Cols();
  int sentenceLength = In.Rows() / batchNum;

  int nThreads = 512;
  int nBlocks =  (stateLength / 512) + ((stateLength % 512 == 0) ?  0 : 1);

  gMean<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (Out.data(), In.data(), thrust::raw_pointer_cast(mapping.data()),
     batchNum, sentenceLength, stateLength);
}

Matrix& Transpose(Matrix& Out, const Matrix& In) {
  size_t m = In.Rows();
  size_t n = In.Cols();

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
  Out.Resize(Out.Rows() + In.Rows(), Out.Cols(), 1);
  mblas::copy(In.begin(), In.end(), Out.begin() + oldSize);
  return Out;
}

Matrix& Copy(Matrix& Out, const Matrix& In) {
  Out.Resize(In.Rows(), In.Cols(), 1);
  mblas::copy(In.begin(), In.end(), Out.begin());
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
  int nColumns = In.Cols();
  int nRows = In.Rows();
  int nThreads = 512;
  int nBlocks =  (In.size() / 512) + ((In.size() % 512 == 0) ?  0 : 1);


  gPasteRows<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (Out.data(), rowNo, Out.Cols(), In.data(), In.Rows(), In.Cols(), colNo, sparse);
}

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r, const size_t c) {
  size_t start = r * Out.Cols() + c;
  mblas::copy(In.begin(), In.end(), Out.begin() + start);
  return Out;
}

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r, const size_t c) {
  size_t length = In.Cols() - c;
  Out.Resize(1, length, 1);
  size_t start = r * In.Cols() + c;
  size_t end   = start + length;
  mblas::copy(In.begin() + start, In.begin() + end, Out.begin());
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

  int threads = std::min(MAX_THREADS, (int)In.Cols());
  int blocks = std::min(MAX_BLOCKS, (int)numPairs);;

  gCopyRows<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (d_out, d_in, In.Cols(), dev, numPairs);

  return Out;
}


Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<size_t>& indeces) {
  Out.Resize(indeces.size(), In.Cols(), 1);
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

  Out.Resize(In.Rows(), dim, 1);

  float* d_out = Out.data();
  const float* d_in = In.data();

  int threads = std::min(MAX_THREADS, (int)dim);
  int blocks = std::min(MAX_BLOCKS, (int)In.Rows());

  gSlice<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (d_out, d_in, n, dim, In.Rows(), In.Cols());
  return Out;
}

Matrix& Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB) {
  Matrix::value_type alpha = 1.0;
  Matrix::value_type beta = 0.0;

  size_t m = A.Rows();
  size_t k = A.Cols();
  if(transA)
    std::swap(m, k);

  size_t l = B.Rows();
  size_t n = B.Cols();
  if(transB)
    std::swap(l, n);

  size_t lda = A.Cols();
  size_t ldb = B.Cols();
  size_t ldc = B.Cols();

  if(transB)
    ldc = B.Rows();

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
  int blocks = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  int shared = sizeof(float) * threads * 2;

  gSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (Out.data(), Out.Rows(), Out.Cols());
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
  int blocks = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  int shared = sizeof(float) * threads * 2;

  gLogSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (Out.data(), Out.Rows(), Out.Cols());

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
  int nColumns = In.Cols();
  int nRows = In.Rows();
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

