#pragma once

#include <memory>
#include <sstream>

#include "common/base_matrix.h"

#ifdef __APPLE__
#include <boost/thread/tss.hpp>
#include <boost/pool/object_pool.hpp>
#endif

#include "handles.h"
#include "gpu/types-gpu.h"

namespace GPU {
namespace mblas {

using namespace thrust::placeholders;


template <class VecType>
class TMatrix : public BaseMatrix {
  public:
    typedef typename VecType::value_type value_type;

    TMatrix()
    :data2_(NULL)
    ,owner_(true)
    {
        //std::cerr << "TMatrix(1)=" << data2_ << std::endl;
    }

    // always init to zero
    TMatrix(const Shape &shape)
    : BaseMatrix(shape)
    //, data_(shape_.elements(), 0.0f)
    //, data_(shape_.elements())
    ,owner_(true)
    {
      cudaStream_t &stream = mblas::CudaStreamHandler::GetStream();

      HANDLE_ERROR( cudaMalloc(&data2_, shape_.elements() * sizeof(value_type)) );
      HANDLE_ERROR( cudaMemsetAsync(data(), 0, shape_.elements() * sizeof(value_type), stream) );

      //std::cerr << "TMatrix(2)=" << data2_ << std::endl;
    }

    TMatrix(TMatrix&& m)
    : BaseMatrix(m)
    //, data_(std::move(m.data_))
    //, data_(shape_.elements())
    ,owner_(true)
    {
      cudaStream_t &stream = mblas::CudaStreamHandler::GetStream();

      HANDLE_ERROR( cudaMalloc(&data2_, shape_.elements() * sizeof(value_type)) );
      HANDLE_ERROR( cudaMemcpyAsync(
          data(),
          m.data(),
          shape_.elements() * sizeof(value_type),
          cudaMemcpyDeviceToDevice,
          stream) );

      //std::cerr << "TMatrix(3)=" << data2_ << std::endl;
    }

    TMatrix(const TMatrix &m, size_t sliceInd)
    :BaseMatrix({m.shape(0), m.shape(1), 1})
    ,owner_(false)
    ,data2_(m.data2_ + m.shape_.matrixSize() * sliceInd)
    {
    }

    TMatrix(const TMatrix& m) = delete;

    ~TMatrix() {
      //std::cerr << "destrucor=" << data2_ << std::endl;
      if (owner_) {
    	  HANDLE_ERROR( cudaFree(data2_) );
      }
    }

    value_type operator()(size_t i, size_t j, size_t k) const {
      value_type ret;
      const value_type &src = data()[k * shape_.numSlices() + i * shape_[1] + j];

      cudaStream_t &stream = mblas::CudaStreamHandler::GetStream();
      HANDLE_ERROR( cudaMemcpyAsync(&ret, &src, sizeof(value_type), cudaMemcpyDeviceToHost, stream) );

      return ret;
    }

    void Resize(const Shape &shape) {
      size_t oldSize = shape_.elements();

      Reshape(shape);

      if (shape_.elements() > oldSize) {
        //data_.resize(shape_.elements());
        cudaStream_t &stream = mblas::CudaStreamHandler::GetStream();

        value_type *temp;
        HANDLE_ERROR( cudaMalloc(&temp, shape_.elements() * sizeof(value_type)) );

        if (oldSize) {
        	HANDLE_ERROR( cudaMemcpyAsync(temp, data(), oldSize * sizeof(value_type), cudaMemcpyDeviceToDevice, stream) );
        }

        HANDLE_ERROR( cudaFree(data2_) );

        data2_ = temp;
      }
    }

    virtual std::string Debug() const
    {
      std::stringstream strm;
      /*
      strm << shape(0) << "x" << shape(1) << ":";
      for (size_t row = 0; row < shape(0); ++row) {
        value_type rowSum = 0;
        for (size_t col = 0; col < shape(1); ++col) {
          rowSum += (*this)(row, col);
        }
        strm << rowSum << " ";
      }
      */
      return strm.str();
    }

    value_type* data() {
      //return thrust::raw_pointer_cast(data_.data());
      return data2_;
    }

    const value_type* data() const {
      //return thrust::raw_pointer_cast(data_.data());
      return data2_;
    }

    size_t size() const {
      return shape_.elements();
    }

    void swap(TMatrix &other) {
      shape_.swap(other.shape_);

      //data_.swap(other.data_);

      value_type *temp = data2_;
      data2_ = other.data2_;
      other.data2_ = temp;

    }

    void copy(const TMatrix &other, size_t outOffset = 0) {
      cudaStream_t &stream = mblas::CudaStreamHandler::GetStream();
      HANDLE_ERROR( cudaMemcpyAsync(
          data() + outOffset,
          other.data(),
          other.shape_.elements() * sizeof(value_type),
          cudaMemcpyDeviceToDevice,
          stream) );
    }

    void copy(const TMatrix &other, size_t inStart, size_t inLength) {
      cudaStream_t &stream = mblas::CudaStreamHandler::GetStream();
      HANDLE_ERROR( cudaMemcpyAsync(
          data(),
          other.data() + inStart,
          inLength * sizeof(value_type),
          cudaMemcpyDeviceToDevice,
          stream) );
    }

    TMatrix Slice(size_t sliceInd) const {
    	return TMatrix(*this, sliceInd);
    }

  private:
    //VecType data_;
    value_type *data2_;
    bool owner_;

};

typedef TMatrix<DeviceVector<float>> Matrix;
typedef TMatrix<DeviceVector<int>> IMatrix;


}  // namespace mblas
}  // namespace GPU
