#pragma once

#include <memory>
#include <sstream>

#include "common/base_matrix.h"

#ifdef __APPLE__
#include <boost/thread/tss.hpp>
#include <boost/pool/object_pool.hpp>
#endif

#include "gpu/types-gpu.h"

namespace GPU {
namespace mblas {

using namespace thrust::placeholders;


template <class VecType>
class TMatrix : public BaseMatrix {
  public:
    typedef typename VecType::value_type value_type;

    TMatrix()
    //:data2_(NULL)
    {}

    // always init to zero
    TMatrix(size_t rows, size_t cols, size_t batchSize)
    : BaseMatrix(rows, cols, batchSize)
    , data_(shape_.elements(), 0.0f)
    {
      //HANDLE_ERROR( cudaMalloc(&data2_, shape_.elements() * sizeof(value_type)) );
      //HANDLE_ERROR( cudaMemset(data2_, 0, shape_.elements() * sizeof(value_type)) );
    }

    TMatrix(TMatrix&& m)
    : BaseMatrix(m)
    , data_(std::move(m.data_))
    {
      /*
      HANDLE_ERROR( cudaMalloc(&data2_, shape_.elements() * sizeof(value_type)) );
      HANDLE_ERROR( cudaMemcpy(
          data2_,
          m.data2_,
          shape_.elements() * sizeof(value_type),
          cudaMemcpyDeviceToDevice) );
      */
    }

    TMatrix(const TMatrix& m) = delete;

    ~TMatrix() {
      //cudaFree(data2_);
    }

    value_type operator()(size_t i, size_t j) const {
      value_type ret;
      const value_type &src = data()[i * shape_[1] + j];
      HANDLE_ERROR( cudaMemcpy(&ret, &src, sizeof(value_type), cudaMemcpyDeviceToHost) );

      return ret;
    }

    size_t Rows() const {
      return shape_[0];
    }

    size_t Cols() const {
      return shape_[1];
    }

    void Resize(size_t rows, size_t cols, size_t batchSize) {
      size_t oldSize = shape_.elements();

      Reshape(rows, cols, batchSize);

      if (shape_.elements() > oldSize) {
        data_.resize(shape_.elements());

        //cudaFree(data2_);
        //HANDLE_ERROR( cudaMalloc((void**)&data2_, shape_.elements() * sizeof(value_type)) );
      }
    }

    virtual std::string Debug() const
    {
      std::stringstream strm;
      /*
      strm << Rows() << "x" << Cols() << ":";
      for (size_t row = 0; row < Rows(); ++row) {
        value_type rowSum = 0;
        for (size_t col = 0; col < Cols(); ++col) {
          rowSum += (*this)(row, col);
        }
        strm << rowSum << " ";
      }
      */
      return strm.str();
    }

    value_type* data() {
      return thrust::raw_pointer_cast(data_.data());
    }

    const value_type* data() const {
      return thrust::raw_pointer_cast(data_.data());
    }

    size_t size() const {
      return shape_.elements();
    }

    void swap(TMatrix &other) {
      shape_.swap(other.shape_);
      data_.swap(other.data_);
    }

    void copy(const TMatrix &other, size_t outOffset = 0) {
      HANDLE_ERROR( cudaMemcpy(
          data() + outOffset,
          other.data(),
          other.shape_.elements() * sizeof(value_type),
          cudaMemcpyDeviceToDevice) );
    }

    void copy(const TMatrix &other, size_t inStart, size_t inLength) {
      HANDLE_ERROR( cudaMemcpy(
          data(),
          other.data() + inStart,
          inLength * sizeof(value_type),
          cudaMemcpyDeviceToDevice) );
    }

  private:
    VecType data_;
    //value_type *data2_;

};

typedef TMatrix<DeviceVector<float>> Matrix;
typedef TMatrix<DeviceVector<int>> IMatrix;


}  // namespace mblas
}  // namespace GPU
