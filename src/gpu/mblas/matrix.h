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
    typedef typename VecType::iterator iterator;
    typedef typename VecType::const_iterator const_iterator;

    TMatrix()
    :data2_(NULL)
    {}

    // always init to zero
    TMatrix(size_t rows, size_t cols, size_t batchSize)
    : BaseMatrix(rows, cols, batchSize)
    , data_(shape_.elements(), 0.0f)
    {
      HANDLE_ERROR( cudaMalloc(&data2_, shape_.elements() * sizeof(float)) );
      HANDLE_ERROR( cudaMemset(data2_, 0, shape_.elements() * sizeof(float)) );
    }

    TMatrix(TMatrix&& m)
    : BaseMatrix(m)
    , data_(std::move(m.data_))
    {
      HANDLE_ERROR( cudaMalloc(&data2_, shape_.elements() * sizeof(float)) );
      HANDLE_ERROR( cudaMemcpy(
          data2_,
          m.data2_,
          shape_.elements() * sizeof(float),
          cudaMemcpyDeviceToDevice) );
    }

    TMatrix(const TMatrix& m) = delete;

    ~TMatrix() {
      cudaFree(data2_);
    }

    value_type operator()(size_t i, size_t j) const {
      return data_[i * shape_[1] + j];
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

        cudaFree(data2_);
        HANDLE_ERROR( cudaMalloc((void**)&data2_, shape_.elements() * sizeof(float)) );
      }
    }

    virtual std::string Debug() const
    {
      std::stringstream strm;
      /*
      strm << Rows() << "x" << Cols() << ":";
      for (size_t row = 0; row < Rows(); ++row) {
        float rowSum = 0;
        for (size_t col = 0; col < Cols(); ++col) {
          rowSum += (*this)(row, col);
        }
        strm << rowSum << " ";
      }
      */
      return strm.str();
    }

    VecType& GetVec() {
      return data_;
    }

    value_type* data() {
      return thrust::raw_pointer_cast(data_.data());
    }

    const value_type* data() const {
      return thrust::raw_pointer_cast(data_.data());
    }

    iterator begin() {
      return data_.begin();
    }

    const_iterator begin() const{
      return data_.begin();
    }

    const_iterator end() const {
      return data_.begin() + size();
      // return data_.end();
    }

    size_t size() const {
      // return data_.size();
      return shape_.matrixSize();
    }

  private:
    VecType data_;
    float *data2_;

};

typedef TMatrix<DeviceVector<float>> Matrix;
typedef TMatrix<DeviceVector<int>> IMatrix;


}  // namespace mblas
}  // namespace GPU
