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
    {}

    TMatrix(size_t rows, size_t cols)
    : BaseMatrix(rows, cols, 1)
    , data_(shape_.elements())
    {}

    TMatrix(size_t rows, size_t cols, value_type val)
    : BaseMatrix(rows, cols, 1)
    , data_(shape_.elements(), val)
    {}

    TMatrix(TMatrix&& m)
    : BaseMatrix(m)
    , data_(std::move(m.data_))
    {}

    TMatrix(const TMatrix& m) = delete;

    value_type operator()(size_t i, size_t j) const {
      return data_[i * shape_[1] + j];
    }

    void Set(size_t i, size_t j, float value)  {
      data_[i * shape_[1] + j] = value;
    }

    size_t Rows() const {
      return shape_[0];
    }

    size_t Cols() const {
      return shape_[1];
    }

    void Resize(size_t rows, size_t cols) {
      Reshape(rows, cols, 1);

      if (shape_.matrixSize() > data_.size()) {
        data_.resize(shape_.elements());
      }
    }

    virtual std::string Debug() const
    {
      std::stringstream strm;
      strm << Rows() << "x" << Cols() << ":";
      for (size_t row = 0; row < Rows(); ++row) {
        float rowSum = 0;
        for (size_t col = 0; col < Cols(); ++col) {
          rowSum += (*this)(row, col);
        }
        strm << rowSum << " ";
      }
      return strm.str();
    }

    void Purge() {
      Clear();
      VecType temp;
      data_.swap(temp);
    }

    void Clear() {
      data_.clear();
      shape_[0] = 0;
      shape_[1] = 0;
    }

    VecType& GetVec() {
      return data_;
    }

    const VecType& GetVec() const {
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

    iterator end() {
      return data_.begin() + size();
      // return data_.end();
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
};

typedef TMatrix<DeviceVector<float>> Matrix;
typedef TMatrix<DeviceVector<int>> IMatrix;


}  // namespace mblas
}  // namespace GPU
