#pragma once

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"
#include "common/shape.h"

class Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Beam;

class Scorer;
typedef std::shared_ptr<Scorer> ScorerPtr;

class BaseMatrix {
  public:
  BaseMatrix()
  : shape_({0, 0, 0})
  {}

  BaseMatrix(size_t rows, size_t cols, size_t batchSize)
  : shape_({rows, cols, batchSize})
  {}

  BaseMatrix(const BaseMatrix &m)
  : shape_(m.shape_)
  {}

  virtual ~BaseMatrix() {}

  virtual void Resize(size_t rows, size_t cols, size_t batchSize) = 0;

  void Reshape(size_t rows, size_t cols, size_t batchSize) {
    shape_[0] = rows;
    shape_[1] = cols;
    shape_[2] = batchSize;
  }

  virtual std::string Debug() const = 0;

  size_t shape(size_t  ind) const {
    return shape_[ind];
  }

  const Shape &shape() const {
    return shape_;
  }

  Shape &shape() {
    return shape_;
  }

protected:
  Shape shape_;

};
