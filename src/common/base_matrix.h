#pragma once

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"

class Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Beam;

class Scorer;
typedef std::shared_ptr<Scorer> ScorerPtr;

class BaseMatrix {
  public:
  BaseMatrix()
  : rows_(0), cols_(0)
  {}

  BaseMatrix(size_t rows, size_t cols)
  : rows_(rows), cols_(cols)
  {}

  BaseMatrix(const BaseMatrix &m)
  : rows_(m.rows_), cols_(m.cols_)
  {}

  virtual ~BaseMatrix() {}

  virtual size_t Rows() const = 0;
  virtual size_t Cols() const = 0;
  virtual void Resize(size_t rows, size_t cols) = 0;

  virtual std::string Debug() const = 0;

protected:
  size_t rows_;
  size_t cols_;

};
