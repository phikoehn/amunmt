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

class History;

///////////////////////////////////////////////////////////////////
class BaseMatrix;
typedef std::vector<BaseMatrix*> BaseMatrices;

///////////////////////////////////////////////////////////////////
class Shape
{
public:
  size_t rows, cols, batches;

  Shape(size_t rows, size_t cols, size_t batches)
  {
    Resize(rows, cols, batches);
  }

  virtual std::string Debug() const;

  void Resize(size_t rows, size_t cols, size_t batches = 1) {
    this->rows = rows;
    this->cols = cols;
    this->batches = batches;
  }

  size_t GetSize() const {
    return GetMatrixSize() * batches;
  }

  size_t GetMatrixSize() const {
    return rows * cols;
  }
};

///////////////////////////////////////////////////////////////////

class BaseMatrix {
public:
  BaseMatrix()
  :shape_(0,0,0)
  {
  }

  BaseMatrix(size_t rows, size_t cols, size_t batches)
  :shape_(rows, cols, batches)
  {
  }

  virtual ~BaseMatrix() {}

  const Shape &GetShape() const {
    return shape_;
  }

  Shape &GetShape() {
    return shape_;
  }

  virtual void Resize(size_t rows, size_t cols) = 0;

  virtual void BestHyps(Beam& bestHyps,
      const Beam& prevHyps,
      BaseMatrix& Probs,
      const size_t beamSize,
      History& history,
      const std::vector<ScorerPtr> &scorers,
      const Words &filterIndices,
      bool returnAlignment=false) const = 0;

  virtual std::string Debug() const = 0;

  protected:
    Shape shape_;
};
