#include <sstream>
#include "base_matrix.h"

std::string Shape::Debug() const
{
  std::stringstream strm;
  strm << rows << "x" << cols << "x" << batches; // ":\n";
  return strm.str();
}
