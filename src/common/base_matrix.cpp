#include <sstream>
#include "base_matrix.h"

std::string Shape::Debug() const
{
  std::stringstream strm;
  strm << rows << "x" << cols; // ":\n";
  return strm.str();
}
