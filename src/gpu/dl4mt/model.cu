#include "model.h"

namespace GPU {

Weights::Weights(const std::string& npzFile, size_t device)
: Weights(NpzConverter(npzFile), device)
{
  //std::cerr << "Weights constructor(1)" << std::endl;
}

Weights::Weights(const NpzConverter& model, size_t device)
: encEmbeddings_(model),
  encForwardGRU_(model),
  encBackwardGRU_(model),
  decEmbeddings_(model),
  decInit_(model),
  decGru1_(model),
  decGru2_(model),
  decAlignment_(model),
  decSoftmax_(model),
  device_(device)
{
  //std::cerr << "Weights constructor(2)" << std::endl;
}

}

