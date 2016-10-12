#include "encoder.h"

using namespace std;

namespace GPU {

Encoder::Encoder(const Weights& model)
: embeddings_(model.encEmbeddings_),
  forwardRnn_(model.encForwardGRU_),
  backwardRnn_(model.encBackwardGRU_)
{}

void Encoder::GetContext(size_t sentInd, const std::vector<size_t>& words,
				mblas::Matrix& Context) {
  Context.Resize(words.size(), forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());
  cerr << "Context=" << Context.DebugShape() << endl;

  embeddedWords_.clear();

  for(auto& w : words) {
    embeddedWords_.emplace_back();
    embeddings_.Lookup(embeddedWords_.back(), w);
  }
  //cerr << "embeddings_=" << embeddings_.w_.E_.Debug() << endl;

  forwardRnn_.GetContext(embeddedWords_.cbegin(),
						 embeddedWords_.cend(),
						 Context, false);
  backwardRnn_.GetContext(embeddedWords_.crbegin(),
						  embeddedWords_.crend(),
						  Context, true);
}

void Encoder::GetContext(const Sentences& sentences, size_t tab,
        mblas::Matrix& Context) {

}

}

