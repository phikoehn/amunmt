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
  //cerr << "Context=" << Context.DebugShape() << endl;

  EmbeddedSentence & embeddedSentence = embeddedSentences_[sentInd];
  //cerr << "embeddings_=" << embeddings_.w_.E_.Debug() << endl;

  forwardRnn_.GetContext(embeddedSentence.cbegin(),
      embeddedSentence.cend(),
						 Context, false);
  backwardRnn_.GetContext(embeddedSentence.crbegin(),
      embeddedSentence.crend(),
						  Context, true);
}

void Encoder::GetContext(const Sentences& sentences, size_t tab,
        mblas::Matrix& Context) {
  embeddedSentences_.resize(sentences.size());

  for (size_t sentInd = 0; sentInd < sentences.size(); ++sentInd) {
    const Sentence *sentence = sentences.at(sentInd);
    const std::vector<size_t>& words = sentence->GetWords(tab);

    EmbeddedSentence & embeddedSentence = embeddedSentences_[sentInd];
    embeddedSentence.clear();

    for(auto& w : words) {
      embeddedSentence.emplace_back();
      embeddings_.Lookup(embeddedSentence.back(), w);
    }

  }
}

}

