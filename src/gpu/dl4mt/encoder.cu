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

  EmbeddedSentence &embeddedSentenceFwd = embeddedSentencesFwd_[sentInd];
  EmbeddedSentence &embeddedSentenceBck = embeddedSentencesBck_[sentInd];
  //cerr << "embeddings_=" << embeddings_.w_.E_.Debug() << endl;

  forwardRnn_.GetContext(embeddedSentenceFwd.cbegin(),
      embeddedSentenceFwd.cend(),
						 Context, false);
  backwardRnn_.GetContext(embeddedSentenceBck.cbegin(),
      embeddedSentenceBck.cend(),
						  Context, true);
}

void Encoder::GetContext(const Sentences& sentences, size_t tab,
        mblas::Matrix& Context) {
  embeddedSentencesFwd_.resize(sentences.size());
  embeddedSentencesBck_.resize(sentences.size());

  for (size_t sentInd = 0; sentInd < sentences.size(); ++sentInd) {
    const Sentence *sentence = sentences.at(sentInd);
    const std::vector<size_t>& words = sentence->GetWords(tab);
    size_t sentenceLen = words.size();

    EmbeddedSentence & embeddedSentenceFwd = embeddedSentencesFwd_[sentInd];
    embeddedSentenceFwd.resize(sentenceLen);

    EmbeddedSentence & embeddedSentenceBck = embeddedSentencesBck_[sentInd];
    embeddedSentenceBck.resize(sentenceLen);

    for (size_t i = 0; i < words.size(); ++i) {
      const Word &wordFwd = words[i];
      embeddings_.Lookup(embeddedSentenceFwd[i], wordFwd);

      const Word &wordBck = words[words.size() - i - 1];
      embeddings_.Lookup(embeddedSentenceBck[i], wordBck);

    }
  }
}

}

