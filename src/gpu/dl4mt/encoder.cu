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
  cerr << "Context=" << Context.GetShape().Debug() << endl;

  EmbeddedSentence &embeddedSentenceFwd = embeddedSentencesFwd_[sentInd];
  EmbeddedSentence &embeddedSentenceBck = embeddedSentencesBck_[sentInd];

  cerr << "embeddings_=" << embeddings_.w_.E_.GetShape().Debug() << endl;
  cerr << "embeddedSentenceFwd=" << embeddedSentenceFwd.size() << endl;

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
      mblas::Matrix &mFwd = embeddedSentenceFwd[i];
      embeddings_.Lookup(mFwd, wordFwd);

      const Word &wordBck = words[words.size() - i - 1];
      mblas::Matrix &mBck = embeddedSentenceBck[i];
      embeddings_.Lookup(mBck, wordBck);

      cerr << "mFwd=" << mFwd.GetShape().Debug() << endl;
      cerr << "mBck=" << mBck.GetShape().Debug() << endl;
    }
  }
}

}

