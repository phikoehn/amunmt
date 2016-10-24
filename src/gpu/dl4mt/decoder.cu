#include "decoder.h"

namespace GPU {

Decoder::Decoder(const Weights& model)
: embeddings_(model.decEmbeddings_),
  rnn1_(model.decInit_, model.decGru1_),
  rnn2_(model.decGru2_),
  alignment_(model.decAlignment_),
  softmax_(model.decSoftmax_)
{}

void Decoder::MakeStep(mblas::Matrix& NextState,
              mblas::Matrix& Probs,
              const mblas::Matrix& State,
              const mblas::Matrix& Embeddings,
              const mblas::Matrix& SourceContext) {
  GetHiddenState(HiddenState_, State, Embeddings);

  GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext);


  GetNextState(NextState, HiddenState_, AlignedSourceContext_);

  /*
  std::cerr << "BEFORE="
            << Probs.GetShape().Debug() << " "
            << NextState.GetShape().Debug() << " "
            << Embeddings.GetShape().Debug() << " "
            << AlignedSourceContext_.GetShape().Debug() << " "
            << std::endl;
   */
  GetProbs(Probs, NextState, Embeddings, AlignedSourceContext_);
  /*
  std::cerr << "AFTER="
            << Probs.GetShape().Debug() << " "
            << NextState.GetShape().Debug() << " "
            << Embeddings.GetShape().Debug() << " "
            << AlignedSourceContext_.GetShape().Debug() << " "
            << std::endl;
  */
}

void Decoder::EmptyState(mblas::Matrix& State,
                const mblas::Matrix& SourceContext,
                size_t batchSize) {
  rnn1_.InitializeState(State, SourceContext, batchSize);
  alignment_.Init(SourceContext);
}

void Decoder::EmptyEmbedding(mblas::Matrix& Embedding,
                    size_t batchSize) {
  Embedding.Clear();
  Embedding.Resize(batchSize, embeddings_.GetCols(), 0);
}

void Decoder::Lookup(mblas::Matrix& Embedding,
            const std::vector<size_t>& w) {
  embeddings_.Lookup(Embedding, w);
}

void Decoder::Filter(const std::vector<size_t>& ids) {
  softmax_.Filter(ids);
}

void Decoder::GetAttention(mblas::Matrix& Attention) {
  alignment_.GetAttention(Attention);
}

size_t Decoder::GetVocabSize() const {
  return embeddings_.GetRows();
}

mblas::Matrix& Decoder::GetAttention() {
  return alignment_.GetAttention();
}

void Decoder::GetHiddenState(mblas::Matrix& HiddenState,
                    const mblas::Matrix& PrevState,
                    const mblas::Matrix& Embedding) {
  rnn1_.GetNextState(HiddenState, PrevState, Embedding);
}

void Decoder::GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                             const mblas::Matrix& HiddenState,
                             const mblas::Matrix& SourceContext) {
  alignment_.GetAlignedSourceContext(AlignedSourceContext, HiddenState, SourceContext);
}

void Decoder::GetNextState(mblas::Matrix& State,
                  const mblas::Matrix& HiddenState,
                  const mblas::Matrix& AlignedSourceContext) {
  rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
}

void Decoder::GetProbs(mblas::Matrix& Probs,
              const mblas::Matrix& State,
              const mblas::Matrix& Embedding,
              const mblas::Matrix& AlignedSourceContext) {
  softmax_.GetProbs(Probs, State, Embedding, AlignedSourceContext);
}

}

