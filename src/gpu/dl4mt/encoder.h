#pragma once

#include "gpu/mblas/matrix.h"
#include "model.h"
#include "gru.h"

namespace GPU {

class Encoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Matrix& Row, size_t i) {
          using namespace mblas;
          //std::cerr << "BEFORE Row=" << Row.GetShape().Debug() << std::endl;
          if(i < w_.E_.GetShape().rows)
            CopyRow(Row, w_.E_, i);
          else
            CopyRow(Row, w_.E_, 1); // UNK
          //std::cerr << "AFTER Row=" << Row.GetShape().Debug() << std::endl;
        }

        const Weights& w_;
      private:
    };

    template <class Weights>
    class RNN {
      public:
        RNN(const Weights& model)
        : gru_(model) {}

        void InitializeState(size_t batchSize = 1) {
          states_.resize(batchSize);

          for (mblas::Matrix &state: states_) {
            state.Clear();
            state.Resize(1, gru_.GetStateLength(), 0.0);
          }
        }

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Embd) {
          gru_.GetNextState(NextState, State, Embd);
        }

        template <class It>
        void GetContext(It it, It end, 
                        mblas::Matrix& Context, bool invert) {

          mblas::Matrix &state = states_[0];

          size_t n = std::distance(it, end);
          size_t i = 0;
          while(it != end) {
            GetNextState(state, state, *it++);
            if(invert)
              mblas::PasteRow(Context, state, n - i - 1, gru_.GetStateLength());
            else
              mblas::PasteRow(Context, state, i, 0);
            ++i;
          }

          //std::cerr << "Context=" << Context.GetShape().Debug() << std::endl;
          //std::cerr << "State_=" << State_.GetShape().Debug() << std::endl;
        }

        size_t GetStateLength() const {
          return gru_.GetStateLength();
        }

      private:
        // Model matrices
        const GRU<Weights> gru_;

        std::vector<mblas::Matrix> states_;
    };

  public:
    Encoder(const Weights& model);

    void GetContext(size_t sentInd, const std::vector<size_t>& words,
        EncoderDecoder::SourceContext& Context);

    void GetContextes(const Sentences& sentences, size_t tab,
        EncoderDecoder::SourceContextes &contextes);

  private:
    Embeddings<Weights::EncEmbeddings> embeddings_;
    RNN<Weights::EncForwardGRU> forwardRnn_;
    RNN<Weights::EncBackwardGRU> backwardRnn_;

    typedef std::vector<mblas::Matrix> EmbeddedSentence;
    typedef std::vector<EmbeddedSentence> EmbeddedSentences;

    EmbeddedSentences embeddedSentencesFwd_, embeddedSentencesBck_;

};

}

