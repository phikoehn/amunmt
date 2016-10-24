#pragma once

#include "../mblas/matrix.h"
#include "model.h"
#include "gru.h"

namespace GPU {

class Decoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Matrix& Rows, const std::vector<size_t>& ids) {
          using namespace mblas;
          std::vector<size_t> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_.GetShape().rows)
              id = 1;
          Assemble(Rows, w_.E_, tids);
        }

        size_t GetCols() {
          return w_.E_.GetShape().cols;
        }

        size_t GetRows() const {
          return w_.E_.GetShape().rows;
        }

      private:
        const Weights& w_;
    };

    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel)
        : w_(initModel), gru_(gruModel) {}

        void InitializeState(mblas::Matrix& State,
                             const mblas::Matrix& SourceContext,
                             const size_t batchSize = 1) {
          using namespace mblas;

          Mean(Temp1_, SourceContext);
          Temp2_.Clear();
          Temp2_.Resize(batchSize, SourceContext.GetShape().cols, 0.0);
          BroadcastVec(_1 + _2, Temp2_, Temp1_);
          Prod(State, Temp2_, w_.Wi_);
          BroadcastVec(Tanh(_1 + _2), State, w_.Bi_);
        }

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) {
          //std::cerr << "RNNHidden" << std::endl;
          gru_.GetNextState(NextState, State, Context);
        }

      private:
        const Weights1& w_;
        const GRU<Weights2> gru_;

        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
    };

    template <class Weights>
    class RNNFinal {
      public:
        RNNFinal(const Weights& model)
        : gru_(model) {}

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) {
          //std::cerr << "RNNFinal" << std::endl;
          gru_.GetNextState(NextState, State, Context);
        }

      private:
        const GRU<Weights> gru_;
    };

    template <class Weights>
    class Alignment {
      public:
        Alignment(const Weights& model)
        : w_(model)
        {
          //for(int i = 0; i < 2; ++i) {
          //  cudaStreamCreate(&s_[i]);
          //  cublasCreate(&h_[i]);
          //  cublasSetStream(h_[i], s_[i]);
          //}
        }

        void Init(const mblas::Matrix& SourceContext) {
          using namespace mblas;
          Prod(/*h_[0],*/ SCU_, SourceContext, w_.U_);
        }

        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     const mblas::Matrix& HiddenState,
                                     const mblas::Matrix& SourceContext) {
          using namespace mblas;

          Prod(/*h_[1],*/ Temp2_, HiddenState, w_.W_);
          BroadcastVec(_1 + _2, Temp2_, w_.B_/*, s_[1]*/);

          Copy(Temp1_, SCU_);
          Broadcast(Tanh(_1 + _2), Temp1_, Temp2_);

          //Temp();

          Prod(A_, w_.V_, Temp1_, false, true);

          size_t rows1 = SourceContext.GetShape().rows;
          size_t rows2 = HiddenState.GetShape().rows;
          A_.GetShape().Resize(rows2, rows1); // due to broadcasting above
          Element(_1 + w_.C_(0,0), A_);

          mblas::Softmax(A_);
          Prod(AlignedSourceContext, A_, SourceContext);
        }

        void GetAttention(mblas::Matrix& Attention) {
          mblas::Copy(Attention, A_);
        }

        mblas::Matrix& GetAttention() {
          return A_;
        }

      private:
        const Weights& w_;

        cublasHandle_t h_[2];
        cudaStream_t s_[2];

        mblas::Matrix SCU_;
        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix A_;

        mblas::Matrix Ones_;
        mblas::Matrix Sums_;

        void Temp() {
          using namespace mblas;

          GPU::mblas::Matrix m1(2,3);
           float v = 1;
           for (size_t i = 0; i < 2; ++i) {
             for (size_t j = 0; j < 3; ++j) {
               m1.Set(i, j, v++);
             }
           }

           GPU::mblas::Matrix m2(4,3);
           v = 10;
           for (size_t i = 0; i < 4; ++i) {
             for (size_t j = 0; j < 3; ++j) {
               m2.Set(i, j, v++);
             }
           }
           /*
           std::cerr << "BEFORE m1=" << m1.Debug() << std::endl;
           std::cerr << "BEFORE m2=" << m2.Debug() << std::endl;
		   */
           //Broadcast(_1 + _2, m1, m2);
           Broadcast(_1, m1, m2);
           /*
           std::cerr << "AFTER  m1=" << m1.Debug() << std::endl;
           std::cerr << "AFTER  m2=" << m2.Debug() << std::endl;
		   */
        }
    };

    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model), filtered_(false)
        {
          //for(int i = 0; i < 3; ++i) {
          //  cudaStreamCreate(&s_[i]);
          //  cublasCreate(&h_[i]);
          //  cublasSetStream(h_[i], s_[i]);
          //}
        }

        void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
          using namespace mblas;

          Prod(/*h_[0],*/ T1_, State, w_.W1_);
          Prod(/*h_[1],*/ T2_, Embedding, w_.W2_);
          Prod(/*h_[2],*/ T3_, AlignedSourceContext, w_.W3_);

          BroadcastVec(_1 + _2, T1_, w_.B1_ /*,s_[0]*/);
          BroadcastVec(_1 + _2, T2_, w_.B2_ /*,s_[1]*/);
          BroadcastVec(_1 + _2, T3_, w_.B3_ /*,s_[2]*/);

          //cudaDeviceSynchronize();

          Element(Tanh(_1 + _2 + _3), T1_, T2_, T3_);

          if(!filtered_) {
            Prod(Probs, T1_, w_.W4_);
            BroadcastVec(_1 + _2, Probs, w_.B4_);
          } else {
            Prod(Probs, T1_, FilteredW4_);
            BroadcastVec(_1 + _2, Probs, FilteredB4_);
          }

          // @TODO logsoftmax!
          mblas::Softmax(Probs);
          Element(Log(_1), Probs);
        }

        void Filter(const std::vector<size_t>& ids) {
          filtered_ = true;
          using namespace mblas;

          Matrix TempW4, TempB4;
          Transpose(TempW4, w_.W4_);
          Transpose(TempB4, w_.B4_);

          Assemble(FilteredW4_, TempW4, ids);
          Assemble(FilteredB4_, TempB4, ids);

          Transpose(FilteredW4_);
          Transpose(FilteredB4_);
        }

      private:
        const Weights& w_;

        cublasHandle_t h_[3];
        cudaStream_t s_[3];

        bool filtered_;
        mblas::Matrix FilteredW4_;
        mblas::Matrix FilteredB4_;

        mblas::Matrix T1_;
        mblas::Matrix T2_;
        mblas::Matrix T3_;
    };

  public:
    Decoder(const Weights& model);

    void MakeStep(mblas::Matrix& NextState,
                  mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embeddings,
                  const mblas::Matrix& SourceContext);

    void EmptyState(mblas::Matrix& State,
                    const mblas::Matrix& SourceContext,
                    size_t batchSize = 1);

    void EmptyEmbedding(mblas::Matrix& Embedding,
                        size_t batchSize = 1);

    void Lookup(mblas::Matrix& Embedding,
                const std::vector<size_t>& w);

    void Filter(const std::vector<size_t>& ids);

    void GetAttention(mblas::Matrix& Attention);

    size_t GetVocabSize() const;

    mblas::Matrix& GetAttention();

  private:

    void GetHiddenState(mblas::Matrix& HiddenState,
                        const mblas::Matrix& PrevState,
                        const mblas::Matrix& Embedding);

    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                 const mblas::Matrix& HiddenState,
                                 const mblas::Matrix& SourceContext);

    void GetNextState(mblas::Matrix& State,
                      const mblas::Matrix& HiddenState,
                      const mblas::Matrix& AlignedSourceContext);

    void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext);

  private:
    mblas::Matrix HiddenState_;
    mblas::Matrix AlignedSourceContext_;

    Embeddings<Weights::DecEmbeddings> embeddings_;
    RNNHidden<Weights::DecInit, Weights::DecGRU1> rnn1_;
    RNNFinal<Weights::DecGRU2> rnn2_;
    Alignment<Weights::DecAlignment> alignment_;
    Softmax<Weights::DecSoftmax> softmax_;
};

}
