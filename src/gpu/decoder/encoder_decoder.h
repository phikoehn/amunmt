#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/loader.h"

#include "common/threadpool.h"
#include <thrust/device_vector.h>

namespace GPU {

class EncoderDecoderState;
class Encoder;
class Decoder;
class Weights;

namespace mblas {
  template <class VecType>
  class TMatrix;
  typedef TMatrix<thrust::device_vector<float>> Matrix;
}

////////////////////////////////////////////
class EncoderDecoder : public Scorer {
  private:
    typedef EncoderDecoderState EDState;

  public:
    typedef std::unique_ptr<mblas::Matrix> SourceContext;
    typedef std::vector<SourceContext> SourceContextes;

    EncoderDecoder(const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const Weights& model);

    virtual void Score(
          size_t sentInd,
          const State& in,
          BaseMatrix& prob,
          State& out) const;

    virtual State* NewState();

    virtual void BeginSentenceState(size_t sentInd, State& state);

    virtual void SetSources(const Sentences& sources);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);

    mblas::Matrix& GetAttention();

    size_t GetVocabSize() const;

    void Filter(const std::vector<size_t>& filterIds);

    virtual BaseMatrix *CreateMatrix();

    virtual ~EncoderDecoder();

  private:
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;

    //SourceContext sourceContext_;
    SourceContextes sourceContextes_;

};

////////////////////////////////////////////
class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config);

    virtual void Load();

    virtual ScorerPtr NewScorer(size_t taskId);

  private:
    std::vector<std::unique_ptr<Weights>> weights_;
};

}
