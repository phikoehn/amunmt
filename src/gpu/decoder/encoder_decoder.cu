#include <iostream>

#include "common/god.h"

#include "encoder_decoder.h"
#include "gpu/mblas/matrix.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"

using namespace std;

namespace GPU {

////////////////////////////////////////////
std::string EncoderDecoderState::Debug() const
{
	return states_.Debug();
}

mblas::Matrix& EncoderDecoderState::GetStates() {
  return states_;
}

mblas::Matrix& EncoderDecoderState::GetEmbeddings() {
  return embeddings_;
}

const mblas::Matrix& EncoderDecoderState::GetStates() const {
  return states_;
}

const mblas::Matrix& EncoderDecoderState::GetEmbeddings() const {
  return embeddings_;
}

////////////////////////////////////////////

EncoderDecoder::EncoderDecoder(const std::string& name,
               const YAML::Node& config,
               size_t tab,
               const Weights& model)
: Scorer(name, config, tab), model_(model),
  encoder_(new Encoder(model_)), decoder_(new Decoder(model_))
{}

void EncoderDecoder::Score(
    size_t sentInd,
    const State& in,
		BaseMatrix& prob,
		State& out) {
  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  mblas::Matrix &probCast = static_cast<mblas::Matrix&>(prob);
  decoder_->MakeStep(edOut.GetStates(), probCast,
                     edIn.GetStates(), edIn.GetEmbeddings(),
                     *sourceContext_);
}

State* EncoderDecoder::NewState() {
  return new EDState();
}

void EncoderDecoder::BeginSentenceState(State& state) {
  EDState& edState = state.get<EDState>();
  decoder_->EmptyState(edState.GetStates(), *sourceContext_, 1);
  decoder_->EmptyEmbedding(edState.GetEmbeddings(), 1);
}

void EncoderDecoder::SetSource(size_t sentInd, const Sentence& source) {
  sourceContext_.reset(new mblas::Matrix());
  //cerr << "SetSource" << source.Debug() << endl;
  encoder_->GetContext(sentInd, source.GetWords(tab_),
                       *sourceContext_);
}

void EncoderDecoder::SetSources(const Sentences& sources)
{
  encoder_->GetContext(sources, tab_, *sourceContext_);
}

void EncoderDecoder::AssembleBeamState(const State& in,
                               const Beam& beam,
                               State& out) {
  std::vector<size_t> beamWords;
  std::vector<size_t> beamStateIds;
  for(auto h : beam) {
     beamWords.push_back(h->GetWord());
     beamStateIds.push_back(h->GetPrevStateIndex());
  }

  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  mblas::Assemble(edOut.GetStates(),
                  edIn.GetStates(), beamStateIds);
  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
}

void EncoderDecoder::GetAttention(mblas::Matrix& Attention) {
  decoder_->GetAttention(Attention);
}

mblas::Matrix& EncoderDecoder::GetAttention() {
  return decoder_->GetAttention();
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<size_t>& filterIds) {
  decoder_->Filter(filterIds);
}

BaseMatrix *EncoderDecoder::CreateMatrix()
{
  mblas::Matrix *ret = new mblas::Matrix();
  return ret;
}

EncoderDecoder::~EncoderDecoder() {}

////////////////////////////////////////////
EncoderDecoderLoader::EncoderDecoderLoader(const std::string name,
                     const YAML::Node& config)
 : Loader(name, config) {}

void EncoderDecoderLoader::Load() {
  std::string path = Get<std::string>("path");
  auto devices = God::Get<std::vector<size_t>>("devices");
  ThreadPool devicePool(devices.size());
  weights_.resize(devices.size());

  size_t i = 0;
  for(auto d : devices) {
    devicePool.enqueue([i, d, &path, this] {
      LOG(info) << "Loading model " << path << " onto gpu" << d;
      cudaSetDevice(d);
      weights_[i].reset(new Weights(path, d));
    });
    ++i;
  }
}

ScorerPtr EncoderDecoderLoader::NewScorer(size_t taskId) {
  size_t i = taskId % weights_.size();
  size_t d = weights_[i]->GetDevice();
  cudaSetDevice(d);
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  return ScorerPtr(new EncoderDecoder(name_, config_,
                                      tab, *weights_[i]));
}

}

