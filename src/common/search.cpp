#include <boost/timer/timer.hpp>
#include "common/search.h"
#include "common/god.h"
#include "common/filter.h"
#include "common/base_matrix.h"

using namespace std;

Search::Search(size_t threadId)
  : scorers_(God::GetScorers(threadId)) {}

size_t Search::MakeFilter(const Words& srcWords, const size_t vocabSize) {
  filterIndices_ = God::GetFilter().GetFilteredVocab(srcWords, vocabSize);
  for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->Filter(filterIndices_);
  }
  return filterIndices_.size();
}

Search::~Search()
{
}

Histories Search::Process(const Sentences *sentences) {
  size_t numSentences = sentences->size();
  size_t beamSize = God::Get<size_t>("beam-size");

  Histories histories;
  histories.resize(numSentences);

  //cerr << "start batch" << endl;
  // batching
  std::vector<States> batchStates(numSentences);
  std::vector<States> batchNextStates(numSentences);
  std::vector<BaseMatrices> batchMatrices(numSentences);

  for (size_t i = 0; i < numSentences; ++i) {
	  States &states = batchStates[i];
	  States &nextStates = batchNextStates[i];
	  BaseMatrices &matrices = batchMatrices[i];

	  states.resize(scorers_.size());
	  nextStates.resize(scorers_.size());
	  matrices.resize(scorers_.size());

    Scorer &scorer = *scorers_[0];
    matrices[0] = scorer.CreateMatrix();

    StatePtr &state = states[0];
    StatePtr &nextState = nextStates[0];

    state.reset(scorer.NewState());
    nextState.reset(scorer.NewState());
  }

  // encode
  Scorer &scorer = *scorers_[0];
  scorer.SetSources(*sentences);

  // decode
  for (size_t i = 0; i < numSentences; ++i) {
    const Sentence *sentence = sentences->at(i);
    States &states = batchStates[i];
    States &nextStates = batchNextStates[i];
    BaseMatrices &matrices = batchMatrices[i];

    Scorer &scorer = *scorers_[0];
    StatePtr &state = states[0];
    StatePtr &nextState = nextStates[0];
    scorer.BeginSentenceState(i, *state);

    History &history = histories[i];


    BaseMatrix *prob = matrices[0];
    size_t vocabSize = scorers_[0]->GetVocabSize();
    prob->Resize(beamSize, vocabSize);

    Decode(i, sentence, *state, *nextState, prob, history);

    delete prob;
  }
  //cerr << "end batch" << endl;

  return histories;
}

void Search::Decode(
		size_t sentInd,
		const Sentence *sentence,
		State &state,
		State &nextState,
		BaseMatrix *prob,
		History &history) {

  boost::timer::cpu_timer timer;

  //cerr << "probs=" << probs.size() << endl;
  bool normalize = God::Get<bool>("normalize");
  size_t beamSize = God::Get<size_t>("beam-size");
  size_t vocabSize = scorers_[0]->GetVocabSize();

  assert(scorers_.size() == 1);
  Scorer &scorer = *scorers_[0];

  //cerr << "prob=" << prob.GetShape().Debug() << endl;


  // @TODO Future: in order to do batch sentence decoding
  // it should be enough to keep track of hypotheses in
  // separate History objects.

  Beam prevHyps = { HypothesisPtr(new Hypothesis()) };
  history.Add(prevHyps);

  /*
  bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();
  if (filter) {
    vocabSize = MakeFilter(sentence->GetWords(), vocabSize);
  }
  */

  const size_t maxLength = sentence->GetWords().size() * 3;
  do {
    scorer.Score(sentInd, state, *prob, nextState);

    // Looking at attention vectors
    // mblas::Matrix A;
    // std::static_pointer_cast<EncoderDecoder>(scorers_[0])->GetAttention(A);
    // mblas::debug1(A, 0, sentence.GetWords().size());

    Beam hyps;

    bool returnAlignment = God::Get<bool>("return-alignment");

    prob->BestHyps(hyps, prevHyps, beamSize, scorers_, filterIndices_, returnAlignment);
    history.Add(hyps, history.size() == maxLength);

    Beam survivors;
    for (auto h : hyps) {
      if (h->GetWord() != EOS) {
        survivors.push_back(h);
      }
    }
    beamSize = survivors.size();
    if (beamSize == 0) {
      break;
    }


    scorer.AssembleBeamState(nextState, survivors, state);

    prevHyps.swap(survivors);

  } while(history.size() <= maxLength);

  LOG(progress) << "Line " << sentence->GetLine()
	              << ": Search took " << timer.format(3, "%ws");

  // cleanup
  scorer.CleanUpAfterSentence();

}
