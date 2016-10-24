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
  Histories ret;

  //cerr << "start batch" << endl;
  // batching
  std::vector<States> batchStates(sentences->size());
  std::vector<States> batchNextStates(sentences->size());
  std::vector<BaseMatrices> batchMatrices(sentences->size());

  for (size_t i = 0; i < sentences->size(); ++i) {
	  States &states = batchStates[i];
	  States &nextStates = batchNextStates[i];
	  BaseMatrices &matrices = batchMatrices[i];

	  states.resize(scorers_.size());
	  nextStates.resize(scorers_.size());
	  matrices.resize(scorers_.size());

	  for (size_t scorerInd = 0; scorerInd < scorers_.size(); scorerInd++) {
	    Scorer &scorer = *scorers_[scorerInd];
	    matrices[scorerInd] = scorer.CreateMatrix();

	    StatePtr &state = states[scorerInd];
	    StatePtr &nextState = nextStates[scorerInd];

	    state.reset(scorer.NewState());
	    nextState.reset(scorer.NewState());
	  }
  }

  // encode
  for (size_t i = 0; i < scorers_.size(); i++) {
    Scorer &scorer = *scorers_[i];
    scorer.SetSources(*sentences);
  }

  // decode
  for (size_t i = 0; i < sentences->size(); ++i) {
    const Sentence *sentence = sentences->at(i);
    States &states = batchStates[i];
    States &nextStates = batchNextStates[i];
    BaseMatrices &matrices = batchMatrices[i];

    for (size_t scorerInd = 0; scorerInd < scorers_.size(); scorerInd++) {
        Scorer &scorer = *scorers_[scorerInd];
        StatePtr &state = states[scorerInd];
        scorer.BeginSentenceState(i, *state);
    }

    History history = Decode(i, sentence, states, nextStates, matrices);
    ret.push_back(history);
  }
  //cerr << "end batch" << endl;

  return ret;
}

History Search::Decode(
		size_t sentInd,
		const Sentence *sentence,
		States &states,
		States &nextStates,
		BaseMatrices &probs) {
  boost::timer::cpu_timer timer;

  //cerr << "probs=" << probs.size() << endl;
  size_t beamSize = God::Get<size_t>("beam-size");
  bool normalize = God::Get<bool>("normalize");

  // @TODO Future: in order to do batch sentence decoding
  // it should be enough to keep track of hypotheses in
  // separate History objects.

  History history;
  Beam prevHyps = { HypothesisPtr(new Hypothesis()) };
  history.Add(prevHyps);

  size_t vocabSize = scorers_[0]->GetVocabSize();

  bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();
  if (filter) {
    vocabSize = MakeFilter(sentence->GetWords(), vocabSize);
  }

  const size_t maxLength = sentence->GetWords().size() * 3;
  do {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      BaseMatrix &prob = *probs[i];
      State &state = *states[i];
      State &nextState = *nextStates[i];

      prob.Resize(beamSize, vocabSize);
      //cerr << "prob=" << prob.GetShape().Debug() << endl;

      scorer.Score(sentInd, state, prob, nextState);
    }

    // Looking at attention vectors
    // mblas::Matrix A;
    // std::static_pointer_cast<EncoderDecoder>(scorers_[0])->GetAttention(A);
    // mblas::debug1(A, 0, sentence.GetWords().size());

    Beam hyps;
    const BaseMatrix &firstMatrix = *probs[0];

    bool returnAlignment = God::Get<bool>("return-alignment");

    firstMatrix.BestHyps(hyps, prevHyps, probs, beamSize, history, scorers_, filterIndices_, returnAlignment);
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

    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);

  } while(history.size() <= maxLength);

  LOG(progress) << "Line " << sentence->GetLine()
	              << ": Search took " << timer.format(3, "%ws");

  for (size_t i = 0; i < scorers_.size(); i++) {
	  Scorer &scorer = *scorers_[i];
	  scorer.CleanUpAfterSentence();

	  BaseMatrix *prob = probs[i];
	  delete prob;
  }

  return history;
}
