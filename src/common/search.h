#pragma once

#include <memory>
#include "common/scorer.h"
#include "common/sentence.h"
#include "common/history.h"

class Search {
  public:
    Search(size_t threadId);
    virtual ~Search();

    Histories Process(const Sentences *sentences);

  private:
    size_t MakeFilter(const Words& srcWords, const size_t vocabSize);
    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;

    History Decode(
    		size_t sentInd,
    		const Sentence *sentence,
    		States &states,
    		States &nextStates,
    		BaseMatrices &probs);

};
