#pragma once

#include <memory>
#include "common/scorer.h"
#include "common/sentence.h"
#include "common/history.h"

class Search {
  public:
    Search(size_t threadId);
    virtual ~Search();

    void SetSource(const Sentence *sentence) {
      sentence_ = sentence;
    }

    History Decode();

  private:
    size_t MakeFilter(const Words& srcWords, const size_t vocabSize);
    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;
    const Sentence *sentence_;

};
