#pragma once

#include <memory>

#include "god.h"
#include "sentence.h"
#include "history.h"

class Search {
  private:
    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;
    const Sentence *sentence_;

  public:
    Search(size_t threadId);
    virtual ~Search();

    size_t MakeFilter(const Words& srcWords, const size_t vocabSize);
    void SetSource(const Sentence *sentence) {
      sentence_ = sentence;
    }

    History Decode();

};
