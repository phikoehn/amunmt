#pragma once
#include <string>
#include "ThreadPool.h"
#include "common/sentence.h"


namespace Moses2
{

class TranslationTask: public Task
{
public:

  TranslationTask(long translationId, const Sentences *sentences);
  virtual ~TranslationTask();
  virtual void Run();

protected:
  long translationId_;
  const Sentences *sentences_;
};

}

