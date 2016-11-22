#pragma once
#include <string>
#include "ThreadPool.h"

class Sentence;

namespace Moses2
{

class TranslationTask: public Task
{
public:

  TranslationTask(const std::string &line, long translationId);
  virtual ~TranslationTask();
  virtual void Run();

protected:
  Sentence *sentence_;
  long translationId_;
};

}

