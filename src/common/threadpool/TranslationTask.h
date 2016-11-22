#pragma once
#include <string>
#include "ThreadPool.h"

namespace Moses2
{

class TranslationTask: public Task
{
public:

  TranslationTask(const std::string &line, long translationId);
  virtual ~TranslationTask();
  virtual void Run();

protected:
};

}

