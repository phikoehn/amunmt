#include "TranslationTask.h"
#include "OutputCollector.h"
#include "common/sentence.h"
#include "common/search.h"
#include "common/history.h"
#include "common/printer.h"

using namespace std;

namespace Moses2
{

TranslationTask::TranslationTask(long translationId, const Sentences *sentences)
:translationId_(translationId)
,sentences_(sentences)
{

}

TranslationTask::~TranslationTask()
{
  delete sentences_;
}

void TranslationTask::Run()
{
  Search search(translationId_);
  Histories histories = search.Decode(*sentences_);

  for (History& history : histories) {
    Printer(history, translationId_, std::cout);
  }
}

}

