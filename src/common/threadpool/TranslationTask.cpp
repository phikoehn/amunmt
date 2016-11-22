#include "TranslationTask.h"
#include "OutputCollector.h"
#include "common/sentence.h"
#include "common/search.h"
#include "common/history.h"
#include "common/printer.h"

using namespace std;

namespace Moses2
{

TranslationTask::TranslationTask(
		const std::string &line,
		long translationId)
{
  translationId_ = translationId;
  sentence_ = new Sentence(translationId, line);

}

TranslationTask::~TranslationTask()
{
  delete sentence_;
}

void TranslationTask::Run()
{
  Search search(translationId_);
  Sentences sentences;
  sentences.push_back(*sentence_);
  Histories histories = search.Decode(sentences);

  for (History& history : histories) {
    Printer(history, translationId_, std::cout);
  }
}

}

