#include <sstream>
#include <boost/thread/tss.hpp>
#include "translation_task.h"
#include "search.h"
#include "printer.h"
#include "god.h"
#include "output_collector.h"

void TranslationTask(const Sentences *sentences, size_t taskCounter) {
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search(taskCounter));
  }

  assert(sentences->size());
  Histories histories = search->Decode(*sentences);

  std::stringstream strm;
  Printer(histories, taskCounter, strm);
  OutputCollector &outputCollector = God::GetOutputCollector();
  outputCollector.Write(taskCounter, strm.str());

  delete sentences;
}
