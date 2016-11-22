#include <boost/thread/tss.hpp>
#include "translation_task.h"
#include "search.h"
#include "printer.h"

Histories TranslationTask(const Sentences *sentences, size_t taskCounter) {
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search(taskCounter));
  }

  assert(sentences->size());
  Histories histories = search->Decode(*sentences);
  Printer(histories, taskCounter, std::cout);

  delete sentences;

  return histories;
}
