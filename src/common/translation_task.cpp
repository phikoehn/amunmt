#include <boost/thread/tss.hpp>
#include "translation_task.h"
#include "search.h"

Histories TranslationTask(const Sentences *sentences, size_t taskCounter) {

#ifdef __APPLE__
  static boost::thread_specific_ptr<Search> s_search;
  Search *search = s_search.get();

  if(search == NULL) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search = new Search(taskCounter);
    s_search.reset(search);
  }
#else
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search(taskCounter));
  }
#endif

  return search->Process(sentences);
}
