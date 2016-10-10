#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>
#include <boost/thread/tss.hpp>
#include <boost/python.hpp>

#include "common/god.h"
#include "common/logging.h"
#include "common/threadpool.h"
#include "common/search.h"
#include "common/printer.h"
#include "common/sentence.h"
#include "common/translation_task.h"

void init(const std::string& options) {
  God::Init(options);
}

boost::python::list translate(boost::python::list& in) {
  size_t threadCount = God::Get<size_t>("threads");
  LOG(info) << "Setting number of threads to " << threadCount;

  ThreadPool pool(threadCount);
  std::vector<std::future<Histories>> results;

  boost::python::list output;
  for(int i = 0; i < boost::python::len(in); ++i) {
    std::string s = boost::python::extract<std::string>(boost::python::object(in[i]));
    Sentence *sentence = new Sentence(i, s);
    Sentences sentences;
    sentences.push_back(sentence);


    results.emplace_back(
        pool.enqueue(
            [=]{ return TranslationTask(sentences, i); }
        )
    );
  }

  size_t lineCounter = 0;

  for (auto&& result : results) {
    std::stringstream ss;
    Printer(result.get(), lineCounter++, ss);
    output.append(ss.str());
  }

  return output;
}

BOOST_PYTHON_MODULE(libamunmt)
{
  boost::python::def("init", init);
  boost::python::def("translate", translate);
}
