#pragma once
#include <string>
#include "history.h"

typedef  std::vector<const Sentence*> Sentences;

Histories TranslationTask(const Sentences *sentences, size_t taskCounter);

