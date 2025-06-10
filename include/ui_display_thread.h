#pragma once
#include "video_stream.h"
#include <vector>
#include <memory>

void UiThreadFunc(std::vector<std::shared_ptr<video::VideoStream>> streams);
