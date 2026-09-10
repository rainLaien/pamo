#pragma once
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace CadMesh {
class RemeshWorkerPool {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<std::function<void()>> queue;
  std::vector<std::thread> workers;
  bool stopping=false;
  void stop(){
    {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
    changed.notify_all();
    for(auto &worker:workers)if(worker.joinable())worker.join();
  }
public:
  explicit RemeshWorkerPool(std::size_t count){
    try{for(std::size_t i=0;i<count;++i)workers.emplace_back([this]{
      for(;;){std::function<void()> job;
        {std::unique_lock<std::mutex> lock(mutex);
          changed.wait(lock,[this]{return stopping || !queue.empty();});
          if(queue.empty()){if(stopping)return;else continue;}
          job=std::move(queue.front());queue.pop_front();}
        job();
      }
    });}catch(...){stop();throw;}
  }
  ~RemeshWorkerPool(){stop();}
  RemeshWorkerPool(const RemeshWorkerPool&)=delete;
  RemeshWorkerPool& operator=(const RemeshWorkerPool&)=delete;
  template<class F> auto submit(F work)->std::future<decltype(work())>{
    using Result=decltype(work());
    auto job=std::make_shared<std::packaged_task<Result()>>(std::move(work));
    auto future=job->get_future();
    {std::lock_guard<std::mutex> lock(mutex);queue.emplace_back([job]{(*job)();});}
    changed.notify_one();return future;
  }
};
}
