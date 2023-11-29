#include <algorithm>
#include <deque>
#include <stdexcept>

template <class T> class PercentileTracker {
public:
  PercentileTracker(size_t sample_size) : sample_size(sample_size) {
    refreshStatsCalled = false;
  }

  void addNumber(T num) {
    data.push_back(num);
    if (data.size() > sample_size) {
      data.pop_front();
    }
  }

  void refreshStats() {
    std::sort(data.begin(), data.end());
    refreshStatsCalled = true;
  }

  double getPercentile(double percent) {
    if (data.empty())
      return -1;
    if (!refreshStatsCalled) {
      throw std::runtime_error("refreshStats() not called before calling "
                               "getPercentile()");
    }

    int index = percent / 100.0 * data.size() - 1;
    return data[index];
  }

  auto sampleCount() { return data.size(); }

private:
  std::deque<T> data;
  size_t sample_size;
  bool refreshStatsCalled;
};