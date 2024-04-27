#include <algorithm>
#include <deque>
#include <stdexcept>

template <class T> class PercentileTracker {
  static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

public:
  PercentileTracker(size_t sample_size) : sample_size(sample_size) {
    refreshStatsCalled = false;
    total_sample_count = 0;
  }

  void addSample(T num) {
    ++total_sample_count;
    samples.push_back(num);
    if (samples.size() > sample_size) {
      samples.pop_front();
    }
  }

  void refreshStats() {
    std::sort(samples.begin(), samples.end());
    refreshStatsCalled = true;
  }

  T getPercentile(double percent) {
    if (samples.empty())
      return -1;
    if (!refreshStatsCalled) {
      throw std::runtime_error("refreshStats() not called before calling "
                               "getPercentile()");
    }

    int index = percent / 100.0 * samples.size() - 1;
    return samples[index];
  }

  auto sampleCount() { return samples.size(); }

  auto totalSampleCount() { return total_sample_count; }

private:
  // Need to use std::deque to have std::sort() operational
  std::deque<T> samples;
  size_t sample_size;
  size_t total_sample_count;
  bool refreshStatsCalled;
};