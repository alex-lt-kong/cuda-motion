#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <mosquitto.h>

namespace MatrixPipeline::ProcessingUnit {

class YoloPublishMqtt : public ISynchronousProcessingUnit {
public:
  friend mosquitto *init_mosquitto(void *);
  friend void on_publish(struct mosquitto *, void *, int);

  explicit YoloPublishMqtt(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YoloPublishMqtt") {}
  ~YoloPublishMqtt() override;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  mosquitto *m_mosq{nullptr};
  std::string m_mqtt_broker_url;
  std::string m_mqtt_username;
  std::string m_mqtt_password;
  std::string m_mqtt_ca_file;
  std::string m_mqtt_topic;
};

} // namespace MatrixPipeline::ProcessingUnit