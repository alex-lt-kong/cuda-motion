CC = gcc
CXX = g++
CFLAGS = -O2 -Wall -Wc++-compat -pedantic -Wextra
CXXFLAGS = -O2 -Wall -pedantic -Wextra
LDFLAGS_SHM = -lrt -lpthread
LDFLAGS_CUDA_VIDEO_WRITER = -lopencv_videoio -lopencv_core -lopencv_cudacodec
LDFLAGS_NATIVE_VIDEO_WRITER = -lopencv_videoio -lopencv_core
INC = -I/usr/local/include/opencv4/ -I/usr/include/opencv4/

# TARGETS = shm-reader test-video-writer
# Directory to store the built targets
BUILD_DIR = build
# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

main: $(BUILD_DIR)/shm-reader $(BUILD_DIR)/cuda-video-writer $(BUILD_DIR)/native-video-writer

$(BUILD_DIR)/shm-reader: shm-reader.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(INC) $(CFLAGS) -o $@ $< $(LDFLAGS_SHM)

$(BUILD_DIR)/cuda-video-writer: cuda-video-writer.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(INC) $(CXXFLAGS) -o $@ $< $(LDFLAGS_VIDEO_WRITER)

$(BUILD_DIR)/native-video-writer: native-video-writer.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(INC) $(CXXFLAGS) -o $@ $< $(LDFLAGS_NATIVE_VIDEO_WRITER)

.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)
