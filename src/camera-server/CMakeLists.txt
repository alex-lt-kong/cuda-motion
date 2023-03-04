set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(/usr/local/include/opencv4)

add_executable(main main.cpp)
add_library(deviceManager deviceManager.cpp)
add_library(motionDetector motionDetector.cpp)
add_library(logger logger.cpp)

target_link_libraries(motionDetector deviceManager opencv_highgui opencv_imgcodecs opencv_video opencv_videoio opencv_imgproc opencv_core pthread)
target_link_libraries(main deviceManager logger motionDetector)


#main: ./src/main.cpp deviceManager.o motionDetector.o logger.o
#	g++ ./src/main.cpp deviceManager.o motionDetector.o logger.o -o motionDetector.out -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_core -pthread $(OPT)
#deviceManager.o: ./src/classes/deviceManager.cpp ./src/classes/deviceManager.h
# g++ -c ./src/classes/deviceManager.cpp -I/usr/local/include/opencv4 $(OPT)

#motionDetector.o: ./src/classes/motionDetector.cpp ./src/classes/motionDetector.h
#	g++ -c ./src/classes/motionDetector.cpp -I/usr/local/include/opencv4 $(OPT)
