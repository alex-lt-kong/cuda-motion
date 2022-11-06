OPT = -O3 -Wall

main: ./src/main.cpp deviceManager.o motionDetector.o logger.o
	g++ ./src/main.cpp deviceManager.o motionDetector.o logger.o -o motionDetector.out -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_core -pthread $(OPT)
deviceManager.o: ./src/classes/deviceManager.cpp ./src/classes/deviceManager.h
	g++ -c ./src/classes/deviceManager.cpp -I/usr/local/include/opencv4 $(OPT)

motionDetector.o: ./src/classes/motionDetector.cpp ./src/classes/motionDetector.h
	g++ -c ./src/classes/motionDetector.cpp -I/usr/local/include/opencv4 $(OPT)

logger.o: ./src/classes/logger.cpp ./src/classes/logger.h
	g++ -c ./src/classes/logger.cpp $(OPT)

clean:
	rm *.out *.o
