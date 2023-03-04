# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/bin/motion-detector

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/bin/motion-detector/build

# Include any dependencies generated for this target.
include src/camera-server/CMakeFiles/deviceManager.dir/depend.make

# Include the progress variables for this target.
include src/camera-server/CMakeFiles/deviceManager.dir/progress.make

# Include the compile flags for this target's objects.
include src/camera-server/CMakeFiles/deviceManager.dir/flags.make

src/camera-server/CMakeFiles/deviceManager.dir/deviceManager.cpp.o: src/camera-server/CMakeFiles/deviceManager.dir/flags.make
src/camera-server/CMakeFiles/deviceManager.dir/deviceManager.cpp.o: ../src/camera-server/deviceManager.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/bin/motion-detector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/camera-server/CMakeFiles/deviceManager.dir/deviceManager.cpp.o"
	cd /root/bin/motion-detector/build/src/camera-server && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deviceManager.dir/deviceManager.cpp.o -c /root/bin/motion-detector/src/camera-server/deviceManager.cpp

src/camera-server/CMakeFiles/deviceManager.dir/deviceManager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deviceManager.dir/deviceManager.cpp.i"
	cd /root/bin/motion-detector/build/src/camera-server && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/bin/motion-detector/src/camera-server/deviceManager.cpp > CMakeFiles/deviceManager.dir/deviceManager.cpp.i

src/camera-server/CMakeFiles/deviceManager.dir/deviceManager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deviceManager.dir/deviceManager.cpp.s"
	cd /root/bin/motion-detector/build/src/camera-server && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/bin/motion-detector/src/camera-server/deviceManager.cpp -o CMakeFiles/deviceManager.dir/deviceManager.cpp.s

# Object files for target deviceManager
deviceManager_OBJECTS = \
"CMakeFiles/deviceManager.dir/deviceManager.cpp.o"

# External object files for target deviceManager
deviceManager_EXTERNAL_OBJECTS =

src/camera-server/libdeviceManager.a: src/camera-server/CMakeFiles/deviceManager.dir/deviceManager.cpp.o
src/camera-server/libdeviceManager.a: src/camera-server/CMakeFiles/deviceManager.dir/build.make
src/camera-server/libdeviceManager.a: src/camera-server/CMakeFiles/deviceManager.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/bin/motion-detector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libdeviceManager.a"
	cd /root/bin/motion-detector/build/src/camera-server && $(CMAKE_COMMAND) -P CMakeFiles/deviceManager.dir/cmake_clean_target.cmake
	cd /root/bin/motion-detector/build/src/camera-server && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/deviceManager.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/camera-server/CMakeFiles/deviceManager.dir/build: src/camera-server/libdeviceManager.a

.PHONY : src/camera-server/CMakeFiles/deviceManager.dir/build

src/camera-server/CMakeFiles/deviceManager.dir/clean:
	cd /root/bin/motion-detector/build/src/camera-server && $(CMAKE_COMMAND) -P CMakeFiles/deviceManager.dir/cmake_clean.cmake
.PHONY : src/camera-server/CMakeFiles/deviceManager.dir/clean

src/camera-server/CMakeFiles/deviceManager.dir/depend:
	cd /root/bin/motion-detector/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/bin/motion-detector /root/bin/motion-detector/src/camera-server /root/bin/motion-detector/build /root/bin/motion-detector/build/src/camera-server /root/bin/motion-detector/build/src/camera-server/CMakeFiles/deviceManager.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/camera-server/CMakeFiles/deviceManager.dir/depend

