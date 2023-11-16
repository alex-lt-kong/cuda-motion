option(BUILD_ASAN "Build with AddressSanitizer to detect memory error" OFF)
option(BUILD_UBSAN "Build with UndefinedBehaviorSanitizer to detect undefined behavior" OFF)
# MemorySanitizer and ThreadSanitizer require that all program code is instrumented. This also
# includes any libraries that the program depends on, even libc. 
# option(BUILD_MSAN "Build with MemorySanitizer to detect memory error" OFF)
# option(BUILD_TSAN "Build with ThreadSanitizer to detect data race issues" OFF)

set(COUNTER 0)
set(ALL_OPTIONS BUILD_ASAN;BUILD_MSAN;BUILD_TSAN)
foreach(option IN LISTS ALL_OPTIONS)
    if(${option})
        math(EXPR COUNTER "${COUNTER}+1")
    endif()
endforeach()

if(${COUNTER} GREATER 1)
    message(FATAL_ERROR "Can't enable more than one sanitizer at a time")
endif()


set(SANITIZER_NAME "None")
if(BUILD_ASAN)
    message("-- AddressSanitizer WILL be compiled in as BUILD_ASAN=ON")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
    set(SANITIZER_NAME "AddressSanitizer")
    add_definitions(-DSANITIZER_NAME="${SANITIZER_NAME}")    
else()
    message("-- AddressSanitizer will NOT be compiled in as BUILD_ASAN=OFF")
endif()

if(BUILD_UBSAN)
    message("-- UndefinedBehaviorSanitizer WILL be compiled in as BUILD_UBSAN=ON")
    add_compile_options(-fsanitize=undefined -g)
    add_link_options(-fsanitize=undefined)
    set(SANITIZER_NAME "UndefinedBehaviorSanitizer")
    add_definitions(-DSANITIZER_NAME="${SANITIZER_NAME}")
else()
    message("-- UndefinedBehaviorSanitizer will NOT be compiled in as BUILD_UBSAN=OFF")
endif()


#if(BUILD_MSAN)
#    if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
#        message(FATAL_ERROR "MemorySanitizer is supported by clang only, but "
#            "current compiler is [${CMAKE_C_COMPILER_ID}]")
#    endif()

#    message("-- MemorySanitizer WILL be compiled in as BUILD_MSAN=ON")
#    add_compile_options(-fsanitize=memory -fsanitize-memory-track-origins
#        -fno-omit-frame-pointer -g)
#    add_link_options(-fsanitize=memory)
#else()
#    message("-- MemorySanitizer will NOT be compiled in as BUILD_MSAN=OFF")
#endif()

#if(BUILD_TSAN)
#    message("-- ThreadSanitizer WILL be compiled in as BUILD_TSAN=ON")
#    add_compile_options(-fsanitize=thread -g)
#    add_link_options(-fsanitize=thread)
#    set(SANITIZER_NAME "ThreadSanitizer")
#    add_definitions(-DSANITIZER_NAME="${SANITIZER_NAME}")
#else()
#    message("-- ThreadSanitizer will NOT be compiled in as BUILD_TSAN=OFF")
#endif()
