#pragma once

#include <nlohmann/json.hpp>

#include <csignal>

/* The extern keyword tells the compiler that please dont generate any
definition for it when compiling the source files that include the header.
Without extern, multiple object files that include this header file
will generate its own version of ev_flag, causing the "multiple definition
of `ev_flag';" error. By adding extern, we need to manually add the definition
of ev_flag in one .c/.cpp file. In this particular case, this is done
in main.cpp. */
extern volatile sig_atomic_t ev_flag;

// settings variable is used by multiple threads, possibly concurrently, is
// reading it thread-safe? According to the below response from the library
// author: https://github.com/nlohmann/json/issues/651
// The answer seems to be affirmative.
extern nlohmann::json settings;
