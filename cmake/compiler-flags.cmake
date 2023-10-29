###################################################################################################
## COMPILER FLAGS #################################################################################
###################################################################################################

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)

#
# Generic flags
#
add_compile_options("-Wall")
add_compile_options("-Wextra")
add_compile_options("-pedantic")
# Silence many OATPP's marco-related warnings
add_compile_options("-Wno-gnu-zero-variadic-macro-arguments")
add_compile_options("-g3")

  


#
# C/C++ Warning Flags
# See https://github.com/vadz/gcc-warnings-tools for gcc flags and versions
#
if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  add_compile_options("-Wcast-align")
  add_compile_options("-Wconversion")
  add_compile_options("-Wdouble-promotion")
  add_compile_options("-Winvalid-pch")
  add_compile_options("-Wmissing-declarations")
  add_compile_options("-Wmissing-format-attribute")
  add_compile_options("-Wmissing-include-dirs")
  add_compile_options("-Wpointer-arith")
  add_compile_options("-Wredundant-decls")
  add_compile_options("-Wshadow")
  add_compile_options("-Wsign-conversion")
  #add_compile_options("-Wsuggest-attribute=const")
  add_compile_options("-Wsuggest-attribute=noreturn")
  #add_compile_options("-Wsuggest-attribute=pure")
  add_compile_options("-Wswitch-default")
  add_compile_options("-Wswitch-enum")
  add_compile_options("-Wtype-limits")
  add_compile_options("-Wundef")
  add_compile_options("-Wuninitialized")
  add_compile_options("-Wunknown-pragmas")
  add_compile_options("-Wunsafe-loop-optimizations")
  add_compile_options("-Wunused-but-set-parameter")
  add_compile_options("-Wunused-but-set-variable")
  add_compile_options("-Wunused-function")
  add_compile_options("-Wunused")
  add_compile_options("-Wunused-label")
  add_compile_options("-Wunused-macros")
  #add_compile_options("-Wunused-parameter")
  add_compile_options("-Wunused-result")
  add_compile_options("-Wunused-value")
  add_compile_options("-Wunused-variable")

  add_compile_options("-Wunused-local-typedefs")

  add_compile_options("-Wformat=2")
  add_compile_options("-Wsuggest-attribute=format")

  add_compile_options("-Wformat-signedness")
  #add_compile_options("-Wsuggest-final-methods")
  #add_compile_options("-Wsuggest-final-types")

  add_compile_options("-Wduplicated-cond")
  add_compile_options("-Wlogical-op")
  add_compile_options("-Wnull-dereference")

  add_compile_options("-Wduplicated-branches")
  add_compile_options("-Wformat-overflow=2")
  add_compile_options("-Wformat-truncation=2")

  #add_compile_options("-Wcast-align=strict")
  add_compile_options("-Wsuggest-attribute=cold")
  add_compile_options("-Wsuggest-attribute=malloc")

  add_compile_options("-Warith-conversion")

  add_compile_options("-ftrivial-auto-var-init=zero")
  add_compile_options("-Warray-compare")
  add_compile_options("-Wbidi-chars=unpaired,ucn")
  add_compile_options("-Winfinite-recursion")
  add_compile_options("-Wopenacc-parallelism")
  add_compile_options("-Wtrivial-auto-var-init")
endif (CMAKE_CXX_COMPILER_ID MATCHES GNU)


#
# Allow the linker to remove unused data and functions
#
if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  add_compile_options("-fdata-sections")
  add_compile_options("-ffunction-sections")
  add_compile_options("-fno-common")
  add_compile_options("-Wl,--gc-sections")
endif (CMAKE_CXX_COMPILER_ID MATCHES GNU)


#
# Hardening flags
# See https://developers.redhat.com/blog/2018/03/21/compiler-and-linker-flags-gcc
#
if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
add_compile_options("-D_GLIBCXX_ASSERTIONS")
  add_compile_options("-fasynchronous-unwind-tables")
  add_compile_options("-fexceptions")
  add_compile_options("-fstack-clash-protection")
  add_compile_options("-fstack-protector-strong")
  add_compile_options("-grecord-gcc-switches")
  # Issue 872: https://github.com/oatpp/oatpp/issues/872
  # -fcf-protection is supported only on x86 GNU/Linux per this gcc doc:
  # https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html#index-fcf-protection
  # add_compile_options("-fcf-protection")
  add_compile_options("-pipe")
  add_compile_options("-Werror=format-security")
  add_compile_options("-Wno-format-nonliteral")
  add_compile_options("-fPIE")
  add_compile_options("-Wl,-z,defs")
  add_compile_options("-Wl,-z,now")
  add_compile_options("-Wl,-z,relro")
endif (CMAKE_CXX_COMPILER_ID MATCHES GNU)


#
# Sanitize flags
#
#if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
#  add_compile_options("-fsanitize=address")
#  add_compile_options("-fsanitize=thread")
#endif (CMAKE_CXX_COMPILER_ID MATCHES GNU)