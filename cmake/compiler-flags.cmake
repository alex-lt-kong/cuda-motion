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
add_compile_options("-O2")
#add_compile_options("-g3")

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
