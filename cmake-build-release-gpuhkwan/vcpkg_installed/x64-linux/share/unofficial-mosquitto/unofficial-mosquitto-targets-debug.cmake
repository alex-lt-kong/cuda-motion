#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "unofficial::mosquitto::mosquittopp" for configuration "Debug"
set_property(TARGET unofficial::mosquitto::mosquittopp APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(unofficial::mosquitto::mosquittopp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/lib/libmosquittopp_static.a"
  )

list(APPEND _cmake_import_check_targets unofficial::mosquitto::mosquittopp )
list(APPEND _cmake_import_check_files_for_unofficial::mosquitto::mosquittopp "${_IMPORT_PREFIX}/debug/lib/libmosquittopp_static.a" )

# Import target "unofficial::mosquitto::mosquitto" for configuration "Debug"
set_property(TARGET unofficial::mosquitto::mosquitto APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(unofficial::mosquitto::mosquitto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/lib/libmosquitto_static.a"
  )

list(APPEND _cmake_import_check_targets unofficial::mosquitto::mosquitto )
list(APPEND _cmake_import_check_files_for_unofficial::mosquitto::mosquitto "${_IMPORT_PREFIX}/debug/lib/libmosquitto_static.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
