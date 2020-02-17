#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "kokkos" for configuration "Release"
set_property(TARGET kokkos APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kokkos PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libkokkos.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS kokkos )
list(APPEND _IMPORT_CHECK_FILES_FOR_kokkos "/usr/local/lib/libkokkos.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
