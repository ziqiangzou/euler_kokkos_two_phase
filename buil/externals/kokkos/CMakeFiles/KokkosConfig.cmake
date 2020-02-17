# - Config file for the Kokkos package
# It defines the following variables
#  Kokkos_INCLUDE_DIRS - include directories for Kokkos
#  Kokkos_LIBRARIES    - libraries to link against

# Compute paths
GET_FILENAME_COMPONENT(Kokkos_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
SET(Kokkos_INCLUDE_DIRS "${Kokkos_CMAKE_DIR}/../../../include/kokkos")

# Our library dependencies (contains definitions for IMPORTED targets)
IF(NOT TARGET kokkos AND NOT Kokkos_BINARY_DIR)
  INCLUDE("${Kokkos_CMAKE_DIR}/KokkosTargets.cmake")
ENDIF()

# These are IMPORTED targets created by KokkosTargets.cmake
SET(Kokkos_LIBRARY_DIRS /usr/local/lib)
SET(Kokkos_LIBRARIES kokkos)
SET(Kokkos_TPL_LIBRARIES -lkokkos;-ldl;-lrt)
