set (all_regime_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/hydro_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LevelSet2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LevelSet3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroInitFunctors2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroInitFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroAllRegime.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroAllRegime.cpp
  )

add_library(all_regime
  ${all_regime_SRCS}
  )
target_include_directories(all_regime
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/kokkos/core/src
  ${CMAKE_SOURCE_DIR}/kokkos/containers/src
  ${CMAKE_SOURCE_DIR}/kokkos/algorithms/src
  ${CMAKE_BINARY_DIR}/kokkos
  )
target_link_libraries(all_regime
  kokkos
  )
