add_subdirectory(kokkos)

add_subdirectory(io)

if(USE_MPI)
  add_subdirectory(mpiBasic)
endif(USE_MPI)

# Add build tests
set(config_Serial     -DUSE_MPI=OFF -DKokkos_ENABLE_Serial=ON -DKokkos_ENABLE_OpenMP=OFF -DKokkos_ENABLE_Cuda=OFF)
set(config_OpenMP     -DUSE_MPI=OFF -DKokkos_ENABLE_Serial=ON -DKokkos_ENABLE_OpenMP=ON  -DKokkos_ENABLE_Cuda=OFF)
set(config_Cuda       -DUSE_MPI=OFF -DKokkos_ENABLE_Serial=ON -DKokkos_ENABLE_OpenMP=OFF -DKokkos_ENABLE_Cuda=ON )
set(config_Serial_MPI -DUSE_MPI=ON  -DKokkos_ENABLE_Serial=ON -DKokkos_ENABLE_OpenMP=OFF -DKokkos_ENABLE_Cuda=OFF)
set(config_OpenMP_MPI -DUSE_MPI=ON  -DKokkos_ENABLE_Serial=ON -DKokkos_ENABLE_OpenMP=ON  -DKokkos_ENABLE_Cuda=OFF)
set(config_Cuda_MPI   -DUSE_MPI=ON  -DKokkos_ENABLE_Serial=ON -DKokkos_ENABLE_OpenMP=OFF -DKokkos_ENABLE_Cuda=ON )

set(env_Serial)
set(env_OpenMP)
set(env_Cuda       CXX=${NVCC_WRAPPER})
set(env_Serial_MPI CXX=mpicxx)
set(env_OpenMP_MPI CXX=mpicxx)
set(env_Cuda_MPI   CXX=mpicxx OMPI_CXX=${NVCC_WRAPPER})

function(do_test arg)
  set(config_loc ${config_${arg}})
  set(env_loc    ${env_${arg}})
  add_test(
    NAME config_${arg}_test
    COMMAND ${CMAKE_COMMAND} -E env ${env_loc} ${CMAKE_COMMAND} ${config_loc} -H./.. -Bbuild_${arg}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  add_test(
    NAME build_${arg}_test
    COMMAND ${CMAKE_COMMAND} -E env ${env_loc} ${CMAKE_COMMAND} --build . --target euler_kokkos --clean-first -- -j2
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/build_${arg})
  set_tests_properties(build_${arg}_test PROPERTIES DEPENDS config_${arg}_test)
endfunction(do_test)

do_test(Serial)
do_test(OpenMP)
do_test(Cuda)
do_test(Serial_MPI)
do_test(OpenMP_MPI)
do_test(Cuda_MPI)
