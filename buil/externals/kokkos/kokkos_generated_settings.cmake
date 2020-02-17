#Global Settings used to generate this library
set(KOKKOS_PATH /usr/local CACHE FILEPATH "Kokkos installation path" FORCE)
set(KOKKOS_GMAKE_DEVICES "OpenMP,Serial" CACHE STRING "Kokkos devices list" FORCE)
set(KOKKOS_GMAKE_ARCH "None" CACHE STRING "Kokkos architecture flags" FORCE)
set(KOKKOS_DEBUG_CMAKE ON CACHE BOOL "Kokkos debug enabled ?" FORCE)
set(KOKKOS_GMAKE_USE_TPLS "librt" CACHE STRING "Kokkos templates list" FORCE)
set(KOKKOS_CXX_STANDARD c++11 CACHE STRING "Kokkos C++ standard" FORCE)
set(KOKKOS_GMAKE_OPTIONS "disable_dualview_modify_check" CACHE STRING "Kokkos options" FORCE)
set(KOKKOS_GMAKE_CUDA_OPTIONS "" CACHE STRING "Kokkos Cuda options" FORCE)
set(KOKKOS_GMAKE_TPL_INCLUDE_DIRS "" CACHE STRING "Kokkos TPL include directories" FORCE)
set(KOKKOS_GMAKE_TPL_LIBRARY_DIRS "" CACHE STRING "Kokkos TPL library directories" FORCE)
set(KOKKOS_GMAKE_TPL_LIBRARY_NAMES " dl rt" CACHE STRING "Kokkos TPL library names" FORCE)
if(NOT DEFINED ENV{NVCC_WRAPPER})
set(NVCC_WRAPPER /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/bin/nvcc_wrapper CACHE FILEPATH "Path to command nvcc_wrapper" FORCE)
else()
  set(NVCC_WRAPPER $ENV{NVCC_WRAPPER} CACHE FILEPATH "Path to command nvcc_wrapper")
endif()

#Source and Header files of Kokkos relative to KOKKOS_PATH
set(KOKKOS_HEADERS /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_AnonymousSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Array.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Atomic.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Complex.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Concepts.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_CopyViews.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Core_fwd.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Core.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Crs.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Cuda.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_CudaSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_ExecPolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/KokkosExp_MDRangePolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_HBWSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_HostSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_hwloc.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Layout.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Macros.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_MasterLock.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_MemoryPool.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_MemoryTraits.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_NumericTraits.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_OpenMP.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_OpenMPTarget.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_OpenMPTargetSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Pair.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Parallel.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Parallel_Reduce.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Profiling_ProfileSection.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Qthreads.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_ROCm.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_ROCmSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_ScratchSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Serial.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_TaskPolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_TaskScheduler.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Threads.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Timer.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_UniqueToken.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Vectorization.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_View.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_WorkGraphPolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_AnalyzePolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Assembly.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Compare_Exchange_Strong.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Decrement.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Exchange.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Fetch_Add.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Fetch_And.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Fetch_Or.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Fetch_Sub.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Generic.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Increment.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_View.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Windows.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_BitOps.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ClockTic.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ConcurrentBitset.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_CPUDiscovery.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Error.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/KokkosExp_Host_IterateTile.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/KokkosExp_ViewMapping.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_FunctorAdapter.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_FunctorAnalysis.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_HostBarrier.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_HostThreadTeam.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Memory_Fence.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_OldMacros.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_PhysicalLayout.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Profiling_DeviceInfo.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Profiling_Interface.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Serial_Task.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Serial_WorkGraphPolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_SharedAlloc.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Spinwait.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_StaticAssert.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Tags.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_TaskQueue.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_TaskQueue_impl.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Timer.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Traits.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Utilities.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ViewArray.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ViewCtor.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ViewMapping.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ViewTile.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Volatile_Load.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_Bitset.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_DualView.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_DynamicView.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_DynRankView.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_ErrorReporter.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_Functional.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_ScatterView.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_StaticCrsGraph.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_UnorderedMap.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_Vector.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_Bitset_impl.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_Functional_impl.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_StaticCrsGraph_factory.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_UnorderedMap_impl.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/algorithms/src/Kokkos_Random.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/algorithms/src/Kokkos_Sort.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Exec.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Parallel.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Task.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Team.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_WorkGraphPolicy.hpp CACHE STRING "Kokkos headers list" FORCE)
set(KOKKOS_HEADERS_IMPL CACHE STRING "Kokkos headers impl list" FORCE)
set(KOKKOS_HEADERS_CUDA CACHE STRING "Kokkos headers Cuda list" FORCE)
set(KOKKOS_HEADERS_OPENMP /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Exec.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Parallel.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Task.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Team.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_WorkGraphPolicy.hpp CACHE STRING "Kokkos headers OpenMP list" FORCE)
set(KOKKOS_HEADERS_ROCM CACHE STRING "Kokkos headers ROCm list" FORCE)
set(KOKKOS_HEADERS_THREADS CACHE STRING "Kokkos headers Threads list" FORCE)
set(KOKKOS_HEADERS_QTHREADS CACHE STRING "Kokkos headers QThreads list" FORCE)
set(KOKKOS_SRC /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Core.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_CPUDiscovery.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Error.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ExecPolicy.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_HostBarrier.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_HostSpace.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_HostThreadTeam.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_hwloc.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_MemoryPool.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Profiling_Interface.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Serial.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Serial_Task.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_SharedAlloc.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Spinwait.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_UnorderedMap_impl.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Exec.cpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Task.cpp CACHE STRING "Kokkos source list" FORCE)

#Variables used in application Makefiles
set(KOKKOS_OS Linux CACHE STRING "" FORCE)
set(KOKKOS_CPP_DEPENDS KokkosCore_config.h /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_AnonymousSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Array.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Atomic.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Complex.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Concepts.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_CopyViews.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Core_fwd.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Core.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Crs.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Cuda.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_CudaSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_ExecPolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/KokkosExp_MDRangePolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_HBWSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_HostSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_hwloc.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Layout.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Macros.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_MasterLock.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_MemoryPool.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_MemoryTraits.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_NumericTraits.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_OpenMP.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_OpenMPTarget.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_OpenMPTargetSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Pair.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Parallel.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Parallel_Reduce.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Profiling_ProfileSection.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Qthreads.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_ROCm.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_ROCmSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_ScratchSpace.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Serial.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_TaskPolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_TaskScheduler.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Threads.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Timer.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_UniqueToken.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_Vectorization.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_View.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/Kokkos_WorkGraphPolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_AnalyzePolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Assembly.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Compare_Exchange_Strong.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Decrement.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Exchange.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Fetch_Add.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Fetch_And.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Fetch_Or.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Fetch_Sub.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Generic.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Increment.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_View.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Atomic_Windows.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_BitOps.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ClockTic.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ConcurrentBitset.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_CPUDiscovery.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Error.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/KokkosExp_Host_IterateTile.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/KokkosExp_ViewMapping.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_FunctorAdapter.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_FunctorAnalysis.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_HostBarrier.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_HostThreadTeam.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Memory_Fence.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_OldMacros.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_PhysicalLayout.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Profiling_DeviceInfo.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Profiling_Interface.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Serial_Task.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Serial_WorkGraphPolicy.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_SharedAlloc.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Spinwait.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_StaticAssert.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Tags.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_TaskQueue.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_TaskQueue_impl.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Timer.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Traits.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Utilities.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ViewArray.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ViewCtor.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ViewMapping.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_ViewTile.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/impl/Kokkos_Volatile_Load.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_Bitset.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_DualView.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_DynamicView.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_DynRankView.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_ErrorReporter.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_Functional.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_ScatterView.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_StaticCrsGraph.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_UnorderedMap.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/Kokkos_Vector.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_Bitset_impl.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_Functional_impl.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_StaticCrsGraph_factory.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src/impl/Kokkos_UnorderedMap_impl.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/algorithms/src/Kokkos_Random.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/algorithms/src/Kokkos_Sort.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Exec.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Parallel.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Task.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_Team.hpp /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src/OpenMP/Kokkos_OpenMP_WorkGraphPolicy.hpp CACHE STRING "" FORCE)
set(KOKKOS_LINK_DEPENDS libkokkos.a CACHE STRING "" FORCE)
set(KOKKOS_CXXFLAGS --std=c++11 -fopenmp CACHE STRING "" FORCE)
set(KOKKOS_CPPFLAGS -I./ -I/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src -I/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src -I/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/algorithms/src CACHE STRING "" FORCE)
set(KOKKOS_LDFLAGS -L/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/externals/kokkos -fopenmp CACHE STRING "" FORCE)
set(KOKKOS_LIBS -lkokkos -ldl -lrt CACHE STRING "" FORCE)
set(KOKKOS_EXTRA_LIBS -ldl -lrt CACHE STRING "" FORCE)
set(KOKKOS_LINK_FLAGS -fopenmp CACHE STRING "extra flags to the link step (e.g. OpenMP)" FORCE)

#Internal settings which need to propagated for Kokkos examples
set(KOKKOS_INTERNAL_USE_CUDA 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_OPENMP 1 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_PTHREADS 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_SERIAL 1 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_ROCM 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_QTHREADS 0 CACHE STRING "" FORCE)
set(KOKKOS_CXX_FLAGS --std=c++11 -fopenmp)
set(KOKKOS_CPP_FLAGS -I./ -I/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/core/src -I/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/containers/src -I/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos/algorithms/src)
set(KOKKOS_LD_FLAGS -L/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/externals/kokkos -fopenmp)
set(KOKKOS_LIBS_LIST "-lkokkos -ldl -lrt")
set(KOKKOS_EXTRA_LIBS_LIST "-ldl -lrt")
set(KOKKOS_LINK_FLAGS -fopenmp)
