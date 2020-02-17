# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /gpfs1l/gpfslocal/pub/cmake/build-3.14.1-gnu54/bin/cmake

# The command to remove a file.
RM = /gpfs1l/gpfslocal/pub/cmake/build-3.14.1-gnu54/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil

# Include any dependencies generated for this target.
include src/all_regime/CMakeFiles/all_regime.dir/depend.make

# Include the progress variables for this target.
include src/all_regime/CMakeFiles/all_regime.dir/progress.make

# Include the compile flags for this target's objects.
include src/all_regime/CMakeFiles/all_regime.dir/flags.make

src/all_regime/CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.o: src/all_regime/CMakeFiles/all_regime.dir/flags.make
src/all_regime/CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.o: ../src/all_regime/SolverHydroAllRegime.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/all_regime/CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.o"
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/src/all_regime && /gpfslocal/pub/gnu/7.3.0/install/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.o -c /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/src/all_regime/SolverHydroAllRegime.cpp

src/all_regime/CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.i"
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/src/all_regime && /gpfslocal/pub/gnu/7.3.0/install/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/src/all_regime/SolverHydroAllRegime.cpp > CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.i

src/all_regime/CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.s"
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/src/all_regime && /gpfslocal/pub/gnu/7.3.0/install/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/src/all_regime/SolverHydroAllRegime.cpp -o CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.s

# Object files for target all_regime
all_regime_OBJECTS = \
"CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.o"

# External object files for target all_regime
all_regime_EXTERNAL_OBJECTS =

src/all_regime/liball_regime.a: src/all_regime/CMakeFiles/all_regime.dir/SolverHydroAllRegime.cpp.o
src/all_regime/liball_regime.a: src/all_regime/CMakeFiles/all_regime.dir/build.make
src/all_regime/liball_regime.a: src/all_regime/CMakeFiles/all_regime.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library liball_regime.a"
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/src/all_regime && $(CMAKE_COMMAND) -P CMakeFiles/all_regime.dir/cmake_clean_target.cmake
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/src/all_regime && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/all_regime.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/all_regime/CMakeFiles/all_regime.dir/build: src/all_regime/liball_regime.a

.PHONY : src/all_regime/CMakeFiles/all_regime.dir/build

src/all_regime/CMakeFiles/all_regime.dir/clean:
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/src/all_regime && $(CMAKE_COMMAND) -P CMakeFiles/all_regime.dir/cmake_clean.cmake
.PHONY : src/all_regime/CMakeFiles/all_regime.dir/clean

src/all_regime/CMakeFiles/all_regime.dir/depend:
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/src/all_regime /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/src/all_regime /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/src/all_regime/CMakeFiles/all_regime.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/all_regime/CMakeFiles/all_regime.dir/depend

