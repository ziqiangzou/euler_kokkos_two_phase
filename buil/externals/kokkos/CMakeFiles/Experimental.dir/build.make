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

# Utility rule file for Experimental.

# Include the progress variables for this target.
include externals/kokkos/CMakeFiles/Experimental.dir/progress.make

externals/kokkos/CMakeFiles/Experimental:
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/externals/kokkos && /gpfs1l/gpfslocal/pub/cmake/build-3.14.1-gnu54/bin/ctest -D Experimental

Experimental: externals/kokkos/CMakeFiles/Experimental
Experimental: externals/kokkos/CMakeFiles/Experimental.dir/build.make

.PHONY : Experimental

# Rule to build all files generated by this target.
externals/kokkos/CMakeFiles/Experimental.dir/build: Experimental

.PHONY : externals/kokkos/CMakeFiles/Experimental.dir/build

externals/kokkos/CMakeFiles/Experimental.dir/clean:
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/externals/kokkos && $(CMAKE_COMMAND) -P CMakeFiles/Experimental.dir/cmake_clean.cmake
.PHONY : externals/kokkos/CMakeFiles/Experimental.dir/clean

externals/kokkos/CMakeFiles/Experimental.dir/depend:
	cd /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/externals/kokkos /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/externals/kokkos /gpfsdata/zziqiang/new/aLaR/new/euler_kokkos_two_phase_ftac/buil/externals/kokkos/CMakeFiles/Experimental.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : externals/kokkos/CMakeFiles/Experimental.dir/depend
