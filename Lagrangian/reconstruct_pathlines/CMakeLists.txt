project(Reconstruct_Pathlines)

cmake_minimum_required(VERSION 2.8.10)

set(CMAKE_C_COMPILER /usr/local/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/local/bin/g++)
set(CMAKE_C_FLAGS "-g -fopenmp -Ofast")
set(CMAKE_CXX_STANDARD 14)

SET(VTKm_DIR /home/sci/ssane/projects/uncertainty_vis/VTKM/install/lib/cmake/vtkm-1.5)
find_package(VTKm REQUIRED QUIET)


set(VTK_DIR "/home/sci/ssane/packages/VTK-build")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(CGAL QUIET COMPONENTS Core)

if(CGAL_FOUND)
	include(${CGAL_USE_FILE})
	include(CGAL_CreateSingleSourceCGALProgram)
	include_directories (BEFORE "../../include")
else()
	message(STATUS "This program requires the CGAL library, and will not be compiled.")
endif()

add_executable(Reconstruct_Pathlines reconstruct_pathlines.cxx)

  target_include_directories(Reconstruct_Pathlines PRIVATE ${VTKm_INCLUDE_DIRS})
  target_link_libraries(Reconstruct_Pathlines PRIVATE vtkm_cont vtkm_io ${VTKm_LIBRARIES} ${VTK_LIBRARIES} -fopenmp)
#  target_compile_definitions(Reconstruct_Pathlines PRIVATE "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_SERIAL")
  target_compile_options(Reconstruct_Pathlines PRIVATE ${VTKm_COMPILE_OPTIONS})
