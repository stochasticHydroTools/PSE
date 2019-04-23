project(PSEv1)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.2.3
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

include(FindHOOMD.cmake)

# plugins must be built as shared libraries
if (ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif (ENABLE_STATIC)

set(BUILD_SHARED_LIBS on)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${HOOMD_ROOT} CACHE PATH "Installation prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(PYTHON_MODULE_BASE_DIR ${CMAKE_INSTALL_PREFIX})
message(STATUS "Install plugin to: " ${PYTHON_MODULE_BASE_DIR})

# add subdirectories
add_subdirectory(${PROJECT_NAME})
