file(GLOB _SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB _CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

list(APPEND PLUGIN_SRC ${_SRCS})

if (ENABLE_CUDA)
    CUDA_COMPILE(_CUDA_GENERATED_FILES ${_CUDA_SRCS} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
    list(APPEND PLUGIN_SRC ${_CUDA_GENERATED_FILES})
endif (ENABLE_CUDA)

# Find additional libraries to be linked for the pluging
find_library( LAPACKE_LIBRARIES lapacke PATHS /usr/local/lapack-3.6.0/lib/ )
find_library( LAPACK_LIBRARIES lapack PATHS /usr/local/lapack-3.6.0/lib/ )
find_library( BLAS_LIBRARIES blas PATHS /usr/local/lapack-3.6.0/lib/ )
find_library( CBLAS_LIBRARIES cblas PATHS /usr/local/lapack-3.6.0/lib/ )

if( LAPACKE_LIBRARIES-NOTFOUND )
	message(FATAL_ERROR "lapacke libraries not found")
endif( LAPACKE_LIBRARIES-NOTFOUND )

message(STATUS  "found lapacke libraries: ${LAPACKE_LIBRARIES}")

if( LAPACK_LIBRARIES-NOTFOUND )
	message(FATAL_ERROR "lapack libraries not found")
endif( LAPACK_LIBRARIES-NOTFOUND )

message(STATUS  "found lapack libraries: ${LAPACK_LIBRARIES}")

if( BLAS_LIBRARIES-NOTFOUND )
	message(FATAL_ERROR "blas libraries not found")
endif( BLAS_LIBRARIES-NOTFOUND )

message(STATUS  "found blas libraries: ${BLAS_LIBRARIES}")

if( CBLAS_LIBRARIES-NOTFOUND )
	message(FATAL_ERROR "cblas libraries not found")
endif( CBLAS_LIBRARIES-NOTFOUND )

message(STATUS  "found cblas libraries: ${CBLAS_LIBRARIES}")

set( LAPACK_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES}  ${BLAS_LIBRARIES} )

# build the C++ python module
set(LIB_TARGET "_${PROJECT_NAME}")
add_library(${LIB_TARGET} ${PLUGIN_SRC} ${PYMODULE_BUILD})
target_link_libraries(${LIB_TARGET} ${HOOMD_LIBRARIES} ${LAPACK_LIBRARIES})
# fix up the file name so it will work as a python module
set_target_properties(${LIB_TARGET} PROPERTIES PREFIX "" OUTPUT_NAME "${LIB_TARGET}")
if(APPLE)
    set_target_properties(${LIB_TARGET} PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(${LIB_TARGET})

# install the library to the hoomd_plugins directory
install(TARGETS ${LIB_TARGET}
        LIBRARY DESTINATION hoomd_plugins/${PROJECT_NAME}
        )
