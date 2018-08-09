# Maintainer: Andrew M. Fiore

set(COMPONENT_NAME PSEv1)

set(_${COMPONENT_NAME}_sources
    module.cc
    Stokes.cc
    ShearFunction.cc
    ShearFunctionWrap.cc
    SpecificShearFunction.cc
    VariantShearFunction.cc
    )

set(_${COMPONENT_NAME}_cu_sources
    Stokes.cu
    Brownian.cu
    Helper.cu
    Mobility.cu
    )

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${COMPONENT_NAME}_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module (_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_CUDA_GENERATED_FILES} NO_EXTRAS)
if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# Find additional libraries to be linked for the plugin
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
set( LAPACK_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES}  ${BLAS_LIBRARIES} )

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME} PRIVATE ${HOOMD_LIBRARIES} ${LAPACK_LIBRARIES})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_${COMPONENT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_${COMPONENT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

fix_cudart_rpath(_${COMPONENT_NAME})

# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/${COMPONENT_NAME}/${file}"
    )
ENDMACRO(copy_file)

set(files
    __init__.py
    integrate.py
    shear_function.py
    variant.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${files})

if (BUILD_TESTING)
    add_subdirectory(test-py)
endif()
