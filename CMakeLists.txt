set(LIBRARY_TARGET_NAME  offset-evaluation-chamfer-distance)

find_package(Torch REQUIRED)

add_library(${LIBRARY_TARGET_NAME} STATIC   include/chamfer_distance/chamfer_distance.h
                                            src/chamfer_distance.cpp
                                            src/chamfer_distance.cu
                                            )

# Library properties
set_target_properties(${LIBRARY_TARGET_NAME} PROPERTIES VERSION       ${${PROJECT_NAME}_VERSION}
                                                        PUBLIC_HEADER "include/chamfer_distance/chamfer_distance.h")

# Include directories
target_include_directories(${LIBRARY_TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                         "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>")

target_compile_features(${LIBRARY_TARGET_NAME} PUBLIC cxx_std_11)

set_target_properties( ${LIBRARY_TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${LIBRARY_TARGET_NAME} PUBLIC "${TORCH_LIBRARIES}")
