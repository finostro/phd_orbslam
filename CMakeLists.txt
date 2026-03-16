
cmake_minimum_required(VERSION 3.20)

project(phd_orbslam_minimal LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")





set(MARCH_NATIVE_CONFIGS "$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>")




# ROS2 stuff
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)


# OpenMP
if(APPLE AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
  set(USE_OPENMP_DEFAULT OFF)
else()
  set(USE_OPENMP_DEFAULT ON)
endif()
option(USE_OPENMP "Use OpenMP to compile multithreaded versions of SLAM algorithms." ${USE_OPENMP_DEFAULT})
if(USE_OPENMP)
  find_package(OpenMP)
endif()


#Threads

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME ON)
cmake_policy(SET CMP0167 NEW)
find_package(Boost CONFIG REQUIRED COMPONENTS timer chrono system filesystem graph container)
set(BOOST_COMPONENT_TARGETS
  Boost::timer
  Boost::chrono
  Boost::system
  Boost::filesystem
  Boost::graph
  Boost::container
)
message(STATUS "Boost targets: ${BOOST_COMPONENT_TARGETS}")


# Eigen
find_package(Eigen3 REQUIRED)

# Google Ceres

# FIND_PACKAGE(Ceres REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)

# # G2O
# FIND_PACKAGE(g2o REQUIRED)
# FIND_PACKAGE(CSparse REQUIRED)
find_package(yaml-cpp REQUIRED)

# GTSAM
find_package(GTSAM 4.3 REQUIRED)
# include_directories(${GTSAM_INCLUDE_DIR})
# set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs
message("GTSAM_INCLUDE_DIR = " ${GTSAM_INCLUDE_DIR})

find_package(GTSAMCMakeTools)
include(GtsamMakeConfigFile)
include(GtsamBuildTypes)
include(GtsamTesting)

link_directories(${GTSAM_LIBRARY_DIRS})
#  Opencv
find_package(OpenCV 4.4 REQUIRED)

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})


#ADD_SUBDIRECTORY()
# INCLUDE_DIRECTORIES(include)





add_library(rfsslam
  src/Landmark.cpp
  src/Pose.cpp
  src/TimeStamp.cpp
  src/Timer.cpp
  src/Frame.cpp
  src/ProcessModel_Odometry1D.cpp
  src/ProcessModel_Odometry2D.cpp
  src/ProcessModel_Odometry6D.cpp
  src/ProcessModel_Ackerman2D.cpp
  src/MeasurementModel_Rng1D.cpp
  src/MeasurementModel_RngBrg.cpp
  src/MeasurementModel_RngBrg_amplitude.cpp
  src/MeasurementModel_XY.cpp
  src/MeasurementModel_6D.cpp
  src/MeasurementModel_3D_stereo_orb.cpp
  src/isInFrustum.cpp
  src/MeasurementModel_VictoriaPark.cpp
  src/KalmanFilter_RngBrg_amplitude.cpp
  src/KalmanFilter.cpp
  src/HungarianMethod.cpp
  src/MurtyAlgorithm.cpp
  src/BruteForceAssignment.cpp
  src/CostMatrix.cpp
  src/PermutationLexicographic.cpp
  src/MatrixPermanent.cpp
  src/misc/memProfile.cpp
  src/GaussianGenerators.cpp
  src/AssociationSampler.cpp
  src/OrbslamMapPoint.cpp
  src/OrbslamPose.cpp
  src/external/ORBextractor.cc
  src/external/Converter.cc
  )

target_include_directories(rfsslam
  PUBLIC
    ${GTSAM_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PERFTOOLS_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_compile_options(rfsslam PRIVATE "$<${MARCH_NATIVE_CONFIGS}:-march=native>")
if(TARGET Eigen3::Eigen)
  target_link_libraries(rfsslam PUBLIC Eigen3::Eigen)
else()
  target_include_directories(rfsslam PUBLIC ${EIGEN_INCLUDE_DIRS})
endif()
target_link_libraries(rfsslam
  PUBLIC
    ${geometry_msgs_TARGETS}
    ${visualization_msgs_TARGETS}
    ${OpenCV_LIBS}
    ${BOOST_COMPONENT_TARGETS}
    gtsam
    Sophus::Sophus
)
if(TARGET yaml-cpp)
  target_link_libraries(rfsslam PUBLIC yaml-cpp)
else()
  target_link_libraries(rfsslam PUBLIC ${YAML_CPP_LIBRARIES})
endif()
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(rfsslam PUBLIC OpenMP::OpenMP_CXX)
endif()

# TARGET_LINK_LIBRARIES(rfsslam  gtsam ${Boost_LIBRARIES}  ${OpenCV_LIBS} ${Sophus_LIBRARIES} fmt::fmt   )

get_target_property(rfs_slam_include_dirs rfsslam INCLUDE_DIRECTORIES)
message(STATUS "rfsslam include dir: ${rfs_slam_include_dirs}")



add_executable(rbphdslam6dSim_euroc src/rbphdslam6d_euroc.cpp)

target_include_directories(rbphdslam6dSim_euroc
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_compile_options(rbphdslam6dSim_euroc PRIVATE "$<${MARCH_NATIVE_CONFIGS}:-march=native>")

target_link_libraries(rbphdslam6dSim_euroc
  PUBLIC
    rfsslam
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${visualization_msgs_TARGETS}
    rclcpp::rclcpp
    tf2::tf2
  PRIVATE
    ${BOOST_COMPONENT_TARGETS}
    ${YAML_CPP_LIBRARIES}
    ${Perftools_PROFILER_LIBRARY}
    ${Perftools_TCMALLOC_LIBRARY}
    Threads::Threads
    gtsam
)

if(BUILD_TESTING)

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(${PROJECT_NAME}_tests test/tests.cpp)
  target_include_directories(${PROJECT_NAME}_tests PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${PROJECT_NAME}_tests
    ${visualization_msgs_TARGETS}
    rclcpp::rclcpp
    rfsslam
    ${GTEST_LIBRARIES}
    ${BOOST_COMPONENT_TARGETS}
  )


endif()

install(
  TARGETS rbphdslam6dSim_euroc
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)



ament_package()
