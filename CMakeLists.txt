
CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE RelWithDebInfo)



PROJECT(phd_orbslam_minimal)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")





set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")




# ROS2 stuff
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)


# OpenMP
IF(APPLE AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
  SET(USE_OPENMP OFF) # OpenMP does not yet work with Clang, but hopefully soon
ELSE()
  SET(USE_OPENMP ON CACHE BOOL "Use OpenMP to compile multithreaded versions of SLAM algorithms.")
ENDIF()
IF(USE_OPENMP)
  IF("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libomp ")
  ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
  ENDIF()
ENDIF()


#Threads

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON) 
# set(Boost_USE_STATIC_RUNTIME ON)
FIND_PACKAGE(Boost COMPONENTS timer chrono system filesystem graph container REQUIRED )
MESSAGE("Boost_INCLUDE_DIRS = " ${Boost_INCLUDE_DIRS})
MESSAGE("Boost_LIBRARIES = " ${Boost_LIBRARIES})


# For cmake custom library searches
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Eigen
FIND_PACKAGE(Eigen REQUIRED)

# Google Ceres

# FIND_PACKAGE(Ceres REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)

# # G2O 
# FIND_PACKAGE(g2o REQUIRED)
# FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(yaml-cpp REQUIRED)

# GTSAM
find_package(GTSAM 4.3 REQUIRED)
# include_directories(${GTSAM_INCLUDE_DIR})
# set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs
message("GTSAM_INCLUDE_DIR = " ${GTSAM_INCLUDE_DIR})

# find_package(GTSAMCMakeTools)
# include(GtsamMakeConfigFile)
# include(GtsamBuildTypes)
# include(GtsamTesting)

link_directories(${GTSAM_LIBRARY_DIRS})
#  Opencv
find_package(OpenCV 4.4)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})




    

#ADD_SUBDIRECTORY()
# INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(test)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIRS})
# INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${PERFTOOLS_INCLUDE_DIRS})
# INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CSPARSE_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIRS})


ADD_LIBRARY(rfsslam 
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
  
target_include_directories(rfsslam PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                 "$<INSTALL_INTERFACE:include>")

ament_target_dependencies(rfsslam Boost OpenCV GTSAM)

target_link_libraries(rfsslam  gtsam Sophus::Sophus   )
  
# TARGET_LINK_LIBRARIES(rfsslam  gtsam ${Boost_LIBRARIES}  ${OpenCV_LIBS} ${Sophus_LIBRARIES} fmt::fmt   )




ADD_EXECUTABLE(rbphdslam6dSim_euroc src/rbphdslam6d_euroc.cpp)
ament_target_dependencies(rbphdslam6dSim_euroc
GTSAM
rclcpp
visualization_msgs
std_msgs
sensor_msgs
tf2
)

target_include_directories(rbphdslam6dSim_euroc PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                 "$<INSTALL_INTERFACE:include>")

TARGET_LINK_LIBRARIES(rbphdslam6dSim_euroc  rfsslam ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES}   ${Perftools_PROFILER_LIBRARY} ${Perftools_TCMALLOC_LIBRARY} Threads::Threads   gtsam )


install(
  TARGETS rbphdslam6dSim_euroc
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)



ament_package()
