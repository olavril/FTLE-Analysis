include(analysis_add_test)

set(UNIT_TESTS test_main.cpp)

function(addMpiTest source exename testname ranks)
    add_executable(${exename} ${source})
    target_include_directories(${exename} PUBLIC ${MPI_CXX_INCLUDE_PATH} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/extern/io ${PROJECT_SOURCE_DIR}/extern/Eigen ${PROJECT_SOURCE_DIR}/main/src)
    target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX GTest::gtest_main)

    if(ENABLE_H5PART)
        enableH5Part(${exename})
    endif()

    analysis_add_test(${testname} EXECUTABLE ${exename} RANKS ${ranks})
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)
endfunction()

function(addAnalysisMpiTest source exename testname ranks)
    list(APPEND source test_main.cpp)
    addMpiTest("${source}" ${exename} ${testname} ${ranks})
endfunction()

addMpiTest(analysis_tests.cpp test_analysis_mpi 1 2)
