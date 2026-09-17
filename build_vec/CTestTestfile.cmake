# CMake generated Testfile for 
# Source directory: D:/AI/study/CPU GEMM/cpu-gemm
# Build directory: D:/AI/study/CPU GEMM/cpu-gemm/build_vec
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(GemmCorrectnessTest "D:/AI/study/CPU GEMM/cpu-gemm/build_vec/Debug/test_gemm.exe")
  set_tests_properties(GemmCorrectnessTest PROPERTIES  _BACKTRACE_TRIPLES "D:/AI/study/CPU GEMM/cpu-gemm/CMakeLists.txt;43;add_test;D:/AI/study/CPU GEMM/cpu-gemm/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(GemmCorrectnessTest "D:/AI/study/CPU GEMM/cpu-gemm/build_vec/Release/test_gemm.exe")
  set_tests_properties(GemmCorrectnessTest PROPERTIES  _BACKTRACE_TRIPLES "D:/AI/study/CPU GEMM/cpu-gemm/CMakeLists.txt;43;add_test;D:/AI/study/CPU GEMM/cpu-gemm/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(GemmCorrectnessTest "D:/AI/study/CPU GEMM/cpu-gemm/build_vec/MinSizeRel/test_gemm.exe")
  set_tests_properties(GemmCorrectnessTest PROPERTIES  _BACKTRACE_TRIPLES "D:/AI/study/CPU GEMM/cpu-gemm/CMakeLists.txt;43;add_test;D:/AI/study/CPU GEMM/cpu-gemm/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(GemmCorrectnessTest "D:/AI/study/CPU GEMM/cpu-gemm/build_vec/RelWithDebInfo/test_gemm.exe")
  set_tests_properties(GemmCorrectnessTest PROPERTIES  _BACKTRACE_TRIPLES "D:/AI/study/CPU GEMM/cpu-gemm/CMakeLists.txt;43;add_test;D:/AI/study/CPU GEMM/cpu-gemm/CMakeLists.txt;0;")
else()
  add_test(GemmCorrectnessTest NOT_AVAILABLE)
endif()
