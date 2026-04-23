cmake_minimum_required(VERSION 3.20)
project(mini_v_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  asio_standalone
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG asio-1-30-2
  GIT_SHALLOW TRUE
)
FetchContent_GetProperties(asio_standalone)
if(NOT asio_standalone_POPULATED)
  FetchContent_Populate(asio_standalone)
endif()
set(ASIO_INCLUDE_DIR "${asio_standalone_SOURCE_DIR}/asio/include"
    CACHE PATH "Standalone Asio headers" FORCE)

set(CROW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CROW_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  Crow
  GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
  GIT_TAG v1.2.1.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Crow)

add_executable(server src/main.cpp)
target_link_libraries(server PRIVATE Crow::Crow nlohmann_json::nlohmann_json)
