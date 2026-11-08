# CortexModelServer: from CortexAgent (crates/llama-engine/native/CMakeLists.txt).
#
# Builds `llama-server` from CortexAgent's vendored llama.cpp (../vendor/llama.cpp, byte for byte
# CortexAgent's vendor/llama.cpp: unslothai/llama.cpp `468ad58` (b11876) with its four patch sites). CortexAgent copied
# only the sources it builds, so the tree has no tools/CMakeLists.txt, no tools/server/CMakeLists.txt,
# no server.cpp / main.cpp and no tools/ui. The vendored tree is therefore added exactly the way
# CortexAgent's engine shim adds it (same cache settings), and the executable is assembled here:
# the server core CortexAgent links (server-context.cpp and the rest, carrying the patches), plus
# the vendored HTTP layer (server-http.cpp, cpp-httplib), plus agent_server.cpp (this folder), which
# hosts them the way upstream's server.cpp does for one model, with CortexAgent's route set
# (gc_engine.cpp, including /tool-call-markers). ui/ is the UI table upstream's ui-assets.cmake
# generates when no UI is built (zero assets).
cmake_minimum_required(VERSION 3.14)
project(cms_llama_agent_server C CXX)

set(LLAMA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/llama.cpp)

set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
# The vendored tree is the source CortexTerm's model-llama and the knowledge app's worker build
# (vendor/llama.cpp/REVISION), a 0.7.0 development build; it is named as they name it, so all
# three report the same version.
set(LLAMA_BUILD_IS_DEV    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_NUMBER    11876)
set(LLAMA_BUILD_COMMIT    468ad58)
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD      ON  CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL   OFF CACHE BOOL "" FORCE)
# Portable CPU code: the binary must run on every supported Mac, not only the build machine.
set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)
if (APPLE)
    set(GGML_METAL               ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(${LLAMA_ROOT} llama.cpp)

# CortexModelServer patch: the UI table of a build without UI (upstream tools/ui/ui.h.in and
# ui.cpp.in expanded with zero assets), which server-http.cpp includes.
add_library(llama-ui-none STATIC ui/ui.cpp)
target_include_directories(llama-ui-none PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ui)
target_compile_features(llama-ui-none PRIVATE cxx_std_17)

set(SERVER ${LLAMA_ROOT}/tools/server)
add_executable(llama-server
    agent_server.cpp
    # the server core, exactly the files CortexAgent's gc_engine links
    ${SERVER}/server-chat.cpp
    ${SERVER}/server-task.cpp
    ${SERVER}/server-queue.cpp
    ${SERVER}/server-common.cpp
    ${SERVER}/server-decision.cpp
    ${SERVER}/server-context.cpp
    ${SERVER}/server-stream.cpp
    ${SERVER}/server-tools.cpp
    ${SERVER}/server-mcp.cpp
    ${SERVER}/server-schema.cpp
    # CortexModelServer patch: the vendored HTTP layer (CortexAgent hosts the core in-process)
    ${SERVER}/server-http.cpp
)
target_include_directories(llama-server PRIVATE ${SERVER} ${LLAMA_ROOT}/tools/mtmd ${LLAMA_ROOT})
find_package(Threads REQUIRED)
target_link_libraries(llama-server PRIVATE llama-common mtmd cpp-httplib llama-ui-none Threads::Threads)
target_compile_features(llama-server PRIVATE cxx_std_17)
set_target_properties(llama-server PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
