cmake_minimum_required(VERSION 3.8)

project("MyVulkan")

set(CMAKE_CXX_STANDARD 17)


add_definitions(-DVK_ASSETS_DIR=\"${CMAKE_SOURCE_DIR}/assets/\")
add_definitions(-DVK_SHADERS_DIR=\"${CMAKE_SOURCE_DIR}/shaders/\")

find_package(Vulkan REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_subdirectory(third_party)
add_subdirectory(src)

find_program(GLSL_VALIDATOR glslangValidator HINTS $ENV{VULKAN_SDK}/Bin/)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    )

foreach(GLSL ${GLSL_SOURCE_FILES})
    message(STATUS "BUILDING SHADER")
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
    message(STATUS ${GLSL})
    ##execute glslang command to compile that specific shader
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL ${GLSL_SOURCE_FILES})

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )