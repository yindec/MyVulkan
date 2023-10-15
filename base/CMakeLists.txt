project(base)

file(GLOB BASE_SRC "*.cpp" "*.hpp" "*.h")


add_library(${PROJECT_NAME} STATIC ${BASE_SRC})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} glm glfw stb_image tiny_obj_loader Vulkan::Vulkan imgui)
