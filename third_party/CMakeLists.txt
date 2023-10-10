find_package(Vulkan REQUIRED)

add_library(glm INTERFACE)
target_include_directories(glm INTERFACE glm)

add_library(glfw INTERFACE)
target_include_directories(glfw INTERFACE glfw/include)

add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE stb_image)