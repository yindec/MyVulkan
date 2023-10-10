# Add source to this project's executable.
add_executable(MyVulkan
    main.cpp
    VulkanBase.h
    )


set_property(TARGET MyVulkan PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:MyVulkan>")

target_include_directories(MyVulkan PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(MyVulkan glm glfw stb_image Vulkan::Vulkan)

target_link_libraries(MyVulkan ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/glfw/lib-vc2022/glfw3.lib)

add_dependencies(MyVulkan Shaders)
