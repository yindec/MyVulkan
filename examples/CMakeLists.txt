

function(buildExample EXAMPLE_NAME)

    add_executable(${EXAMPLE_NAME}  ${EXAMPLE_NAME}/${EXAMPLE_NAME}.cpp)

    set_property(TARGET ${EXAMPLE_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXAMPLE_NAME}>")

    target_include_directories(${EXAMPLE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../base)
    target_link_libraries(${EXAMPLE_NAME} glm glfw stb_image tiny_obj_loader Vulkan::Vulkan)

    target_link_libraries(${EXAMPLE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/glfw/lib-vc2022/glfw3.lib)

    add_dependencies(${EXAMPLE_NAME} Shaders)
endfunction(buildExample)

# Build all examples
function(buildExamples)
	foreach(EXAMPLE ${EXAMPLES})
		buildExample(${EXAMPLE})
	endforeach(EXAMPLE)
endfunction(buildExamples)

set(EXAMPLES
    MyVulkan
    triangle
)

buildExamples()