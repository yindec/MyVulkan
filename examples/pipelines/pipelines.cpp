/*
* Vulkan Example - Using different pipelines in one single renderpass
*
* Copyright (C) 2016-2023 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"


#ifdef NDEBUG
#define ENABLE_VALIDATION false
#else
#define ENABLE_VALIDATION true
#endif

class VulkanExample :public VulkanExampleBase {

public:
	vkglTF::Model scene;

	vks::Buffer uniformBuffer;

	// Same uniform buffer layout as shader
	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 modelView;
		glm::vec4 lightPos = glm::vec4(0.0f, 2.0f, 1.0f, 0.0f); /// seems like a vector for light dirction
	} uboVS;

	VkPipelineLayout pipelineLayout;
	VkDescriptorSet descriptorSet;
	VkDescriptorSetLayout descriptorSetLayout;

	struct {
		VkPipeline phong;
		VkPipeline wireframe;
		VkPipeline toon;
	} pipelines;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION) {

		title = "Pipeline state objects";
		camera.type = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -10.5f));
		camera.setRotation(glm::vec3(-25.0f, 15.0f, 0.0f));
		camera.setRotationSpeed(0.5f);
		camera.setPerspective(60.0f, (float)(width / 3.0f) / (float)(height), 0.1f, 256.0f);
	}

	~VulkanExample() {

		vkDestroyPipeline(device, pipelines.phong, nullptr);
		if (enabledFeatures.fillModeNonSolid) {
			vkDestroyPipeline(device, pipelines.wireframe, nullptr);
		}
		vkDestroyPipeline(device, pipelines.toon, nullptr);

		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		uniformBuffer.destroy();
	}

	virtual void getEnabledFeatures() {

		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		}

		if (deviceFeatures.wideLines) {
			enabledFeatures.wideLines = VK_TRUE;
		}
	}

	void buildCommandBuffers() {

		VkCommandBufferBeginInfo cmdBufferBeginInfo{};
		cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo{};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i) {

			renderPassBeginInfo.framebuffer = frameBuffers[i];
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufferBeginInfo));
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport{};
			viewport.width = (float)width;
			viewport.height = (float)height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			//vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor{};
			scissor.extent.width = width;
			scissor.extent.height = height;
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			scene.bindBuffers(drawCmdBuffers[i]);

			// Left : Solid colored
			viewport.width = (float)width / 3.0f;
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.phong);
			vkCmdSetLineWidth(drawCmdBuffers[i], 1.0f);
			scene.draw(drawCmdBuffers[i]);

			// Center : Toon
			viewport.x = (float)width / 3.0f;
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.toon);
			if (enabledFeatures.wideLines) {
				vkCmdSetLineWidth(drawCmdBuffers[i], 2.0f);
			}
			scene.draw(drawCmdBuffers[i]);

			// Right : Wireframe
			if (enabledFeatures.fillModeNonSolid) {
				viewport.x = (float)width / 3.0f * 2.0f;
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.wireframe);
				scene.draw(drawCmdBuffers[i]);
			}

			drawUI(drawCmdBuffers[i]);
			vkCmdEndRenderPass(drawCmdBuffers[i]);
			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void loadAssets() {

		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices |
			vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
		scene.loadFromFile(getAssetPath() + "models/treasure_smooth.gltf", vulkanDevice, queue, glTFLoadingFlags);
	}

	void setupDescriptorPool() {
		VkDescriptorPoolSize descriptorPoolSize{};
		descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorPoolSize.descriptorCount = 1;
		std::vector< VkDescriptorPoolSize> descriptorPoolSizes = { descriptorPoolSize };

		VkDescriptorPoolCreateInfo descriptorPoolCI{};
		descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCI.poolSizeCount = 1;
		descriptorPoolCI.pPoolSizes = descriptorPoolSizes.data();
		descriptorPoolCI.maxSets = 2;

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorPool));
	}

	void setupDescriptorSetLayout() {

		std::array<VkDescriptorSetLayoutBinding, 1> setLayoutBindings{};
		//VkDescriptorSetLayoutBinding setLayoutBinding{};
		setLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		setLayoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		setLayoutBindings[0].binding = 0;
		setLayoutBindings[0].descriptorCount = 1;
		
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
		descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
		descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCI{};
		pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCI.setLayoutCount = 1;
		pipelineLayoutCI.pSetLayouts = &descriptorSetLayout;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));
	}

	void setupDescriptorSet() {

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.pSetLayouts = &descriptorSetLayout;
		allocInfo.descriptorSetCount = 1;

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

		std::array<VkWriteDescriptorSet, 1> writeDescriptorSets{};
		writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[0].dstSet = descriptorSet;
		writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeDescriptorSets[0].dstBinding = 0;
		writeDescriptorSets[0].pBufferInfo = &uniformBuffer.descriptor;
		writeDescriptorSets[0].descriptorCount = 1;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()),
			writeDescriptorSets.data(), 0, nullptr);
	}

	void preparePipelines() {

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
		inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssemblyState.flags = 0;
		inputAssemblyState.primitiveRestartEnable = VK_FALSE;

		VkPipelineRasterizationStateCreateInfo rasterizationState{};
		rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizationState.flags = 0;
		rasterizationState.depthClampEnable = VK_FALSE;
		rasterizationState.lineWidth = 1.0f;

		VkPipelineColorBlendAttachmentState blendAttachmentState{};
		blendAttachmentState.colorWriteMask = 0xf;
		blendAttachmentState.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlendState{};
		colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlendState.attachmentCount = 1;
		colorBlendState.pAttachments = &blendAttachmentState;

		VkPipelineDepthStencilStateCreateInfo depthStencilState{};
		depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencilState.depthTestEnable = VK_TRUE;
		depthStencilState.depthWriteEnable = VK_TRUE;
		depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencilState.back.compareOp = VK_COMPARE_OP_ALWAYS;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;
		viewportState.flags = 0;

		VkPipelineMultisampleStateCreateInfo multisampleState{};
		multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampleState.flags = 0;

		std::array<VkDynamicState, 3> dynamicStateEnables{ VK_DYNAMIC_STATE_VIEWPORT, 
			VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH};

		VkPipelineDynamicStateCreateInfo dynamicstate{};
		dynamicstate.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicstate.pDynamicStates = dynamicStateEnables.data();
		dynamicstate.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());
		dynamicstate.flags = 0;

		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
		// begin build pipeline
		VkGraphicsPipelineCreateInfo pipelineCI{};
		pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineCI.layout = pipelineLayout;
		pipelineCI.renderPass = renderPass;
		pipelineCI.flags = 0;
		pipelineCI.basePipelineIndex = -1;
		pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicstate;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::Color });

		// using this pipeline as the base for the other pipelines (derivatives)
		pipelineCI.flags = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;

		// Textured pipeline
		// phong shading pipeline
		shaderStages[0] = loadShader(getShadersPath() + "pipelines/phong.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "pipelines/phong.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.phong));

		// All pipelines created after the base pipeline will ne derivates
		pipelineCI.flags = VK_PIPELINE_CREATE_DERIVATIVE_BIT;
		// Base pipelne will be our first created pipeline
		pipelineCI.basePipelineHandle = pipelines.phong;
		pipelineCI.basePipelineIndex = -1;

		// Toon shading pipeline
		shaderStages[0] = loadShader(getShadersPath() + "pipelines/toon.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "pipelines/toon.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.toon));

		// Pipeline for wire frame rendering
		// Non solid rendering is not a mandatory Vulkan feature
		if (enabledFeatures.fillModeNonSolid) {
			rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
			shaderStages[0] = loadShader(getShadersPath() + "pipelines/wireframe.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader(getShadersPath() + "pipelines/wireframe.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.wireframe));
		}
	}

	void prepareUniformBuffers() {

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffer, sizeof(uboVS)
		));

		VK_CHECK_RESULT(uniformBuffer.map());
		updateUniformBuffers();
	}

	void updateUniformBuffers() {

		uboVS.projection = camera.matrices.perspective;
		uboVS.modelView = camera.matrices.view;
		memcpy(uniformBuffer.mapped, &uboVS, sizeof(uboVS));
	}

	void draw() {

		VulkanExampleBase::prepareFrame();

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
	}

	void prepare() {
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render() {

		if (!prepared)
			return;
		draw();
		if (camera.updated) {
			updateUniformBuffers();
		}
	}

	virtual void viewChanged() {
		camera.setPerspective(60.0f, (float)(width / 3.0f) / (float)height, 0.1f, 256.0f);
		updateUniformBuffers();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {
		if (!enabledFeatures.fillModeNonSolid) {
			if (overlay->header("Info")) {
				overlay->text("Non solid fill modes not supported!");
			}
		}
	}

};


VULKAN_EXAMPLE_MAIN()
