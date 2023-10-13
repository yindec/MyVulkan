#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

class Camera {
private:
	float fov;
	float znear, zfar;
public:
	enum CameraType { lookat, firstperson };
	CameraType type = CameraType::lookat;

	glm::vec3 position;
	glm::vec3 dirction;
	glm::vec3 up;

	bool flipY = true;

	struct
	{
		glm::mat4 perspective;
		glm::mat4 view;
	} matrices;

	void keyboardInput(Camera_Movement dirction) {
		if(dirction == FORWARD)
			matrices.perspective[1][1] *= -1;
	}

	void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
	{
		if(xoffset != yoffset)
			matrices.perspective[1][1] *= -1;
	}


	void mouseInput() {

	}

	void setPosition(glm::vec3 position){
		this->position = position;
	}

	void setDirction(glm::vec3 dirction) {
		this->dirction = dirction;
	}

	void setUp(glm::vec3 up) {
		this->up = up;
	}

	void setView() {
		this->matrices.view = glm::lookAt(position, dirction, up);
	}

	void setPerspective(float fov, float aspect, float znear, float zfar)
	{
		this->fov = fov;
		this->znear = znear;
		this->zfar = zfar;
		matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
		if (flipY) {
			matrices.perspective[1][1] *= -1.0f;
		}
	};
};