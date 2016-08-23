#ifndef PARTICLE_H
#define PARTICLE_H
#include "common.h"
struct Particle {
public:
	glm::vec3 oldPos;
	glm::vec3 newPos;
	glm::vec3 velocity;
	float invMass;
	int phase;
};

#endif