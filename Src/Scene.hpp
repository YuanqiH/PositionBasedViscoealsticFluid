#ifndef SCENE_H
#define SCENE_H

#include "common.h"
#include "parameters.h"
#include "setupFunctions.h"

class Scene {
public:
	Scene(std::string name) : name(name) {}
	virtual void init(tempSolver* tp, solverParams* sp) = 0;
	
private:
	std::string name;

};

class DamBreak : public Scene {
public:
	DamBreak(std::string name) : Scene(name) {}

	virtual void init(tempSolver* tp, solverParams* sp) {
		const float radius = 0.1f;//0.1f;
		const float restDistance = radius * 0.5f;//0.5f;
		float3 lower = make_float3( 0.0f * radius, 20 * radius, 0.0f * radius);
		//int3 dims = make_int3(68, 48, 88);     
		int3 dims = make_int3(20, 20 , 20);
		createParticleGrid(tp, sp, lower, dims, restDistance);  // initial all the tempsolver values 
		
		sp->radius = radius;
		sp->restDistance = restDistance;
		sp->numIterations = 5;
		sp->numDiffuse = 1024 * 1024;
		sp->numParticles = int(tp->positions.size());
		sp->numCloth = 0;
		sp->numConstraints = 0;
		sp->gravity = make_float3(0.0f, -4.0f, 0);
		sp->bounds = make_float3(dims) * radius * 2;
		sp->boundsMin = make_float3(0) * radius;
		sp->gridWidth = int(sp->bounds.x / radius);
		sp->gridHeight = int(sp->bounds.y / radius);
		sp->gridDepth = int(sp->bounds.z / radius);
		sp->gridSize = sp->gridWidth * sp->gridHeight * sp->gridDepth;
		sp->maxContacts = 10;   // for particle collision
		sp->maxNeighbors = 30;
		sp->maxParticles = 50;
		sp->restDensity = 5778.0f;//6378.0f;  // the most important factor for this model to hold shape
		sp->lambdaEps = 600.0f;//600.0f; // density
		sp->vorticityEps = 0.0001f; // 0.0001f;
		sp->C = 0.00025f; //0.0025f;  // maximum 0.03, viscosity
		sp->K = 0.0003f;// 0.00001f; // change the K for artificial tensile pressure, has been less than 0.001f
		sp->KPOLY = 315.0f / (64.0f * PI * pow(radius, 9));
		sp->SPIKY = 45.0f / (PI * pow(radius, 6));
		sp->dqMag = 0.2f * radius;  // the lowver artificial greater
		sp->wQH = sp->KPOLY * pow((radius * radius - sp->dqMag * sp->dqMag), 3);
		// elastic initial
		sp->mu_s = 10000.0f;//0.5f;  // maximum 0.5f // 3000.0f (0 - 10e6)
		sp->threshold = 10.5f; // the larger the stronger

		// viscosity kernel coeficient
		sp->Kernel_Viscosity_c = 45.0f /(PI * pow(radius, 5));
		sp->mu_Viscosity = 700.0f; //for the viscosity force 0 - 800(unstatle)
		
		sp->K_spring = 0.05f;

		tp->diffusePos.resize(sp->numDiffuse);
		tp->diffuseVelocities.resize(sp->numDiffuse);
	}
};

class FluidCloth : public Scene {
public:
	FluidCloth(std::string name) : Scene(name) {}

	virtual void init(tempSolver* tp, solverParams* sp) {
		float stretch = 1.0f;
		float bend = 1.0f;
		float shear = 1.0f;

		const float radius = 0.1f;
		const float restDistance = radius * 0.5f;
		float3 lower = make_float3(0.0f, 1.0f, 0.0f);
		int3 dims = make_int3(64, 1, 64);
		createCloth(tp, sp, lower, dims, radius * 0.25f, 1, stretch, bend, shear, 0.05f);
		sp->numCloth = int(tp->positions.size());

		//Pinned vertices
		int c1 = 0;
		int c2 = dims.x - 1;
		int c3 = dims.x * (dims.z - 1);
		int c4 = (dims.x * dims.z) - 1;

		tp->positions[c1].w = 0;
		tp->positions[c2].w = 0;
		tp->positions[c3].w = 0;
		tp->positions[c4].w = 0;

		//Tethers
		for (int i = 0; i < int(tp->positions.size()); i++) {
			//tp->positions[i].y = 1.5f - sinf(25.0f * 180.0f / PI) * tp->positions[i].x;
			//tp->positions[i].x *= cosf(25.0f * 180.0f / PI);

			if (i != c1 && i != c2 && i != c3 && i != c4) {
				float tether = -0.1f;
				addConstraint(tp, sp, c1, i, tether);
				addConstraint(tp, sp, c2, i, tether);
				addConstraint(tp, sp, c3, i, tether);
				addConstraint(tp, sp, c4, i, tether);
			}
		}

		//move corners closer together?

		//Add water
		lower = make_float3(0.5f, 1.1f, 0.5f);
		dims = make_int3(10, 10, 10);
		createParticleGrid(tp, sp, lower, dims, restDistance);

		sp->radius = radius;
		sp->restDistance = restDistance;
		sp->numIterations = 4;
		sp->numDiffuse = 1024 * 1024;
		sp->numParticles = int(tp->positions.size());
		sp->numConstraints = int(tp->restLengths.size());
		sp->gravity = make_float3(0, 0.0f, 0);
		sp->bounds = make_float3(dims.x * 4 * radius, dims.y * 4 * radius, dims.z * 4 * radius);
		sp->gridWidth = int(sp->bounds.x / radius);
		sp->gridHeight = int(sp->bounds.y / radius);
		sp->gridDepth = int(sp->bounds.z / radius);
		sp->gridSize = sp->gridWidth * sp->gridHeight * sp->gridDepth;
		sp->maxContacts = 15;
		sp->maxNeighbors = 50;
		sp->maxParticles = 50;
		sp->restDensity = 6378.0f;
		sp->lambdaEps = 600.0f;
		sp->vorticityEps = 0.0001f;
		sp->C = 0.01f; //0.0025f;
		sp->K = 0.00001f;
		sp->KPOLY = 315.0f / (64.0f * PI * pow(radius, 9));
		sp->SPIKY = 45.0f / (PI * pow(radius, 6));
		sp->dqMag = 0.3f * radius;
		sp->wQH = sp->KPOLY * pow((radius * radius - sp->dqMag * sp->dqMag), 3);
		sp->mu_s = 0.5f;

		tp->diffusePos.resize(sp->numDiffuse);
		tp->diffuseVelocities.resize(sp->numDiffuse);
	}
};

class Lighthouse : public Scene {
public:

};

#endif