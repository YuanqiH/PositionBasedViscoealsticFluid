#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "common.h"

struct tempSolver {
	std::vector<float4> positions;
	std::vector<float3> velocities;
	std::vector<int> phases;

	std::vector<float4> diffusePos;
	std::vector<float3> diffuseVelocities;

	std::vector<int> clothIndices;
	std::vector<float> restLengths;
	std::vector<float> stiffness;
	std::vector<int> triangles;
	std::vector<mat3> elasticTerm; // elastic term 
	std::vector<float3> Force_internal;// internal force
	
};

struct solver {
	float4* oldPos;
	float4* newPos;
	float3* velocities;
	int* phases;
	float* densities;

	float4* diffusePos;
	float3* diffuseVelocities;

	int* clothIndices;
	float* restLengths;
	float* stiffness;

	int* neighbors;
	int* numNeighbors;
	int* gridCells;
	int* gridCounters;
	int* contacts;
	int* numContacts;

	float3* deltaPs;

	float* buffer0;

	// add the elastic term here 
	mat3* elasticTerm;
	float3* Force_internal;
};

struct solverParams {
public:
	int maxNeighbors;
	int maxParticles;
	int maxContacts;
	int gridWidth, gridHeight, gridDepth;
	int gridSize;

	int numParticles;
	int numDiffuse;

	int numCloth;
	int numConstraints;

	float3 gravity;
	float3 bounds;
	float3 boundsMin;

	int numIterations;
	float radius;
	float restDistance;
	//float sor;
	//float vorticity;

	float KPOLY;
	float SPIKY;
	float restDensity;
	float lambdaEps;
	float vorticityEps;
	float C;
	float K;
	float dqMag;
	float wQH;

	// add the elastic stress coefficient here
	float mu_s;
	float threshold;
	float Kernel_Viscosity_c;
	float mu_Viscosity;

	float K_spring;
	
};

#endif