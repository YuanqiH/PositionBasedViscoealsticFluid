#ifndef PARTICLE_SYSTEM_CU
#define PARTICLE_SYSTEM_CU

#include "common.h"
#include "parameters.h"

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }
static dim3 dims;
static dim3 diffuseDims;
static dim3 clothDims;
static dim3 gridDims;
static const int blockSize = 128;

__constant__ solverParams sp;
__constant__ float deltaT = 0.0083f;
__device__ int foamCount = 0;
__constant__ float distr[] =
{
	-0.34828757091811f, -0.64246175794046f, -0.15712936555833f, -0.28922267225069f, 0.70090742209037f,
	0.54293139350737f, 0.86755128105523f, 0.68346917800767f, -0.74589352018474f, 0.39762042062246f,
	-0.70243115988673f, -0.85088539675385f, -0.25780126697281f, 0.61167922970451f, -0.8751634423971f,
	-0.12334015086449f, 0.10898816916579f, -0.97167591190509f, 0.89839695948101f, -0.71134930649369f,
	-0.33928178406287f, -0.27579196788175f, -0.5057460942798f, 0.2341509513716f, 0.97802030852904f,
	0.49743173248015f, -0.92212845381448f, 0.088328595779989f, -0.70214782175708f, -0.67050553191011f
};

__device__ float WPoly6(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return 0;
	}

	return sp.KPOLY * pow((sp.radius * sp.radius - pow(length(r), 2)), 3);
}

__device__ float3 gradWPoly6(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return make_float3(0.0f);
	}

	float coeff = glm::pow((sp.radius * sp.radius) - (rLen * rLen), 2);
	coeff *= -6 * sp.KPOLY;
	return r * coeff;
}

__device__ float3 WSpiky(float3 const &pi, float3 const &pj) {  // gradient of Wspiky 
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return make_float3(0.0f);
	}

	float coeff = (sp.radius - rLen) * (sp.radius - rLen); // (h-r)^2
	coeff *= sp.SPIKY;
	coeff /= rLen;
	return r * -coeff;
}

__device__ float WAirPotential(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return 0.0f;
	}

	return 1 - (rLen / sp.radius);
}

__device__ float Wviscosity(float3 const &pi, float3 const &pj){    // Laplacian of viscosity 
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) { // outside the smoothing length
		return 0.0f;
	}

	
	return sp.Kernel_Viscosity_c * ( 1 - rLen/sp.radius);

}


//functions for calculating elastic terms 
//=============================================
__device__ float3 elasticForce(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, float* density, mat3* elasticTerm, int index) {

	// update the elsticTerm at this moment 
	// total - plastic 

	// CalculateElasticTerm(); // only calculate plastic term, we can calculate total in this function 

		// total term at this moment, jacobian matrix 
	// this is wrong, since the javobian matrix is 3 x 3, here only have math function for float3 vector
	mat3 graintOfVi = make_mat3(0.0f);
	mat3 graintOfVi_Trans, Stotal = make_mat3(0.0f);

	float3 velocityDiff;
	float3 gradient;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			velocityDiff = (velocities[neighbors[(index * sp.maxNeighbors) + i]] - velocities[index]) * deltaT;
			gradient = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]));		
			graintOfVi += (outer_product(velocityDiff, gradient) / density[i]);
		}
	}

	graintOfVi_Trans = mat3_transpose(graintOfVi);
	Stotal = (graintOfVi + graintOfVi_Trans) * 0.5f;
	
	//printf("the gradient of graintOfVi %d is %f, %f, %f \n",index,graintOfVi.x,graintOfVi.y,graintOfVi.z);
	/*
	float trace_D = sqrtf( 0.5f * trace(Stotal) * trace(Stotal));
	float inverse_trace_D = 1/ trace_D;
	float Jump = 1.0f;
	float yita_D = (1- exp( -(Jump + 1) * trace_D)) * ( sqrt(inverse_trace_D) + inverse_trace_D);

	elasticTerm[index] +=  deltaT* yita_D * Stotal;
	*/
	/*
	//===========================
	//this method doesnt work
	mat3 RotationTensor = (graintOfVi- graintOfVi_Trans) * 0.5f;
	mat3 delta_S = 2 * sp.mu_s * ( Stotal - trace(Stotal) * make_mat3_diag(1) + elasticTerm[index] * RotationTensor - RotationTensor * elasticTerm[index]);
	elasticTerm[index] += delta_S * deltaT;
	*/

	
	//======================
	// 1st method doesnt work
	mat3 Splastic = make_mat3(0.0f);
	float SpmNorm;
	float alph_elas = 2 * sp.mu_s;//1.0f;
	//float threshold = 0.0f;
	mat3 Spm = elasticTerm[index] - (trace(elasticTerm[index])/3.0f) * make_mat3_diag(1);
	SpmNorm = Frobenius_Norm(Spm);
	if(SpmNorm != 0.0f){
	Splastic = (alph_elas/SpmNorm) * fmaxf(0.0f,SpmNorm- sp.threshold) * Spm;
	}
	// update elastic term
	elasticTerm[index] += (Stotal - Splastic)* deltaT;
	

	// update the elstic Force then
	float3 elasticForceValue =  make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			// the elastic stress coefficient need modification later
			// using symetrical approximated 
			elasticForceValue += (elasticTerm[index] + elasticTerm[i]) * WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) * (sp.mu_s / (density[i] * density[index])) ;
			
		}
	}
	/*
	if(index< 10){
		printf("the Stotal r0 %d is %f, %f, %f \n",index,Stotal.r0.x,Stotal.r0.y,Stotal.r0.z);
	printf("the Stotal r1 %d is %f, %f, %f \n",index,Stotal.r1.x,Stotal.r1.y,Stotal.r1.z);
	printf("the Stotal r2 %d is %f, %f, %f \n \n",index,Stotal.r2.x,Stotal.r2.y,Stotal.r2.z);

	printf("the Splastic r0 %d is %f, %f, %f \n",index,Splastic.r0.x,Splastic.r0.y,Splastic.r0.z);
	printf("the Splastic r1 %d is %f, %f, %f \n",index,Splastic.r1.x,Splastic.r1.y,Splastic.r1.z);
	printf("the Splastic r2 %d is %f, %f, %f \n \n",index,Splastic.r2.x,Splastic.r2.y,Splastic.r2.z);
	printf("the gradient of elasticForceValue %d is %f, %f, %f \n \n",index,elasticForceValue.x,elasticForceValue.y,elasticForceValue.z);
	}
	*/
	return elasticForceValue;
}

__device__ float3 ViscosityForce(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, float* density, int index) {
	
	float3 ViscosityForce =  make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			float3	velocityDiff = velocities[neighbors[(index * sp.maxNeighbors) + i]] - velocities[index];
			// using symetrical approximated 
			ViscosityForce += sp.mu_Viscosity * velocityDiff / (density[i] * density[index]) * Wviscosity(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) ;
		}
	}

	return ViscosityForce;
}

__device__ float3 Pressureforce(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, float* density, int index) {
	
	float3 PressureFinal =  make_float3(0.0f);
	float pressure_i = sp.K * ( density[index] - sp.restDensity);

	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			float pressure_j = sp.K * ( density[i] - sp.restDensity);

			// using symetrical approximated 
			PressureFinal +=  -1.0f * (pressure_i/(density[index] * density[index]) + pressure_j /(density[i] * density[i]) ) / density[i] * WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) ;
		}
	}

	return PressureFinal;
}






//==========================================
// all the PBD functions
//Returns the eta vector that points in the direction of the corrective force
__device__ float3 eta(float4* newPos, int* phases, int* neighbors, int* numNeighbors, int &index, float &vorticityMag) {
	float3 eta = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0)
			eta += WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) * vorticityMag;
	}

	return eta;
}

__device__ float3 vorticityForce(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, int index) {
	//Calculate omega_i
	float3 omega = make_float3(0.0f);
	float3 velocityDiff;
	float3 gradient;

	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			velocityDiff = velocities[neighbors[(index * sp.maxNeighbors) + i]] - velocities[index];
			gradient = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]));
			omega += cross(velocityDiff, gradient);
		}
	}

	float omegaLength = length(omega);
	if (omegaLength == 0.0f) {
		//No direction for eta
		return make_float3(0.0f);
	}

	float3 etaVal = eta(newPos, phases, neighbors, numNeighbors, index, omegaLength);
	if (etaVal.x == 0 && etaVal.y == 0 && etaVal.z == 0) {
		//Particle is isolated or net force is 0
		return make_float3(0.0f);
	}

	float3 n = normalize(etaVal);

	return (cross(n, omega) * sp.vorticityEps);
}

__device__ float sCorrCalc(float4 &pi, float4 &pj) {
	//Get Density from WPoly6
	float corr = WPoly6(make_float3(pi), make_float3(pj)) / sp.wQH;
	corr *= corr * corr * corr;
	return -sp.K * corr;
}

__device__ float3 xsphViscosity(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, int index) {
	float3 visc = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			float3 velocityDiff = velocities[neighbors[(index * sp.maxNeighbors) + i]] - velocities[index];
			velocityDiff *= WPoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]));
			visc += velocityDiff;
		}
	}

	return visc * sp.C;
}
#define alph 0.8f

__device__ void confineToBox(float4 &pos, float3 &vel) {


	if (pos.y < sp.bounds.y * 0.5f && pos.y > sp.bounds.y * 0.5f - 2 * sp.radius ) {
		if( !( (pos.x - sp.bounds.x* 0.2f) *(pos.x - sp.bounds.x* 0.2f) +(pos.z - sp.bounds.z* 0.2f) * (pos.z - sp.bounds.z* 0.2f) < 1.5* 1.5 *sp.radius * sp.radius) ){
				pos.y = sp.bounds.y * 0.5f + 0.001f;
				vel.y = 0.0f;
		}

	}
	else if(pos.y < sp.bounds.y * 0.5f - 2 * sp.radius)  // in the lower level
	{
	if (pos.x < sp.boundsMin.x) {
		vel.x = -alph * vel.x;
		pos.x = sp.boundsMin.x + 0.001f;
	} else if (pos.x > sp.bounds.x) {
		vel.x = -alph * vel.x;
		pos.x = sp.bounds.x - 0.001f;
	}

	if (pos.y < sp.boundsMin.y) {
		vel.y = -alph * vel.y;
		pos.y = sp.boundsMin.y + 0.001f;
	} 
		if (pos.z < sp.boundsMin.z) {
		vel.z = -alph * vel.z;
		pos.z =sp.boundsMin.z + 0.001f;
	} else if (pos.z > sp.bounds.z) {
		vel.z = -alph * vel.z;
		pos.z = sp.bounds.z - 0.001f;
	}
	}

	else if (pos.y > sp.bounds.y * 0.5f)  // upper level 
	{

	
		if (pos.x < sp.boundsMin.x) {
		vel.x = -alph * vel.x;
		pos.x = sp.boundsMin.x + 0.001f;
	} else if (pos.x > sp.bounds.x * 0.4f) {
		vel.x = -alph * vel.x;
		pos.x = sp.bounds.x  *0.4f- 0.001f;
		}
	if (pos.z < sp.boundsMin.z) {
		vel.z = -alph * vel.z;
		pos.z = sp.boundsMin.z + 0.001f;
	} else if (pos.z > sp.bounds.z * 0.4f) {
		vel.z = -alph * vel.z;
		pos.z = sp.bounds.z * 0.4f - 0.001f;
		}
	if (pos.y > sp.bounds.y) {
		vel.y = -alph * vel.y;
		pos.y = sp.bounds.y - 0.001f;
	}

	}
}

__device__ int3 getGridPos(float4 pos) {
	return make_int3(int(pos.x / sp.radius) % sp.gridWidth, int(pos.y / sp.radius) % sp.gridHeight, int(pos.z / sp.radius) % sp.gridDepth);
}

__device__ int getGridIndex(int3 pos) {
	return (pos.z * sp.gridHeight * sp.gridWidth) + (pos.y * sp.gridWidth) + pos.x;
}

//===========================================
// spring way of elasticity 
__global__ void springDisplacement(float4* newPos, int* phases, int* neighbors, int* numNeighbors ){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	for (int i = 0; i < numNeighbors[index]; i++) {
	if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
		float3 r_ij = make_float3(newPos[index] - newPos[i]);
		float rlength = length(r_ij);
		if (rlength > 0.0f)
		{
			float3 Displacement = deltaT * deltaT * sp.K_spring * ( 1 - sp.restDistance / sp.radius)* ( sp.restDistance - rlength) * (r_ij / rlength);
			newPos[index] -= make_float4(Displacement, 0.0f) *0.5f;
			newPos[i] += make_float4(Displacement, 0.0f) *0.5f;
		}
		}
	}
}



__global__ void applyExternalForce(float4* newPos, float3* velocities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	//update velocity vi = vi + dt * fExt
	velocities[index] += ((newPos[index].w > 0) ? 1 : 0) * sp.gravity * deltaT;

}

__global__ void predictPositions(float4* newPos, float3* velocities, float3* internal_force) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	//update velocity vi = vi + dt * fExt
	velocities[index] += ((newPos[index].w > 0) ? 1 : 0) * internal_force[index] * deltaT;


	//predict position x* = xi + dt * vi
	newPos[index] += make_float4(velocities[index] * deltaT, 0);

	confineToBox(newPos[index], velocities[index]);  // when collide the wall zero the velocity and set the position 
	
}

__global__ void clearNeighbors(int* numNeighbors, int* numContacts) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	numNeighbors[index] = 0;
	numContacts[index] = 0;
}

__global__ void clearGrid(int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.gridSize) return;

	gridCounters[index] = 0;
}

__global__ void updateGrid(float4* newPos, int* gridCells, int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	int3 pos = getGridPos(newPos[index]);
	int gIndex = getGridIndex(pos);

	int i = atomicAdd(&gridCounters[gIndex], 1);
	i = min(i, sp.maxParticles - 1);
	gridCells[gIndex * sp.maxParticles + i] = index;
}

__global__ void updateNeighbors(float4* newPos, int* phases, int* gridCells, int* gridCounters, int* neighbors, int* numNeighbors, int* contacts, int* numContacts) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;
	
	int3 pos = getGridPos(newPos[index]);
	int pIndex;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					int gIndex = getGridIndex(n);
					int cellParticles = min(gridCounters[gIndex], sp.maxParticles - 1);
					for (int i = 0; i < cellParticles; i++) {
						if (numNeighbors[index] >= sp.maxNeighbors) return;

						pIndex = gridCells[gIndex * sp.maxParticles + i];
						if (length(make_float3(newPos[index]) - make_float3(newPos[pIndex])) <= sp.radius) {
							neighbors[(index * sp.maxNeighbors) + numNeighbors[index]] = pIndex;
							numNeighbors[index]++;
							if (phases[index] == 0 && phases[pIndex] == 1 && numContacts[index] < sp.maxContacts) {
								contacts[index * sp.maxContacts + numContacts[index]] = pIndex;
								numContacts[index]++;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void particleCollisions(float4* newPos, int* contacts, int* numContacts, float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	for (int i = 0; i < numContacts[index]; i++) {
		int nIndex = contacts[index * sp.maxContacts + i];
		if (newPos[nIndex].w == 0) continue;
		float3 dir = make_float3(newPos[index] - newPos[nIndex]);
		float len = length(dir);
		float invMass = newPos[index].w + newPos[nIndex].w;
		float3 dp;
		if (len > sp.radius || len == 0.0f || invMass == 0.0f) dp = make_float3(0); // no contacting 
		else dp = (1 / invMass) * (len - sp.radius) * (dir / len);
		deltaPs[index] -= dp * newPos[index].w;
		buffer0[index]++;

		atomicAdd(&deltaPs[nIndex].x, dp.x * newPos[nIndex].w);
		atomicAdd(&deltaPs[nIndex].y, dp.y * newPos[nIndex].w);
		atomicAdd(&deltaPs[nIndex].z, dp.z * newPos[nIndex].w);
		atomicAdd(&buffer0[nIndex], 1);
	}
}

__global__ void calcDensities(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float* densities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	float rhoSum = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0)
			rhoSum += WPoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]));
	}

	densities[index] = rhoSum;
}

__global__ void calcLambda(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float* densities, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	float densityConstraint = (densities[index] / sp.restDensity) - 1;
	float3 gradientI = make_float3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			//Calculate gradient with respect to j
			float3 gradientJ = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) / sp.restDensity;

			//Add magnitude squared to sum
			sumGradients += pow(length(gradientJ), 2);
			gradientI += gradientJ;
		}
	}

	//Add the particle i gradient magnitude squared to sum
	sumGradients += pow(length(gradientI), 2);
	buffer0[index] = (-1 * densityConstraint) / (sumGradients + sp.lambdaEps);
}

__global__ void calcDeltaP(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;
	deltaPs[index] = make_float3(0);

	float3 deltaP = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			float lambdaSum = buffer0[index] + buffer0[neighbors[(index * sp.maxNeighbors) + i]];
			float sCorr = sCorrCalc(newPos[index], newPos[neighbors[(index * sp.maxNeighbors) + i]]);
			deltaP += WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]])) * (lambdaSum + sCorr);

		}
	}

	deltaPs[index] = deltaP / sp.restDensity;
}

__global__ void applyDeltaP(float4* newPos, float3* deltaPs, float* buffer0, int flag) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	if (buffer0[index] > 0 && flag == 1) newPos[index] += make_float4(deltaPs[index] / buffer0[index], 0);
	else if (flag == 0) newPos[index] += make_float4(deltaPs[index], 0);
	//newPos[index] += make_float4(deltaPs[index], 0);
}

__global__ void updateVelocities(float4* oldPos, float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, float3* deltaPs, float* density, mat3* elasticTerm) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	

	//set new velocity vi = (x*i - xi) / dt
	velocities[index] = (make_float3(newPos[index]) - make_float3(oldPos[index])) / deltaT;

	confineToBox(newPos[index], velocities[index]);

	//apply vorticity confinement
	velocities[index] += vorticityForce(newPos, velocities, phases, neighbors, numNeighbors, index) * deltaT;

	//apply elastic force to the velocity
	//velocities[index] += elasticForce(newPos, velocities, phases, neighbors, numNeighbors, density, elasticTerm, index) * deltaT;

	//trying the viscosity force from muller 2003 paper
	//velocities[index] += ViscosityForce(newPos, velocities, phases, neighbors, numNeighbors, density, index) * deltaT;

	//apply XSPH viscosity
	deltaPs[index] = xsphViscosity(newPos, velocities, phases, neighbors, numNeighbors, index);


	//update position xi = x*i
	oldPos[index] = newPos[index];
}

__global__ void updateXSPHVelocities(float4* newPos, float3* velocities, int* phases, float3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	velocities[index] += deltaPs[index] * deltaT;
}

__global__ void generateFoam(float4* newPos, float3* velocities, int* phases, float4* diffusePos, float3* diffuseVelocities, int* neighbors, int* numNeighbors, float* densities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0 || foamCount >= sp.numDiffuse) return;

	float velocityDiff = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		int nIndex = neighbors[(index * sp.maxNeighbors) + i];
		if (index != nIndex) {
			float wAir = WAirPotential(make_float3(newPos[index]), make_float3(newPos[nIndex]));
			float3 xij = normalize(make_float3(newPos[index] - newPos[nIndex]));
			float3 vijHat = normalize(velocities[index] - velocities[nIndex]);
			velocityDiff += length(velocities[index] - velocities[nIndex]) * (1 - dot(vijHat, xij)) * wAir;
		}
	}

	float ek = 0.5f * pow(length(velocities[index]), 2);
	float potential = velocityDiff * ek * max(1.0f - (1.0f * densities[index] / sp.restDensity), 0.0f);
	int nd = 0;
	if (potential > 0.5f) nd = min(20, (sp.numDiffuse - 1 - foamCount));
	if (nd <= 0) return;
	
	int count = atomicAdd(&foamCount, nd);
	count = min(count, sp.numDiffuse - 1);
	int cap = min(count + nd, sp.numDiffuse - 1);

	for (int i = count; i < cap; i++) {
		float rx = distr[i % 30] * sp.restDistance;
		float ry = distr[(i + 1) % 30] * sp.restDistance;
		float rz = distr[(i + 2) % 30] * sp.restDistance;
		int rd = distr[index % 30] > 0.5f ? 1 : -1;

		float3 xd = make_float3(newPos[index]) + make_float3(rx * rd, ry * rd, rz * rd);

		diffusePos[i] = make_float4(xd, 1);
		diffuseVelocities[i] = velocities[index];
	}

	if (foamCount >= sp.numDiffuse) atomicExch(&foamCount, sp.numDiffuse - 1);
}

__global__ void updateFoam(float4* newPos, float3* velocities, float4* diffusePos, float3* diffuseVelocities, int* gridCells, int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numDiffuse || diffusePos[index].w <= 0) return;

	confineToBox(diffusePos[index], diffuseVelocities[index]);

	int3 pos = getGridPos(diffusePos[index]);
	int pIndex;
	int fluidNeighbors = 0;
	float3 vfSum = make_float3(0.0f);
	float kSum = 0;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					int gIndex = getGridIndex(n);
					int cellParticles = min(gridCounters[gIndex], sp.maxParticles - 1);
					for (int i = 0; i < cellParticles; i++) {
						pIndex = gridCells[gIndex * sp.maxParticles + i];
						if (length(make_float3(diffusePos[index] - newPos[pIndex])) <= sp.radius) {
							fluidNeighbors++;
							float k = WPoly6(make_float3(diffusePos[index]), make_float3(newPos[pIndex]));
							vfSum += velocities[pIndex] * k;
							kSum += k;
						}
					}
				}
			}
		}
	}

	if (fluidNeighbors < 8) {
		//Spray
		//diffuseVelocities[index].x *= 0.8f;
		//diffuseVelocities[index].z *= 0.8f;
		diffuseVelocities[index] += sp.gravity * deltaT;
		diffusePos[index] += make_float4(diffuseVelocities[index] * deltaT, 0);
	} else {
		//Foam
		diffusePos[index] += make_float4((1.0f * (vfSum / kSum)) * deltaT, 0);
	}

	diffusePos[index].w -= deltaT;
	if (diffusePos[index].w <= 0.0f) {
		atomicSub(&foamCount, 1);
		diffusePos[index] = make_float4(0);
		diffuseVelocities[index] = make_float3(0);
	}
}


//========================================
// Cloth update kernels 
__global__ void clearDeltaP(float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	deltaPs[index] = make_float3(0);
	buffer0[index] = 0;
}

__global__ void solveDistance(float4* newPos, int* clothIndices, float* restLengths, float* stiffness, float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numConstraints) return;

	int p1 = clothIndices[2 * index];
	int p2 = clothIndices[2 * index + 1];

	float3 dir = make_float3(newPos[p1] - newPos[p2]);
	float len = length(dir);
	float invMass = newPos[p1].w + newPos[p2].w;
	float3 dp;
	if (len == 0.0f || invMass == 0.0f) dp = make_float3(0);
	else {
		if (stiffness[index] > 0) dp = (1 / invMass) * (len - restLengths[index]) * (dir / len) * (1.0f - pow(1.0f - stiffness[index], 1.0f / sp.numIterations));
		else if (len > restLengths[index]) {
			dp = (1 / invMass) * (len - restLengths[index]) * (dir / len) * (1.0f - pow(1.0f + stiffness[index], 1.0f / sp.numIterations));
		}
	}

	if (newPos[p1].w > 0) {
		atomicAdd(&deltaPs[p1].x, -dp.x * newPos[p1].w);
		atomicAdd(&deltaPs[p1].y, -dp.y * newPos[p1].w);
		atomicAdd(&deltaPs[p1].z, -dp.z * newPos[p1].w);
		atomicAdd(&buffer0[p1], 1);
	}

	if (newPos[p2].w > 0) {
		atomicAdd(&deltaPs[p2].x, dp.x * newPos[p2].w);
		atomicAdd(&deltaPs[p2].y, dp.y * newPos[p2].w);
		atomicAdd(&deltaPs[p2].z, dp.z * newPos[p2].w);
		atomicAdd(&buffer0[p2], 1);
	}
}

__global__ void updateClothVelocity(float4* oldPos, float4* newPos, float3* velocities, int* phases) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numCloth || phases[index] != 1) return;

	velocities[index] = make_float3(newPos[index] - oldPos[index]) / deltaT;
	oldPos[index] = newPos[index];
}

struct OBCmp {
	__host__ __device__
	bool operator()(const float4& a, const float4& b) const {
		return a.w > b.w;
	}
};

void updateWater(solver* s, int numIterations) {
	//------------------WATER-----------------
	for (int i = 0; i < numIterations; i++) {
		//Calculate fluid densities and store in densities
		calcDensities<<<dims, blockSize>>>(s->newPos, s->phases, s->neighbors, s->numNeighbors, s->densities);

		//Calculate all lambdas and store in buffer0
		calcLambda<<<dims, blockSize>>>(s->newPos, s->phases, s->neighbors, s->numNeighbors, s->densities, s->buffer0);

		// add spring displacement effect
		//springDisplacement<<<dims, blockSize>>>(s->newPos, s->phases, s->neighbors, s->numNeighbors);

		//calculate deltaP
		calcDeltaP<<<dims, blockSize>>>(s->newPos, s->phases, s->neighbors, s->numNeighbors, s->deltaPs, s->buffer0);

		//update position x*i = x*i + deltaPi
		applyDeltaP<<<dims, blockSize>>>(s->newPos, s->deltaPs, s->buffer0, 0);
	}

	//Update velocity, apply vorticity confinement, apply xsph viscosity, update position, update elastic force
	updateVelocities<<<dims, blockSize>>>(s->oldPos, s->newPos, s->velocities, s->phases, s->neighbors, s->numNeighbors, s->deltaPs, s->densities, s->elasticTerm);
	
	//Set new velocity
	updateXSPHVelocities<<<dims, blockSize>>>(s->newPos, s->velocities, s->phases, s->deltaPs);

	//generateFoam<<<dims, blockSize>>>(s->newPos, s->velocities, s->phases, s->diffusePos, s->diffuseVelocities, s->neighbors, s->numNeighbors, s->densities);
	//updateFoam<<<diffuseDims, blockSize>>>(s->newPos, s->velocities, s->diffusePos, s->diffuseVelocities, s->gridCells, s->gridCounters);
}

void updateCloth(solver* s, int numIterations) {
	clearDeltaP<<<dims, blockSize>>>(s->deltaPs, s->buffer0);

	for (int i = 0; i < numIterations; i++) {
		solveDistance<<<clothDims, blockSize>>>(s->newPos, s->clothIndices, s->restLengths, s->stiffness, s->deltaPs, s->buffer0);
		applyDeltaP<<<dims, blockSize>>>(s->newPos, s->deltaPs, s->buffer0, 1);
	}

	updateClothVelocity<<<dims, blockSize>>>(s->oldPos, s->newPos, s->velocities, s->phases);
}

//==============================================
// functions for particle based solver
__global__ void calcInternalForce(float4* oldPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, float* density, mat3* elasticTerm, float3* internalForce) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);  // for every particle i
	if (index >= sp.numParticles || phases[index] != 0) return;

	// clear last internal force
	internalForce[index] = make_float3(0);
	
	//add pressure force to internal force
	//internalForce[index] += Pressureforce(oldPos, velocities, phases, neighbors, numNeighbors, density, index);
	
	//add viscosity-force to internal force
	internalForce[index] += ViscosityForce(oldPos, velocities, phases, neighbors, numNeighbors, density, index);
	
	//add elastic force to internal force 
	internalForce[index] += elasticForce(oldPos, velocities, phases, neighbors, numNeighbors, density, elasticTerm, index);

}


//=================================
// try the particle based solver scheme
/*
void update(solver* s, solverParams* sp) {
	
	//Update neighbors first 
	clearNeighbors<<<dims, blockSize>>>(s->numNeighbors, s->numContacts);
	clearGrid<<<gridDims, blockSize>>>(s->gridCounters);
	updateGrid<<<dims, blockSize>>>(s->oldPos, s->gridCells, s->gridCounters);
	updateNeighbors<<<dims, blockSize>>>(s->oldPos, s->phases, s->gridCells, s->gridCounters, s->neighbors, s->numNeighbors, s->contacts, s->numContacts);

	//Calculate fluid densities and store in densities
	calcDensities<<<dims, blockSize>>>(s->oldPos, s->phases, s->neighbors, s->numNeighbors, s->densities);

	for (int i = 0; i < sp->numIterations; i++) {
	calcInternalForce<<<dims, blockSize>>>(s->oldPos,s->velocities, s->phases, s->neighbors, s->numNeighbors, s->densities, s->elasticTerm, s->Force_internal);
	}
	//Predict positions and update velocity
	// maybe wrong with the confin box function
	predictPositions<<<dims, blockSize>>>(s->newPos, s->velocities, s->Force_internal);

	//do the particle collisions 
	for (int i = 0; i < sp->numIterations; i++) {
		clearDeltaP<<<dims, blockSize>>>(s->deltaPs, s->buffer0);
		particleCollisions<<<dims, blockSize>>>(s->newPos, s->contacts, s->numContacts, s->deltaPs, s->buffer0);
		applyDeltaP<<<dims, blockSize>>>(s->newPos, s->deltaPs, s->buffer0, 1);
	}

	//only update position here !!!
	//Update velocity, apply vorticity confinement, apply xsph viscosity, update position, update elastic force
	updateVelocities<<<dims, blockSize>>>(s->oldPos, s->newPos, s->velocities, s->phases, s->neighbors, s->numNeighbors, s->deltaPs, s->densities, s->elasticTerm);
	//Set new velocity
	updateXSPHVelocities<<<dims, blockSize>>>(s->newPos, s->velocities, s->phases, s->deltaPs);

	thrust::device_ptr<float4> devPtr = thrust::device_pointer_cast(s->diffusePos);
	thrust::sort(devPtr, devPtr + sp->numDiffuse, OBCmp());
}
*/

void update(solver* s, solverParams* sp) {
	// apply external force 
	applyExternalForce<<<dims, blockSize>>>(s->newPos, s->velocities);

	calcInternalForce<<<dims, blockSize>>>(s->newPos, s->velocities, s->phases, s->neighbors, s->numNeighbors, s->densities, s->elasticTerm, s->Force_internal);
	
	//Predict positions and update velocity
	predictPositions<<<dims, blockSize>>>(s->newPos, s->velocities, s->Force_internal);

	//Update neighbors
	clearNeighbors<<<dims, blockSize>>>(s->numNeighbors, s->numContacts);
	clearGrid<<<gridDims, blockSize>>>(s->gridCounters);
	updateGrid<<<dims, blockSize>>>(s->newPos, s->gridCells, s->gridCounters);
	updateNeighbors<<<dims, blockSize>>>(s->newPos, s->phases, s->gridCells, s->gridCounters, s->neighbors, s->numNeighbors, s->contacts, s->numContacts);
	
	for (int i = 0; i < sp->numIterations; i++) {
		clearDeltaP<<<dims, blockSize>>>(s->deltaPs, s->buffer0);
		particleCollisions<<<dims, blockSize>>>(s->newPos, s->contacts, s->numContacts, s->deltaPs, s->buffer0);
		applyDeltaP<<<dims, blockSize>>>(s->newPos, s->deltaPs, s->buffer0, 1);
	}
	
	//Solve constraints
	updateWater(s, sp->numIterations);
	thrust::device_ptr<float4> devPtr = thrust::device_pointer_cast(s->diffusePos);
	thrust::sort(devPtr, devPtr + sp->numDiffuse, OBCmp());

	//comment cloth updater for speed?
	//updateCloth(s, sp->numIterations); 
}



void setParams(solverParams *tempParams) {
	dims = int(ceil(tempParams->numParticles / blockSize + 0.5f));
	diffuseDims = int(ceil(tempParams->numDiffuse / blockSize + 0.5f));
	clothDims = int(ceil(tempParams->numConstraints / blockSize + 0.5f));
	gridDims = int(ceil(tempParams->gridSize / blockSize + 0.5f));
	cudaCheck(cudaMemcpyToSymbol(sp, tempParams, sizeof(solverParams)));
}

#endif