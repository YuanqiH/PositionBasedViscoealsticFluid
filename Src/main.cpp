/*

this simulation is build based on code from position based fluids by Korzen
that is compilable and runnable out of the box. 
The original repository is here https://github.com/korzen/PositionBasedFluids


*/

#define GLEW_DYNAMIC
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <GLFW\glfw3.h>
#include "ParticleSystem.h"
#include "Renderer.h"
#include "Scene.hpp"

#include "helper_timer.h"

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include <IL\il.h>
#include <IL\ilut.h>


#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }

static const int width = 1280;
static const int height = 720;
static const GLfloat lastX = (width / 2);
static const GLfloat lastY = (height / 2);
static float deltaTime = 0.0f;
static float lastFrame = 0.0f;
static int w = 0;
static bool ss = false;

void initializeState(ParticleSystem &system, tempSolver &tp, solverParams &tempParams);
void handleInput(GLFWwindow* window, ParticleSystem &system, Camera &cam);
void saveVideo();
void mainUpdate(ParticleSystem &system, Renderer &render, Camera &cam, tempSolver &tp, solverParams &tempParams);

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;
GLFWwindow* window;
int numberParticle;

void computeFPS()
{
    //frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "CUDA Particles ( %d particles): %3.1f fps",numberParticle, ifps);

		glfwSetWindowTitle(window, fps);
        fpsCount = 0;

        fpsLimit = (int)fmaxf(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}


int main() {
	//Checks for memory leaks in debug mode
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	cudaGLSetGLDevice(0);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	window = glfwCreateWindow(width, height, "Position Based Fluids", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	//Set callbacks for keyboard and mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	
	glewExperimental = GL_TRUE;
	glewInit();
	glGetError();
	
	// Define the viewport dimensions
	glViewport(0, 0, width, height);

	ilutInit();
	ilInit();
	ilutRenderer(ILUT_OPENGL);
	
	Camera cam = Camera();
	Renderer render = Renderer();
	ParticleSystem system = ParticleSystem();
	tempSolver tp;
	solverParams tempParams;
	initializeState(system, tp, tempParams);
	render.initVBOS(tempParams.numParticles, tempParams.numDiffuse, tp.triangles);
	sdkCreateTimer(&timer);  // create the timer

	  int nDevices;

	cudaGetDeviceCount(&nDevices);
	 for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	  }


	while (!glfwWindowShouldClose(window)) {
		//Set frame times
		float currentFrame = float(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		/*
		char fps[256];
		float ifps = 1.f / (deltaTime / 1000.f);
        sprintf(fps, "CUDA Particles ( particles): %3.1f fps", ifps);
		
		glfwSetWindowTitle(window, fps);
		*/

		// Check and call events
		glfwPollEvents();
		handleInput(window, system, cam);

		sdkStartTimer(&timer);
		
		//Update physics and render
		mainUpdate(system, render, cam, tp, tempParams);
		numberParticle = tempParams.numParticles;
		//saveVideo(); // need for video
		sdkStopTimer(&timer);
		
		// Swap the buffers
		glfwSwapBuffers(window);

		glfwSetCursorPos(window, lastX, lastY);

		computeFPS();
	}

	glfwTerminate();

	//cudaDeviceReset();

	return 0;
}




void handleInput(GLFWwindow* window, ParticleSystem &system, Camera &cam) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cam.wasdMovement(FORWARD, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cam.wasdMovement(BACKWARD, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cam.wasdMovement(RIGHT, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cam.wasdMovement(LEFT, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		cam.wasdMovement(UP, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		cam.wasdMovement(DOWN, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
		system.running = false;

	if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS)
		system.running = true;

	if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
		system.moveWall = true;

	if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
		system.moveWall = false;

	if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS)
		system.moveXwallLower = true;

	if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
		system.moveXwallLower = false;



	if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS)
		ss = true;

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
		cam.mouseMovement((float(xpos) - lastX), (lastY - float(ypos)), deltaTime);
}
/*
void saveVideo() {
	if (ss == true) {
		ILuint imageID = ilGenImage();
		ilBindImage(imageID);
		ilutGLScreen();
		ilEnable(IL_FILE_OVERWRITE);
		std::string str = std::to_string(w) + ".png";
		const char * c = str.c_str();
		std::cout << c << std::endl;
		//ilSaveImage(c);
		//ilutGLScreenie();
		ilDeleteImage(imageID);
		w++;
	}
}
*/
void initializeState(ParticleSystem &system, tempSolver &tp, solverParams &tempParams) {
	DamBreak scene("DamBreak");
	//FluidCloth scene("FluidCloth");
	scene.init(&tp, &tempParams); // initial all the parameters
	system.initialize(tp, tempParams); // copy all the parameters into the system for cuda device
}

void mainUpdate(ParticleSystem &system, Renderer &render, Camera &cam, tempSolver &tp, solverParams &tempParams) {
	
	
	//Step physics
	system.updateWrapper(tempParams);

	//Update the VBO
	void* positionsPtr;
	void* diffusePosPtr;
	void* diffuseVelPtr;
	cudaCheck(cudaGraphicsMapResources(3, render.resources));
	size_t size;
	cudaGraphicsResourceGetMappedPointer(&positionsPtr, &size, render.resources[0]);
	cudaGraphicsResourceGetMappedPointer(&diffusePosPtr, &size, render.resources[1]);
	cudaGraphicsResourceGetMappedPointer(&diffuseVelPtr, &size, render.resources[2]);
	system.getPositions((float*)positionsPtr, tempParams.numParticles);
	system.getDiffuse((float*)diffusePosPtr, (float*)diffuseVelPtr, tempParams.numDiffuse);
	cudaGraphicsUnmapResources(3, render.resources, 0);

	//Render
	render.run(tempParams.numParticles, tempParams.numDiffuse, tempParams.numCloth, tp.triangles, cam);
}