#pragma once
using namespace std;
#include "octreeNode.h"
#include "vertex.h"
#include "triangleTable.h"


class Octree {
public:
	
	// Octree Marching Cubes function, only need to call this in OpenGL program for producing marching cubes surface, except generating obj file
    // All input vector (list or map) parameters should be empty except SDF data list
    // once you call this function, then you can take grid coordinates list and triangle vertex list for OpenGL rendering
    static void octreeMarchingCubes(unordered_map<int, vertex>& gridCoordinatesMap, vertex * triangleVertexList,
                                    vertex * triVertexUniqueList, uint32_t * lineList, const float * SDFValues, int depth, int height,
                                    int length, int width);

	// build the octree and return the root node
	static OctreeNode buildOctree(const float * SDFValues, int minX, int minY, int minZ, int maxX, int maxY, int maxZ, int depth, int height, int length, int width);

	// judge whether the octree node should be subdivided or not
	static bool needSubdivision(const float * SDFValues, int minX, int minY, int minZ, int maxX, int maxY, int maxZ, int height, int length, int width);

	// Traverse an octree to compute grid vertices coordinates and triangle vertices for each leaf node
    // Because there may be repeated grid coordinates, so we use unorder_map instead of vector list for gridCoordinates
    static void traverseOctree(unordered_map<int, vertex>& gridCoordinatesMap, vertex* triangleVertexList, vertex* triVertexUniqueList, uint32_t* lineList, const float* SDFValues, const OctreeNode& currentNode, int height, int length, int width);

	// calculate the triangle vertex on one edge of the cube, v1 and v2 are two endpoints of the edge (same as the one in original marching cubes algorithm)
	static vertex calculateIntersection(vertex v1, vertex v2, float isoValue);
	
	// calculate the normal vector for each grid coordinate, using SDF values (slightly modified based on the one in original marching cubes algorithm)
	static void calculateNormalForGridVertex(const float * SDFValues, int x, int y, int z, int height, int width, int length, float * normal);

	// eliminate the repeated vertices shared by two cubes on the same edge
	// same as the post processing function in original marching cubes (mcFunctions.cpp)
	static void postProcessingForTriVertices(vertex * triVertexUniqueList);

	static void generateIndices(const vertex * triangleVertexList, const vertex * triVertexUniqueList, uint32_t * indices);
	
	static void setNormal(vertex& v, const float newNormal[3]);

    // record the size of list, each entry is one geometry
	static uint32_t triangleVertexListSize[5];
    static uint32_t triVertexUniqueListSize[5];
    static uint32_t lineListSize[5];
    static uint32_t currentGeometry;
};
