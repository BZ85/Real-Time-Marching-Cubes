#pragma once

using namespace std;

struct OctreeNode {

	// use the integer index as the bounds for octree node
	// it is more easy for input discrete SDF values 
	int maxIndexX, maxIndexY, maxIndexZ;
	int minIndexX, minIndexY, minIndexZ;

	OctreeNode* children[8] = { nullptr };

	bool isLeaf = false;

	OctreeNode(int minX, int minY, int minZ, int maxX, int maxY, int maxZ, bool leaf);
	
	void cleanAllChildren();

};
