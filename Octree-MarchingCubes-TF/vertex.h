#pragma once
#include <unordered_map>
#include <unordered_set>
using namespace std;

// define the vertex struct and its hash function for the program (used in both original and octree marching cubes)
struct vertex {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float scalar = 0.0f;
    float normal[3] = {};

	vertex() {};
	vertex(float px, float py, float pz, float pScalar) {
		x = px; y = py; z = pz; scalar = pScalar;
	};


	// overload the == operator for comparing two vertices
	// only using the position to compare is enough
	bool operator==(const vertex& v1) const {

		float difference = (this->x - v1.x) * (this->x - v1.x) + (this->y - v1.y) * (this->y - v1.y) + (this->z - v1.z) * (this->z - v1.z);
		return difference <= 0.000001;

	}
};



// create the hash function for vertex struct, this hash function is used in generateOBJfile function and postprocessing
template <>
struct hash<vertex> {
	size_t operator()(const vertex& v) const {

		// Because there are some inaccuracy in floating points
		// like 0.575559999 and 0.57556000 should be same coordinate but can have different hash value, thereby are considered as different coordinates
		// so I multiply this precision to the xyz coordinate for hash vaule
		// if the first four decimals of the xyz coordinates are same
		// then the hash values of two vertices will be the same, and these two vertices are considered repeated
		// therefore, 0.575559999 and 0.57556000 will be considered as same (repeated) coordinate
		// so that the floating number inaccuracy problem can be solved
		float precision = 10000.0f;

		// cast the floating number of vertex world coordinates to integer, and keep four decimals places 
		size_t h1 = hash<float>{}(static_cast<int>(v.x * precision));
		size_t h2 = hash<float>{}(static_cast<int>(v.y * precision));
		size_t h3 = hash<float>{}(static_cast<int>(v.z * precision));

		return h1 ^ (h2 << 1) ^ (h3 << 2);

		// making hash value all be 0, degrading hash table to be linked list 
		//so that all vertices will use == to compare, some closed vertices will be merged, but the peformance is a problem, so I comment this way
		//return 0;
	}
};
