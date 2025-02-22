#include "octreeNode.h"

using namespace std;

// OctreeNode constructor
OctreeNode::OctreeNode(int minX, int minY, int minZ, int maxX, int maxY, int maxZ, bool leaf) {
	

	maxIndexX = maxX;
	maxIndexY = maxY;
	maxIndexZ = maxZ;
	minIndexX = minX;
	minIndexY = minY;
	minIndexZ = minZ;
	isLeaf = leaf;

}

// Add for recursively cleaning children
// Because the children now is pointer array and not vector any more
// We need to clean it manually
void OctreeNode::cleanAllChildren()
{
    // Loop over all possible children
    for (int i = 0; i < 8; i++)
    {
        // If the child exists, recursively clean its subtree first
        if (children[i] != nullptr)
        {
            children[i]->cleanAllChildren();
            delete children[i];    // Delete the child node
            children[i] = nullptr; 
        }
    }
}

