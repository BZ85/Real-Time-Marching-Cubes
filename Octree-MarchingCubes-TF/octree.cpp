/*
 Marching Cubes Research Project, Computer Science, USC
 Octree Marching Cubes algorithm part
 Author Name: Xinjie Zhu
 Advised by Professor Oded Stein
*/

#include "octree.h"

//vector<uint32_t> Octree::lineList = {};

uint32_t Octree::triangleVertexListSize[5] = {};
uint32_t Octree::triVertexUniqueListSize[5] = {};
uint32_t Octree::lineListSize[5] = {};
uint32_t  Octree::currentGeometry = 0;

// Octree Marching Cubes function, only need to call this in OpenGL program for producing marching cubes surface, except generating obj file
// All input vector (list or map) parameters should be empty except SDF data list
// once you call this function, then you can take grid coordinates list and triangle vertex list for OpenGL rendering
void Octree::octreeMarchingCubes(unordered_map<int, vertex>& gridCoordinatesMap, vertex * triangleVertexList, vertex * triVertexUniqueList, uint32_t * lineList, const float * SDFValues, int depth, int height, int length, int width)
{
    OctreeNode rootNode = buildOctree(SDFValues, 0, 0, 0, width - 1, height - 1, length - 1, depth, height, length, width);
    traverseOctree(gridCoordinatesMap, triangleVertexList, triVertexUniqueList, lineList, SDFValues, rootNode, height, length, width);    
    postProcessingForTriVertices(triVertexUniqueList);

    rootNode.cleanAllChildren(); // added because the children now are all pointers
}


// build the octree and return the root node, based on input discrete SDF Values
OctreeNode Octree::buildOctree(const float * SDFValues, int minX, int minY, int minZ, int maxX, int maxY, int maxZ, int depth, int height, int length, int width)
{

    OctreeNode currentNode(minX, minY, minZ, maxX, maxY, maxZ, false);
    int midX = (maxX + minX) / 2;
    int midY = (maxY + minY) / 2;
    int midZ = (maxZ + minZ) / 2;

    // add subdividing condition
    bool ifSubdivision = needSubdivision(SDFValues, minX, minY, minZ, maxX, maxY, maxZ, height, length, width);

    if (!ifSubdivision || depth == 0) {

        currentNode.isLeaf = true;
        return currentNode;
    }
    OctreeNode tempNode = buildOctree(SDFValues, minX, minY, minZ, midX, midY, midZ, depth - 1, height, length, width);
    currentNode.children[0] = new OctreeNode(tempNode);
    tempNode = buildOctree(SDFValues, midX, minY, minZ, maxX, midY, midZ, depth - 1, height, length, width);
    currentNode.children[1] = new OctreeNode(tempNode);
    tempNode = buildOctree(SDFValues, minX, midY, minZ, midX, maxY, midZ, depth - 1, height, length, width);
    currentNode.children[2] = new OctreeNode(tempNode);
    tempNode = buildOctree(SDFValues, midX, midY, minZ, maxX, maxY, midZ, depth - 1, height, length, width);
    currentNode.children[3] = new OctreeNode(tempNode);

    tempNode = buildOctree(SDFValues, minX, minY, midZ, midX, midY, maxZ, depth - 1, height, length, width);
    currentNode.children[4] = new OctreeNode(tempNode);
    tempNode = buildOctree(SDFValues, midX, minY, midZ, maxX, midY, maxZ, depth - 1, height, length, width);
    currentNode.children[5] = new OctreeNode(tempNode);
    tempNode = buildOctree(SDFValues, minX, midY, midZ, midX, maxY, maxZ, depth - 1, height, length, width);
    currentNode.children[6] = new OctreeNode(tempNode);
    tempNode = buildOctree(SDFValues, midX, midY, midZ, maxX, maxY, maxZ, depth - 1, height, length, width);
    currentNode.children[7] = new OctreeNode(tempNode);

    return currentNode;
}

// judge whether the octree node should be subdivided or not
bool Octree::needSubdivision(const float * SDFValues, int minX, int minY, int minZ, int maxX, int maxY, int maxZ, int height, int length, int width)
{


    bool ifPositive = false;
    bool ifNegative = false;

    // Go through the bounds of the octree node, to judge whether it is needed to be subdivided
    for(int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
            for (int z = minZ; z <= maxZ; z++) {

                // check each SDF Value between the bounds
               
                int index = y * width * length + x * length + z;
                float value = SDFValues[index];
                if (value > 0) ifPositive = true;
                if (value < 0) ifNegative = true;
               
                // if both positive and negative SDF value occur between bounds, then break the loop and return true (need subdivision)
                if (ifPositive && ifNegative) return true;
            }

    return false;
}

// Traverse each node of octree, and create grid coordinates and surface triangles for each leaf node
// Because there may be repeated grid coordinates, so we use unorder_map instead of vector list for gridCoordinates
void Octree::traverseOctree(unordered_map<int, vertex>& gridCoordinatesMap, vertex* triangleVertexList, vertex* triVertexUniqueList,
                            uint32_t * lineList, const float * SDFValues, const OctreeNode& currentNode, int height,
                            int length, int width)
{
  
    
    // if currentNode is leaf, then compute the grid vertices and triangle vertices of the node (bound box)
    if (currentNode.isLeaf) { 

      // get the bound index of the leaf node
      int minX = currentNode.minIndexX;
      int minY = currentNode.minIndexY;
      int minZ = currentNode.minIndexZ;

      int maxX = currentNode.maxIndexX;
      int maxY = currentNode.maxIndexY;
      int maxZ = currentNode.maxIndexZ;

   
      // compute the world coordinates based on index of bounds' vertices
      vertex currentVertex;
      float maxXCoordiate = 40.0 * maxX / (width - 1);
      float minXCoordiate = 40.0 * minX / (width - 1);

      float maxYCoordiate = 40.0 * maxY / (height - 1);
      float minYCoordiate = 40.0 * minY / (height - 1);

      float maxZCoordiate = -40.0 * maxZ / (length - 1);
      float minZCoordiate = -40.0 * minZ / (length - 1);
      
      // compute the index of the grid coordinates in the whole grid mesh (the index same as in the SDF value list)
      int v0 = minY * width * length + minX * length + minZ;
      int v1 = minY * width * length + minX * length + maxZ;
      int v2 = minY * width * length + maxX * length + maxZ;
      int v3 = minY * width * length + maxX * length + minZ;
          
      int v4 = maxY * width * length + minX * length + minZ;
      int v5 = maxY * width * length + minX * length + maxZ;
      int v6 = maxY * width * length + maxX * length + maxZ;
      int v7 = maxY * width * length + maxX * length + minZ;

      // using hash map to avoid creating repeated grid coordinates
      // because the leaf nodes may share some of the grid coordinates
      // the integer index of the grid coordinate is the key, this index (key) is unique for every grid vertex
      // if this grid vertex hasn't been added in the map, then create it and compute normal vector of it
      if (gridCoordinatesMap.find(v0) == gridCoordinatesMap.end()) { 
          gridCoordinatesMap[v0] = vertex(minXCoordiate, minYCoordiate, minZCoordiate, SDFValues[v0]); 
         // gridCoordinatesMap[v0].normal = calculateNormalForGridVertex(SDFValues, minX, minY, minZ, height, width, length); 
          float normal[3] = {};
          calculateNormalForGridVertex(SDFValues, minX, minY, minZ, height, width, length, normal);
          setNormal(gridCoordinatesMap[v0], normal);
      }
      if (gridCoordinatesMap.find(v1) == gridCoordinatesMap.end()) {
          gridCoordinatesMap[v1] = vertex(minXCoordiate, minYCoordiate, maxZCoordiate, SDFValues[v1]); 
        //  gridCoordinatesMap[v1].normal = calculateNormalForGridVertex(SDFValues, minX, minY, maxZ, height, width, length);
          float normal[3] = {};
          calculateNormalForGridVertex(SDFValues, minX, minY, maxZ, height, width, length, normal);
          setNormal(gridCoordinatesMap[v1], normal);
      }
    
      if (gridCoordinatesMap.find(v2) == gridCoordinatesMap.end()) {
          gridCoordinatesMap[v2] = vertex(maxXCoordiate, minYCoordiate, maxZCoordiate, SDFValues[v2]); 
       //   gridCoordinatesMap[v2].normal = calculateNormalForGridVertex(SDFValues, maxX, minY, maxZ, height, width, length);
          float normal[3] = {};
          calculateNormalForGridVertex(SDFValues, maxX, minY, maxZ, height, width, length, normal);
          setNormal(gridCoordinatesMap[v2], normal);
      }
      
      if (gridCoordinatesMap.find(v3) == gridCoordinatesMap.end()) {
          gridCoordinatesMap[v3] = vertex(maxXCoordiate, minYCoordiate, minZCoordiate, SDFValues[v3]);
        //  gridCoordinatesMap[v3].normal = calculateNormalForGridVertex(SDFValues, maxX, minY, minZ, height, width, length);
          float normal[3] = {};
          calculateNormalForGridVertex(SDFValues, maxX, minY, minZ, height, width, length, normal);
          setNormal(gridCoordinatesMap[v3], normal);
    
      }

      if (gridCoordinatesMap.find(v4) == gridCoordinatesMap.end()) {
          gridCoordinatesMap[v4] = vertex(minXCoordiate, maxYCoordiate, minZCoordiate, SDFValues[v4]);
         // gridCoordinatesMap[v4].normal = calculateNormalForGridVertex(SDFValues, minX, maxY, minZ, height, width, length);
          
          float normal[3] = {};
          calculateNormalForGridVertex(SDFValues, minX, maxY, minZ, height, width, length, normal);
          setNormal(gridCoordinatesMap[v4], normal);
      
      }
      if (gridCoordinatesMap.find(v5) == gridCoordinatesMap.end()) {
          gridCoordinatesMap[v5] = vertex(minXCoordiate, maxYCoordiate, maxZCoordiate, SDFValues[v5]);
         // gridCoordinatesMap[v5].normal = calculateNormalForGridVertex(SDFValues, minX, maxY, maxZ, height, width, length);
          float normal[3] = {};
          calculateNormalForGridVertex(SDFValues, minX, maxY, maxZ, height, width, length, normal);
          setNormal(gridCoordinatesMap[v5], normal);
      }
      if (gridCoordinatesMap.find(v6) == gridCoordinatesMap.end()) {
          gridCoordinatesMap[v6] = vertex(maxXCoordiate, maxYCoordiate, maxZCoordiate, SDFValues[v6]);
        //  gridCoordinatesMap[v6].normal = calculateNormalForGridVertex(SDFValues, maxX, maxY, maxZ, height, width, length);
          float normal[3] = {};
          calculateNormalForGridVertex(SDFValues, maxX, maxY, maxZ, height, width, length, normal);
          setNormal(gridCoordinatesMap[v6], normal);
      }
      if (gridCoordinatesMap.find(v7) == gridCoordinatesMap.end()) {
          gridCoordinatesMap[v7] = vertex(maxXCoordiate, maxYCoordiate, minZCoordiate, SDFValues[v7]);
         // gridCoordinatesMap[v7].normal = calculateNormalForGridVertex(SDFValues, maxX, maxY, minZ, height, width, length);
          float normal[3] = {};
          calculateNormalForGridVertex(SDFValues, maxX, maxY, minZ, height, width, length, normal);
          setNormal(gridCoordinatesMap[v7], normal);
      }

      /* Add the grid coordinates for rendering the wireframe(not necessary for the algorithm, so I use global list here)
       This part of the code can be deleted if there is no need for rendering wire frames
      */
      lineList[lineListSize[currentGeometry]++] = v0;
      lineList[lineListSize[currentGeometry]++] = v1;
      lineList[lineListSize[currentGeometry]++] = v2;
      lineList[lineListSize[currentGeometry]++] = v3;
      lineList[lineListSize[currentGeometry]++] = v4;
      lineList[lineListSize[currentGeometry]++] = v5;
      lineList[lineListSize[currentGeometry]++] = v6;
      lineList[lineListSize[currentGeometry]++] = v7; 
       
      /******************************************************************************************************************/
   
      int caseIndex = 0;
      float isoValue = 0; // isoValue is 0 because of input is SDF data
  

      /* Following part of code references Paul Bourke (1994)
                 url: https://paulbourke.net/geometry/polygonise/
                 similar to the part in original marching cubes
       ************************************************************/

      if (gridCoordinatesMap[v0].scalar < isoValue) caseIndex += 1;
      if (gridCoordinatesMap[v1].scalar < isoValue) caseIndex += 2;
      if (gridCoordinatesMap[v2].scalar < isoValue) caseIndex += 4;
      if (gridCoordinatesMap[v3].scalar < isoValue) caseIndex += 8;

      if (gridCoordinatesMap[v4].scalar < isoValue) caseIndex += 16;
      if (gridCoordinatesMap[v5].scalar < isoValue) caseIndex += 32;
      if (gridCoordinatesMap[v6].scalar < isoValue) caseIndex += 64;
      if (gridCoordinatesMap[v7].scalar < isoValue) caseIndex += 128;

      // use case index to fetch an hexadecimal number
      //  the Nth bit of hexadecimal number determines whether the vertex on the Nth edge in current cube will be used to render triangle
      // if the number on Nth bit is 1, then calculate the intersect vertex on this edge, otherwise not
      vertex defaultV;
      vertex triangleVertexListForThisCube[12] = {};

      if (edgeTable[caseIndex] & 1) triangleVertexListForThisCube[0] = calculateIntersection(gridCoordinatesMap[v0], gridCoordinatesMap[v1], isoValue);
      if (edgeTable[caseIndex] & 2) triangleVertexListForThisCube[1] = calculateIntersection(gridCoordinatesMap[v1], gridCoordinatesMap[v2], isoValue);
      if (edgeTable[caseIndex] & 4) triangleVertexListForThisCube[2] = calculateIntersection(gridCoordinatesMap[v2], gridCoordinatesMap[v3], isoValue);
      if (edgeTable[caseIndex] & 8) triangleVertexListForThisCube[3] = calculateIntersection(gridCoordinatesMap[v3], gridCoordinatesMap[v0], isoValue);

      if (edgeTable[caseIndex] & 16) triangleVertexListForThisCube[4] = calculateIntersection(gridCoordinatesMap[v4], gridCoordinatesMap[v5], isoValue);
      if (edgeTable[caseIndex] & 32) triangleVertexListForThisCube[5] = calculateIntersection(gridCoordinatesMap[v5], gridCoordinatesMap[v6], isoValue);
      if (edgeTable[caseIndex] & 64) triangleVertexListForThisCube[6] = calculateIntersection(gridCoordinatesMap[v6], gridCoordinatesMap[v7], isoValue);
      if (edgeTable[caseIndex] & 128) triangleVertexListForThisCube[7] = calculateIntersection(gridCoordinatesMap[v7], gridCoordinatesMap[v4], isoValue);

      if (edgeTable[caseIndex] & 256) triangleVertexListForThisCube[8] = calculateIntersection(gridCoordinatesMap[v0], gridCoordinatesMap[v4], isoValue);
      if (edgeTable[caseIndex] & 512) triangleVertexListForThisCube[9] = calculateIntersection(gridCoordinatesMap[v1], gridCoordinatesMap[v5], isoValue);
      if (edgeTable[caseIndex] & 1024) triangleVertexListForThisCube[10] = calculateIntersection(gridCoordinatesMap[v2], gridCoordinatesMap[v6], isoValue);
      if (edgeTable[caseIndex] & 2048) triangleVertexListForThisCube[11] = calculateIntersection(gridCoordinatesMap[v3], gridCoordinatesMap[v7], isoValue);

      // push the triangle vertices into triVertexUniqueList, preparing for outputing the obj file
      if (edgeTable[caseIndex] & 1)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[0];
      if (edgeTable[caseIndex] & 2)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[1];
      if (edgeTable[caseIndex] & 4)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[2];
      if (edgeTable[caseIndex] & 8)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[3];

      if (edgeTable[caseIndex] & 16)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[4];
      if (edgeTable[caseIndex] & 32)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[5];
      if (edgeTable[caseIndex] & 64)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[6];
      if (edgeTable[caseIndex] & 128)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[7];

      if (edgeTable[caseIndex] & 256)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[8];
      if (edgeTable[caseIndex] & 512)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[9];
      if (edgeTable[caseIndex] & 1024)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[10];
      if (edgeTable[caseIndex] & 2048)
          triVertexUniqueList[triVertexUniqueListSize[currentGeometry]++] = triangleVertexListForThisCube[11];
      /*************************************************************************************************/


      /* The following for loop partially references Paul Bourke (1994)
                 url: https://paulbourke.net/geometry/polygonise/ */
                 // Get the triangle vertex rendering order from triTable
                 // And push them into triangleVertexList
      for (int i = 0; triTable[caseIndex][i] != -1; i++) {

          int index = triTable[caseIndex][i];

          vertex triV;
          triV.x = triangleVertexListForThisCube[index].x;
          triV.y = triangleVertexListForThisCube[index].y;
          triV.z = triangleVertexListForThisCube[index].z;
         
          triV.normal[0] = (triangleVertexListForThisCube[index].normal[0]);
          triV.normal[1] = (triangleVertexListForThisCube[index].normal[1]);
          triV.normal[2] = (triangleVertexListForThisCube[index].normal[2]);

          
          triangleVertexList[triangleVertexListSize[currentGeometry]++] = triV;

      }

      // return the function and will not traverse anymore
        return; 
    }

    // traverse the children of the current octree node, if it is not the leaf node
    for (int i = 0; i < 8; i++)
    {
        traverseOctree(gridCoordinatesMap, triangleVertexList, triVertexUniqueList, lineList, SDFValues, *(currentNode.children[i]), height, length, width);
    }


}

// calculate the intersection point using interpolation
// same as the function in original marching cubes (mcFunctions.cpp) 
vertex Octree::calculateIntersection(vertex v1, vertex v2, float isoValue)
{
    /* Following part of code references Paul Bourke (1994)
       url: https://paulbourke.net/geometry/polygonise/
   ***********************************************/
   // prevent the surface crack due to floating accuracy
    if (abs(isoValue - v1.scalar) < 0.00001)
        return(v1);
    if (abs(isoValue - v2.scalar) < 0.00001)
        return(v2);
    if (abs(v1.scalar - v2.scalar) < 0.00001)
        return(v1);
    /*************************************************/

    vertex intersectPoint;

    float factor = (isoValue - v1.scalar) / (v2.scalar - v1.scalar);

    // interpolate the positions between two cube vertices
    intersectPoint.x = v1.x + factor * (v2.x - v1.x);
    intersectPoint.y = v1.y + factor * (v2.y - v1.y);
    intersectPoint.z = v1.z + factor * (v2.z - v1.z);

    // the scalar of intersect point is useless, so just assign 0.0 to it
    intersectPoint.scalar = 0.0f;

    // interpolate the normals between two cube vertices
    float normal[3];
    float nx = v1.normal[0] + factor * (v2.normal[0] - v1.normal[0]);
    float ny = v1.normal[1] + factor * (v2.normal[1] - v1.normal[1]);
    float nz = v1.normal[2] + factor * (v2.normal[2] - v1.normal[2]);
    normal[0] = nx;
    normal[1] = ny;
    normal[2] = nz;

    // set the normal into intersect point
    setNormal(intersectPoint, normal);

    return intersectPoint;
}


// calculate the gradient as the normal vector for each grid coordinate, based on SDFValues
// input SDF values size should be the same as the grid size
void Octree::calculateNormalForGridVertex(const float* SDFValues, int x, int y, int z, int height, int width, int length, float * normal)
{
    // the normal of each grid vertex is the gradient 

    int v, vUp, vDown, vLeft, vRight, vFront, vBehind;
    v = y * width * length + x * length + z;

    if (y + 1 < height) vUp = (y + 1) * width * length + x * length + z;
    else vUp = v;
    if (y - 1 >= 0) vDown = (y - 1) * width * length + x * length + z;
    else vDown = v;

    if (z + 1 < length) vLeft = y * width * length + x * length + (z + 1);
    else vLeft = v;
    if (z - 1 >= 0) vRight = y * width * length + x * length + (z - 1);
    else vRight = v;

    if (x + 1 < width) vFront = y * width * length + (x + 1) * length + z;
    else vFront = v;
    if (x - 1 >= 0) vBehind = y * width * length + (x - 1) * length + z;
    else vBehind = v;

  //  float gradient[3] = {};
    float gx = 0.0, gy = 0.0, gz = 0.0;

    gx = (SDFValues[vFront] - SDFValues[vBehind]) / (80.0 / (width - 1));
    gy = (SDFValues[vUp] - SDFValues[vDown]) / (80.0 / (height - 1));

    // Because we use -z axis, the z coordinate of normal also needs to be multiply -1.0
    gz = -1.0 * (SDFValues[vLeft] - SDFValues[vRight]) / (80.0 / (length - 1));

    // normalize the gradient
    float vectorLength = sqrt(gx * gx + gy * gy + gz * gz);
    gx = gx / vectorLength;
    gy = gy / vectorLength;
    gz = gz / vectorLength;

    normal[0] = gx;
    normal[1] = gy;
    normal[2] = gz;

  // return gradient;
}

// eliminate the repeated vertices shared by two cubes on the same edge
// same as the post processing function in original marching cubes (mcFunctions.cpp)
void Octree::postProcessingForTriVertices(vertex * triVertexUniqueList)
{
    
    unordered_set <vertex> vertexSet;
    for (int i = 0; i < triVertexUniqueListSize[currentGeometry]; i++)
    {
        vertexSet.insert(triVertexUniqueList[i]);
    }
   
    int i = 0;
    for (const auto& v : vertexSet) {
        triVertexUniqueList[i++] = v;
    }
    
}


void Octree::generateIndices(const vertex * triangleVertexList, const vertex * triVertexUniqueList, uint32_t * indices)
{
    // create face list based on rendering order
    // Using unorder_map (hash map) is more efficient
    unordered_map<vertex, int> vertexMap;

    // put each vertex into hash map
    for (uint32_t i = 0; i < triVertexUniqueListSize[currentGeometry]; i++)
    {
        vertexMap[triVertexUniqueList[i]] = i;
    }
    // check the map to get index
    for (uint32_t i = 0; i < triangleVertexListSize[currentGeometry]; i++)
    {
        auto iterator = vertexMap.find(triangleVertexList[i]);
        uint32_t vertexIndex = iterator->second;
        indices[i] = vertexIndex;
    }

}

void Octree::setNormal(vertex& v, const float newNormal[3]) {

    v.normal[0] = newNormal[0];
    v.normal[1] = newNormal[1];
    v.normal[2] = newNormal[2];
}


