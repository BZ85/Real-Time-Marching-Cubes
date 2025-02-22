/*
 * Copyright (c) 2017-2024 The Forge Interactive Inc.
 *
 * This file is part of The-Forge
 * (see https://github.com/ConfettiFX/The-Forge).
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Unit Test for testing transformations using a solar system.
// Tests the basic mat4 transformations, such as scaling, rotation, and translation.


#define MAX_PLANETS 20 // Does not affect test, just for allocating space in uniform block. Must match with shader.


#include "octree.h" // implemented by myself

#define DB_PERLIN_IMPL
#include "db_perlin.hpp" // the library got from online source

// Interfaces
#include "../../../../Common_3/Application/Interfaces/IApp.h"
#include "../../../../Common_3/Application/Interfaces/ICameraController.h"
#include "../../../../Common_3/Application/Interfaces/IFont.h"
#include "../../../../Common_3/Application/Interfaces/IProfiler.h"
#include "../../../../Common_3/Application/Interfaces/IScreenshot.h"
#include "../../../../Common_3/Application/Interfaces/IUI.h"
#include "../../../../Common_3/Game/Interfaces/IScripting.h"
#include "../../../../Common_3/Utilities/Interfaces/IFileSystem.h"
#include "../../../../Common_3/Utilities/Interfaces/ILog.h"
#include "../../../../Common_3/Utilities/Interfaces/ITime.h"

#include "../../../../Common_3/Utilities/RingBuffer.h"

// Renderer
#include "../../../../Common_3/Graphics/Interfaces/IGraphics.h"
#include "../../../../Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h"

// Math
#include "../../../../Common_3/Utilities/Math/MathTypes.h"

#include "../../../../Common_3/Utilities/Interfaces/IMemory.h"

// fsl
#include "../../../../Common_3/Graphics/FSL/defaults.h"
#include "./Shaders/FSL/srt.h"

/// Demo structures
struct PlanetInfoStruct
{
   
    mat4  mTranslationMat;
    mat4  mScaleMat;
    mat4  mSharedMat; // Matrix to pass down to children
    vec4  mColor;
    uint  mParentIndex;
    float mYOrbitSpeed; // Rotation speed around parent
    float mZOrbitSpeed;
    float mRotationSpeed; // Rotation speed around self
    float mMorphingSpeed; // Speed of morphing betwee cube and sphere
};

struct UniformBlock
{
    CameraMatrix mProjectView; // View - Projection matrix
    CameraMatrix mSkyProjectView;
    mat4         mToWorldMat[MAX_PLANETS]; // maybe model matrix
    vec4         mColor[MAX_PLANETS];
    float        mGeometryWeight[MAX_PLANETS][4];

    // Point Light Information
    vec4 mLightPosition;
    vec4 mLightColor;
};

struct OctreeMarchingCubesUniformBlock
{
    CameraMatrix mProject;
    mat4         mView;
    mat4         modelMatrix;
    mat4         normalMatrix;
    vec4         color;

    float4 La;
    float4 Ld;
    float4 Ls;
    float4 ka;
    float4 kd;
    float4 ks;
    float4 lightDirection;
    float alpha;
    float3 padding;

};



// But we only need Two sets of resources (one in flight and one being used on CPU)
const uint32_t gDataBufferCount = 2;
const uint     gNumPlanets = 11;     // Sun, Mercury -> Neptune, Pluto, Moon
const uint     gTimeOffset = 600000; // For visually better starting locations
const float    gRotSelfScale = 0.0004f;
const float    gRotOrbitYScale = 0.001f;
const float    gRotOrbitZScale = 0.00001f;

Renderer*  pRenderer = NULL;
Queue*     pGraphicsQueue = NULL;
GpuCmdRing gGraphicsCmdRing = {};

SwapChain*    pSwapChain = NULL;
RenderTarget* pDepthBuffer = NULL;
Semaphore*    pImageAcquiredSemaphore = NULL;

Shader*      pSphereShader = NULL;
Buffer*      pSphereVertexBuffer = NULL;
Buffer*      pSphereIndexBuffer = NULL;
uint32_t     gSphereIndexCount = 0;
Pipeline*    pSpherePipeline = NULL;
VertexLayout gSphereVertexLayout = {};
uint32_t     gSphereLayoutType = 0;

Shader*        pSkyBoxDrawShader = NULL;
Buffer*        pSkyBoxVertexBuffer = NULL;
Pipeline*      pSkyBoxDrawPipeline = NULL;
Texture*       pSkyBoxTextures[6];
Sampler*       pSkyBoxSampler = {};
DescriptorSet* pDescriptorSetTexture = { NULL };


DescriptorSet* pDescriptorSetUniforms = { NULL };

// Setting for octree marching cubes surfaces
const uint32_t octreeMarchingCubesGeometryCount = 5; // the total amount for the prepared geometry  
uint32_t       currentGeometry = 0; // used for switching the shown geometries
Shader*        pOctreeMarchingCubesShader = NULL;
Buffer*        pOctreeMarchingCubesVertexBuffer[octreeMarchingCubesGeometryCount] = { NULL };
Buffer*        pOctreeMarchingCubesIndexBuffer[octreeMarchingCubesGeometryCount] = { NULL };
Pipeline*      pOctreeMarchingCubesPipeline = NULL;
VertexLayout   gOctreeMarchingCubesVertexLayout = {};
DescriptorSet* pOctreeMarchingCubesDescriptorSetUniforms = { NULL };
uint32_t       octreeMarchingCubesIndicesCount[octreeMarchingCubesGeometryCount] = {};
uint32_t       octreeMarchingCubesVertexCount[octreeMarchingCubesGeometryCount] = {};
Buffer*        pOctreeMarchingCubesUniformBuffer[gDataBufferCount] = { NULL };
float          angleMarchingCubes = 0.0f;

// Setting for octree wire frames (show how the space is divided by octree)
Shader*        pOctreeWireframesShader = NULL;
Buffer*        pOctreeWireframesVertexBuffer[octreeMarchingCubesGeometryCount] = { NULL };
Pipeline*      pOctreeWireframesPipeline = NULL;
uint32_t       octreeWireframesVertexCount[octreeMarchingCubesGeometryCount] = {};
bool           showWireframe = true;


Buffer*        pUniformBuffer[gDataBufferCount] = { NULL };

uint32_t       gFrameIndex = 0;
ProfileToken   gGpuProfileToken = PROFILE_INVALID_TOKEN;

int              gNumberOfSpherePoints;
UniformBlock     gUniformData;
PlanetInfoStruct gPlanetInfoData[gNumPlanets];

OctreeMarchingCubesUniformBlock omcUniformData;

ICameraController* pCameraController = NULL;

UIComponent* pGuiWindow = NULL;

UIComponent* pGuiControlBar = NULL;

uint32_t gFontID = 0;

QueryPool* pPipelineStatsQueryPool[gDataBufferCount] = {};




//const char* pSkyBoxImageFileNames[] = { "Skybox_right1.tex",  "Skybox_left2.tex",  "Skybox_top3.tex",
         //                               "Skybox_bottom4.tex", "Skybox_front5.tex", "Skybox_back6.tex" };

const char*  pSkyBoxImageFileNames[] = { "skybox/hw_sahara/sahara_rt.tex", "skybox/hw_sahara/sahara_lf.tex", "skybox/hw_sahara/sahara_up.tex",
                                        "skybox/hw_sahara/sahara_dn.tex", "skybox/hw_sahara/sahara_ft.tex", "skybox/hw_sahara/sahara_bk.tex" };

FontDrawDesc gFrameTimeDraw;

// Generate sky box vertex buffer
const float gSkyBoxPoints[] = {
    10.0f,  -10.0f, -10.0f, 6.0f, // -z
    -10.0f, -10.0f, -10.0f, 6.0f,   -10.0f, 10.0f,  -10.0f, 6.0f,   -10.0f, 10.0f,
    -10.0f, 6.0f,   10.0f,  10.0f,  -10.0f, 6.0f,   10.0f,  -10.0f, -10.0f, 6.0f,

    -10.0f, -10.0f, 10.0f,  2.0f, //-x
    -10.0f, -10.0f, -10.0f, 2.0f,   -10.0f, 10.0f,  -10.0f, 2.0f,   -10.0f, 10.0f,
    -10.0f, 2.0f,   -10.0f, 10.0f,  10.0f,  2.0f,   -10.0f, -10.0f, 10.0f,  2.0f,

    10.0f,  -10.0f, -10.0f, 1.0f, //+x
    10.0f,  -10.0f, 10.0f,  1.0f,   10.0f,  10.0f,  10.0f,  1.0f,   10.0f,  10.0f,
    10.0f,  1.0f,   10.0f,  10.0f,  -10.0f, 1.0f,   10.0f,  -10.0f, -10.0f, 1.0f,

    -10.0f, -10.0f, 10.0f,  5.0f, // +z
    -10.0f, 10.0f,  10.0f,  5.0f,   10.0f,  10.0f,  10.0f,  5.0f,   10.0f,  10.0f,
    10.0f,  5.0f,   10.0f,  -10.0f, 10.0f,  5.0f,   -10.0f, -10.0f, 10.0f,  5.0f,

    -10.0f, 10.0f,  -10.0f, 3.0f, //+y
    10.0f,  10.0f,  -10.0f, 3.0f,   10.0f,  10.0f,  10.0f,  3.0f,   10.0f,  10.0f,
    10.0f,  3.0f,   -10.0f, 10.0f,  10.0f,  3.0f,   -10.0f, 10.0f,  -10.0f, 3.0f,

    10.0f,  -10.0f, 10.0f,  4.0f, //-y
    10.0f,  -10.0f, -10.0f, 4.0f,   -10.0f, -10.0f, -10.0f, 4.0f,   -10.0f, -10.0f,
    -10.0f, 4.0f,   -10.0f, -10.0f, 10.0f,  4.0f,   10.0f,  -10.0f, 10.0f,  4.0f,
};

static unsigned char gPipelineStatsCharArray[2048] = {};
static bstring       gPipelineStats = bfromarr(gPipelineStatsCharArray);

void reloadRequest(void*)
{
    ReloadDesc reload{ RELOAD_TYPE_SHADER };
    requestReload(&reload);
}

const char* gWindowTestScripts[] = { "TestFullScreen.lua", "TestCenteredWindow.lua", "TestNonCenteredWindow.lua", "TestBorderless.lua" };

const char* gReloadServerTestScripts[] = { "TestReloadShader.lua", "TestReloadShaderCapture.lua" };

static void add_attribute(VertexLayout* layout, ShaderSemantic semantic, TinyImageFormat format, uint32_t offset)
{
    uint32_t n_attr = layout->mAttribCount++;

    VertexAttrib* attr = layout->mAttribs + n_attr;

    attr->mSemantic = semantic;
    attr->mFormat = format;
    attr->mBinding = 0;
    attr->mLocation = n_attr;
    attr->mOffset = offset;
}

static void copy_attribute(VertexLayout* layout, void* buffer_data, uint32_t offset, uint32_t size, uint32_t vcount, void* data)
{
    uint8_t* dst_data = static_cast<uint8_t*>(buffer_data);
    uint8_t* src_data = static_cast<uint8_t*>(data);
    for (uint32_t i = 0; i < vcount; ++i)
    {
        memcpy(dst_data + offset, src_data, size);

        dst_data += layout->mBindings[0].mStride;
        src_data += size;
    }
}

static void compute_normal(const float* src, float* dst)
{
    float len = sqrtf(src[0] * src[0] + src[1] * src[1] + src[2] * src[2]);
    if (len == 0)
    {
        dst[0] = 0;
        dst[1] = 0;
        dst[2] = 0;
    }
    else
    {
        dst[0] = src[0] / len;
        dst[1] = src[1] / len;
        dst[2] = src[2] / len;
    }
}

// mapping the cube planet mesh to the sphere planet mesh (don't need for my octree marching cubes program)
static void generate_complex_mesh()
{
    gSphereVertexLayout = {};

// number of vertices on a quad side, must be >= 2
#define DETAIL_LEVEL 64

    // static here to prevent stack overflow
    static float verts[6][DETAIL_LEVEL][DETAIL_LEVEL][3];
    static float sqNormals[6][DETAIL_LEVEL][DETAIL_LEVEL][3];
    static float sphNormals[6][DETAIL_LEVEL][DETAIL_LEVEL][3];

    for (int i = 0; i < 6; ++i)
    {
        for (int x = 0; x < DETAIL_LEVEL; ++x)
        {
            for (int y = 0; y < DETAIL_LEVEL; ++y)
            {
                float* vert = verts[i][x][y];
                float* sqNorm = sqNormals[i][x][y];

                sqNorm[0] = 0;
                sqNorm[1] = 0;
                sqNorm[2] = 0;

                float fx = 2 * (float(x) / float(DETAIL_LEVEL - 1)) - 1;
                float fy = 2 * (float(y) / float(DETAIL_LEVEL - 1)) - 1;

                switch (i)
                {
                case 0:
                    vert[0] = -1, vert[1] = fx, vert[2] = fy;
                    sqNorm[0] = -1;
                    break;
                case 1:
                    vert[0] = 1, vert[1] = -fx, vert[2] = fy;
                    sqNorm[0] = 1;
                    break;
                case 2:
                    vert[0] = -fx, vert[1] = fy, vert[2] = 1;
                    sqNorm[2] = 1;
                    break;
                case 3:
                    vert[0] = fx, vert[1] = fy, vert[2] = -1;
                    sqNorm[2] = -1;
                    break;
                case 4:
                    vert[0] = fx, vert[1] = 1, vert[2] = fy;
                    sqNorm[1] = 1;
                    break;
                case 5:
                    vert[0] = -fx, vert[1] = -1, vert[2] = fy;
                    sqNorm[1] = -1;
                    break;
                }

                compute_normal(vert, sphNormals[i][x][y]);
            }
        }
    }

    static uint8_t sqColors[6][DETAIL_LEVEL][DETAIL_LEVEL][3];
    static uint8_t spColors[6][DETAIL_LEVEL][DETAIL_LEVEL][3];
    for (int i = 0; i < 6; ++i)
    {
        for (int x = 0; x < DETAIL_LEVEL; ++x)
        {
            uint8_t spColorTemplate[3] = {
                uint8_t(randomInt(0, 256)),
                uint8_t(randomInt(0, 256)),
                uint8_t(randomInt(0, 256)),
            };

            float rx = 1 - abs((float(x) / DETAIL_LEVEL) * 2 - 1);

            for (int y = 0; y < DETAIL_LEVEL; ++y)
            {
                float    ry = 1 - abs((float(y) / DETAIL_LEVEL) * 2 - 1);
                uint32_t close_ratio = uint32_t(rx * ry * 255);

                uint8_t* sq_color = sqColors[i][x][y];
                uint8_t* sp_color = spColors[i][x][y];

                sq_color[0] = (uint8_t)((randomInt(0, 256) * close_ratio) / 255);
                sq_color[1] = (uint8_t)((randomInt(0, 256) * close_ratio) / 255);
                sq_color[2] = (uint8_t)((randomInt(0, 256) * close_ratio) / 255);

                sp_color[0] = (uint8_t)((spColorTemplate[0] * close_ratio) / 255);
                sp_color[1] = (uint8_t)((spColorTemplate[1] * close_ratio) / 255);
                sp_color[2] = (uint8_t)((spColorTemplate[2] * close_ratio) / 255);
            }
        }
    }

    static uint16_t indices[6][DETAIL_LEVEL - 1][DETAIL_LEVEL - 1][6];
    for (int i = 0; i < 6; ++i)
    {
        uint32_t o = DETAIL_LEVEL * DETAIL_LEVEL * i;
        for (int x = 0; x < DETAIL_LEVEL - 1; ++x)
        {
            for (int y = 0; y < DETAIL_LEVEL - 1; ++y)
            {
                uint16_t* quadIndices = indices[i][x][y];

#define vid(vx, vy) (o + (vx)*DETAIL_LEVEL + (vy))
                quadIndices[0] = (uint16_t)vid(x, y);
                quadIndices[1] = (uint16_t)vid(x, y + 1);
                quadIndices[2] = (uint16_t)vid(x + 1, y + 1);
                quadIndices[3] = (uint16_t)vid(x + 1, y + 1);
                quadIndices[4] = (uint16_t)vid(x + 1, y);
                quadIndices[5] = (uint16_t)vid(x, y);
#undef vid
            }
        }
    }

#undef DETAIL_LEVEL

    void*    bufferData = nullptr;
    uint32_t vertexCount = sizeof(verts) / 12;
    size_t   bufferSize;

    gSphereVertexLayout.mBindingCount = 1;

    switch (gSphereLayoutType)
    {
    default:
    case 0:
    {
        //  0-12 sq positions,
        // 12-16 sq colors
        // 16-28 sq normals
        // 28-32 sp colors
        // 32-44 sp positions + sp normals

        gSphereVertexLayout.mBindings[0].mStride = 44;
        size_t vsize = vertexCount * gSphereVertexLayout.mBindings[0].mStride;
        bufferSize = vsize;
        bufferData = tf_calloc(1, bufferSize);

        add_attribute(&gSphereVertexLayout, SEMANTIC_POSITION, TinyImageFormat_R32G32B32_SFLOAT, 0);
        add_attribute(&gSphereVertexLayout, SEMANTIC_NORMAL, TinyImageFormat_R32G32B32_SFLOAT, 16);
        add_attribute(&gSphereVertexLayout, SEMANTIC_TEXCOORD1, TinyImageFormat_R32G32B32_SFLOAT, 32);
        add_attribute(&gSphereVertexLayout, SEMANTIC_TEXCOORD3, TinyImageFormat_R32G32B32_SFLOAT, 32);
        add_attribute(&gSphereVertexLayout, SEMANTIC_TEXCOORD0, TinyImageFormat_R8G8B8A8_UNORM, 12);
        add_attribute(&gSphereVertexLayout, SEMANTIC_TEXCOORD2, TinyImageFormat_R8G8B8A8_UNORM, 28);

        copy_attribute(&gSphereVertexLayout, bufferData, 0, 12, vertexCount, verts);
        copy_attribute(&gSphereVertexLayout, bufferData, 12, 3, vertexCount, sqColors);
        copy_attribute(&gSphereVertexLayout, bufferData, 16, 12, vertexCount, sqNormals);
        copy_attribute(&gSphereVertexLayout, bufferData, 28, 3, vertexCount, spColors);
        copy_attribute(&gSphereVertexLayout, bufferData, 32, 12, vertexCount, sphNormals);
    }
    break;
    case 1:
    {
        //  0-12 sq positions,
        // 16-28 sq normals
        // 32-34 sq colors
        // 36-40 sp colors
        // 48-62 sp positions
        // 64-76 sp normals

        gSphereVertexLayout.mBindings[0].mStride = 80;
        size_t vsize = vertexCount * gSphereVertexLayout.mBindings[0].mStride;
        bufferSize = vsize;
        bufferData = tf_calloc(1, bufferSize);

        add_attribute(&gSphereVertexLayout, SEMANTIC_POSITION, TinyImageFormat_R32G32B32_SFLOAT, 0);
        add_attribute(&gSphereVertexLayout, SEMANTIC_NORMAL, TinyImageFormat_R32G32B32_SFLOAT, 16);
        add_attribute(&gSphereVertexLayout, SEMANTIC_TEXCOORD1, TinyImageFormat_R32G32B32_SFLOAT, 48);
        add_attribute(&gSphereVertexLayout, SEMANTIC_TEXCOORD3, TinyImageFormat_R32G32B32_SFLOAT, 64);
        add_attribute(&gSphereVertexLayout, SEMANTIC_TEXCOORD0, TinyImageFormat_R8G8B8A8_UNORM, 32);
        add_attribute(&gSphereVertexLayout, SEMANTIC_TEXCOORD2, TinyImageFormat_R8G8B8A8_UNORM, 36);

        copy_attribute(&gSphereVertexLayout, bufferData, 0, 12, vertexCount, verts);
        copy_attribute(&gSphereVertexLayout, bufferData, 16, 12, vertexCount, sqNormals);
        copy_attribute(&gSphereVertexLayout, bufferData, 36, 3, vertexCount, spColors);
        copy_attribute(&gSphereVertexLayout, bufferData, 32, 3, vertexCount, sqColors);
        copy_attribute(&gSphereVertexLayout, bufferData, 48, 12, vertexCount, sphNormals);
        copy_attribute(&gSphereVertexLayout, bufferData, 64, 12, vertexCount, sphNormals);
    }
    break;
    }

    gSphereIndexCount = sizeof(indices) / sizeof(uint16_t);

    BufferLoadDesc sphereVbDesc = {};
    sphereVbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
    sphereVbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
    sphereVbDesc.mDesc.mSize = bufferSize;
    sphereVbDesc.pData = bufferData;
    sphereVbDesc.ppBuffer = &pSphereVertexBuffer;
    addResource(&sphereVbDesc, nullptr);

    BufferLoadDesc sphereIbDesc = {};
    sphereIbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
    sphereIbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
    sphereIbDesc.mDesc.mSize = sizeof(indices);
    sphereIbDesc.pData = indices;
    sphereIbDesc.ppBuffer = &pSphereIndexBuffer;
    addResource(&sphereIbDesc, nullptr);

    waitForAllResourceLoads();

    tf_free(bufferData);
}

float computeTerrianHeight(float x, float z, float frequency, float amplitude)
{
    // Frequency and amplitude are tunable parameters
 //   float frequency = 0.05f;
  //  float amplitude = 40.0f;

    // PerlinNoise2D is some 2D noise function you implement or use from a library
    return amplitude * db::perlin(x * frequency, z * frequency);
}

float TerrainSDF(float x, float y, float z, float frequency, float amplitude) { return y - computeTerrianHeight(x, z, frequency, amplitude); }



static void generate_Octree_MarchingCubes_mesh()
{
    gOctreeMarchingCubesVertexLayout = {};
   // vector<vector<float>> SDFValues(octreeMarchingCubesGeometryCount);
 
  float** SDFValues = (float**)tf_calloc(octreeMarchingCubesGeometryCount, sizeof(float*));

    for (size_t i = 0; i < octreeMarchingCubesGeometryCount; ++i)
    {
        SDFValues[i] = (float*)tf_calloc(64 * 64 * 64, sizeof(float)); 
    }


    int height = 64;
    int width = 64;
    int length = 64;

    float isoValue = 0.0f;


    /************** generating new SDF ************************/

    // This part of code is used to generate  new SDF values, and then immediately produce marching cubes algorithm based on newly-created
    // SDF values

    // generate new SDF file and directly draw the surfaces
    for (int i = 0; i < height; i++) // y axis
        for (int k = 0; k < width; k++) // x axis
            for (int j = 0; j < length; j++)
            { // z axis, we use -z axis

                // here you can edit the signed distance funtion to create new SDF file
                // Torus
                float dk = static_cast<float> (k - 32);
                float di = static_cast<float> (i - 32);
                float dj = static_cast<float> (j - 32);
                float qk = sqrt(dk * dk + dj * dj) - 20;    // 28
                float scalarA = sqrt(qk * qk + di * di) - 8; // 3

               
                float scalarB = (i - 30) * (i - 30) + (j - 30) * (j - 30) + (k - 30) * (k - 30) - 500;
             
                float scalarC = i * i + j * j + k * k - 600;
                float disp = 4.0 * sin(20.0 * static_cast<float> (k)) * sin(20.0 * static_cast<float> (i)) * sin(20.0 * static_cast<float> (j));
                scalarA = scalarA + disp;
                scalarB = scalarB + 4.0 * disp;
                scalarC = scalarC + 4.0 * disp;

                float scalarD = TerrainSDF(k, i, j, 0.05f, 60.0f);
                float scalarE = TerrainSDF(k, i, j, 0.15f, 40.0f);

                SDFValues[0][i * width * length + k * length + j] = scalarA; // sdf for geometry A
                SDFValues[1][i * width * length + k * length + j] = scalarB; // sdf for geometry B
                SDFValues[2][i * width * length + k * length + j] = scalarC; // sdf for geometry C
                SDFValues[3][i * width * length + k * length + j] = scalarD; // sdf for geometry D (static terrain one)
                SDFValues[4][i * width * length + k * length + j] = scalarE; // sdf for geometry E (static terrain two)
            
            }

    /*****************************************************/

    // each list contains the sublists of different kinds of prepared geometry
   
    // define the lists and allocate memory
    vertex** octreeTriangleVertexLists = (vertex**)tf_calloc(octreeMarchingCubesGeometryCount, sizeof(vertex*));
    vertex** octreeTriVertexUniqueLists = (vertex**)tf_calloc(octreeMarchingCubesGeometryCount, sizeof(vertex*));
    uint32_t** octreeTriVertexIndicesLists = (uint32_t**)tf_calloc(octreeMarchingCubesGeometryCount, sizeof(uint32_t*));
    uint32_t** lineLists = (uint32_t**)tf_calloc(octreeMarchingCubesGeometryCount, sizeof(uint32_t*));

    size_t maximumSizeVertex = 64 * 64 * 25; 
    size_t maximumSizeIndices = 64 * 64 * 40; 
    for (size_t i = 0; i < octreeMarchingCubesGeometryCount; ++i)
    {
        octreeTriangleVertexLists[i] = (vertex*)tf_calloc(maximumSizeIndices, sizeof(vertex));
        octreeTriVertexUniqueLists[i] = (vertex*)tf_calloc(maximumSizeVertex, sizeof(vertex));
        octreeTriVertexIndicesLists[i] = (uint32_t*)tf_calloc(maximumSizeIndices, sizeof(uint32_t));
        lineLists[i] = (uint32_t*)tf_calloc(64 * 64 * 64 * 2, sizeof(uint32_t));
    }

    unordered_map<int, vertex> octreeGridCoordinatesMaps[octreeMarchingCubesGeometryCount];

    // compute the maximum depth according to grid size
    int maxDepth = ceil(log(length * width * height) / log(8));

    // run the octree marching cubes algorithm
    // first build the octree and get the root node, then traverse the octree

    for (unsigned int i = 0; i < octreeMarchingCubesGeometryCount; i++) // there are three prepared marching cubes geometries
    {
        Octree::currentGeometry = i;

        Octree::octreeMarchingCubes(octreeGridCoordinatesMaps[i], octreeTriangleVertexLists[i], octreeTriVertexUniqueLists[i], lineLists[i], SDFValues[i],
                                    maxDepth, height, length, width);

        Octree::generateIndices(octreeTriangleVertexLists[i], octreeTriVertexUniqueLists[i], octreeTriVertexIndicesLists[i]);
    }

    float* octreeTriangleVerticesPositions[octreeMarchingCubesGeometryCount];
    float* octreeTriangleVerticesNormals[octreeMarchingCubesGeometryCount];
    
     for (size_t i = 0; i < octreeMarchingCubesGeometryCount; i++)
    {
        octreeTriangleVerticesPositions[i] = (float*)tf_calloc(Octree::triVertexUniqueListSize[i] * 3, sizeof(float));
        octreeTriangleVerticesNormals[i] = (float*)tf_calloc(Octree::triVertexUniqueListSize[i] * 3, sizeof(float));
    }

    for (int j = 0; j < octreeMarchingCubesGeometryCount; j++)
    {
        int index = 0;
        int indexNormal = 0;
        for (unsigned int i = 0; i < Octree::triVertexUniqueListSize[j]; i++)
        {
            octreeTriangleVerticesPositions[j][index++] = (octreeTriVertexUniqueLists[j][i].x);
            octreeTriangleVerticesPositions[j][index++] = (octreeTriVertexUniqueLists[j][i].y);
            octreeTriangleVerticesPositions[j][index++] = (octreeTriVertexUniqueLists[j][i].z);

            octreeTriangleVerticesNormals[j][indexNormal++] = (octreeTriVertexUniqueLists[j][i].normal[0]);
            octreeTriangleVerticesNormals[j][indexNormal++] = (octreeTriVertexUniqueLists[j][i].normal[1]);
            octreeTriangleVerticesNormals[j][indexNormal++] = (octreeTriVertexUniqueLists[j][i].normal[2]);
        }
    }

     // render the wireframes of octree marching cubes
    float* octreeWireframesPositions[octreeMarchingCubesGeometryCount];
    float* octreeWireframesColors[octreeMarchingCubesGeometryCount];

     for (size_t i = 0; i < octreeMarchingCubesGeometryCount; i++)
    {
        octreeWireframesPositions[i] = (float*)tf_calloc(Octree::lineListSize[i] * 9, sizeof(float));
        octreeWireframesColors[i] = (float*)tf_calloc(Octree::lineListSize[i] * 9, sizeof(float));
    }
    
    // set the wire frame vertices position
    for (unsigned int j = 0; j < octreeMarchingCubesGeometryCount; j++)
    {
        int index = 0;
        int indexColor = 0;

        for (int i = 0; i < Octree::lineListSize[j]; i = i + 8)
        { 
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 0]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 4]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 1]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 5]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 2]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 6]].z;

            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 3]].z;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].x;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].y;
            octreeWireframesPositions[j][index++] = octreeGridCoordinatesMaps[j][lineLists[j][i + 7]].z;
            
            for (int k = 0; k < 12; k++)
            {
                // set the color of the wireframe
                octreeWireframesColors[j][indexColor++] = 0.15f;
                octreeWireframesColors[j][indexColor++] = 0.15f;
                octreeWireframesColors[j][indexColor++] = 0.15f;

                octreeWireframesColors[j][indexColor++] = 0.15f;
                octreeWireframesColors[j][indexColor++] = 0.15f;
                octreeWireframesColors[j][indexColor++] = 0.15f;
            }
        }
        
        octreeWireframesVertexCount[j] = index / 3;
    }
    
  gOctreeMarchingCubesVertexLayout.mBindingCount = 1;

    //  0-12  positions,
    // 12-24  normals

    gOctreeMarchingCubesVertexLayout.mBindings[0].mStride = 24;
    add_attribute(&gOctreeMarchingCubesVertexLayout, SEMANTIC_POSITION, TinyImageFormat_R32G32B32_SFLOAT, 0);
    add_attribute(&gOctreeMarchingCubesVertexLayout, SEMANTIC_NORMAL, TinyImageFormat_R32G32B32_SFLOAT, 12);
    

    for (int i = 0; i < octreeMarchingCubesGeometryCount; i++)
    {
        void* bufferData = nullptr;
        // uint32_t vertexCount = octreeTriangleVertexList.size();
        octreeMarchingCubesIndicesCount[i] = Octree::triangleVertexListSize[i];
        uint32_t vertexCount = Octree::triVertexUniqueListSize[i];
        //   octreeMarchingCubesVertexCount = octreeTriangleVertexList.size();

        size_t bufferSize;

        //  gOctreeMarchingCubesVertexLayout.mBindingCount = 1;

        //  0-12  positions,
        // 12-24  normals

        //   gOctreeMarchingCubesVertexLayout.mBindings[0].mStride = 24;
        size_t vsize = vertexCount * gOctreeMarchingCubesVertexLayout.mBindings[0].mStride;
        bufferSize = vsize;
        bufferData = tf_calloc(1, bufferSize);

        // add_attribute(&gOctreeMarchingCubesVertexLayout, SEMANTIC_POSITION, TinyImageFormat_R32G32B32_SFLOAT, 0);
        //  add_attribute(&gOctreeMarchingCubesVertexLayout, SEMANTIC_NORMAL, TinyImageFormat_R32G32B32_SFLOAT, 12);

        copy_attribute(&gOctreeMarchingCubesVertexLayout, bufferData, 0, 12, vertexCount, octreeTriangleVerticesPositions[i]);
        copy_attribute(&gOctreeMarchingCubesVertexLayout, bufferData, 12, 12, vertexCount, octreeTriangleVerticesNormals[i]);

        BufferLoadDesc OctreeMarchingCubesVbDesc = {};
        OctreeMarchingCubesVbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        OctreeMarchingCubesVbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        OctreeMarchingCubesVbDesc.mDesc.mSize = bufferSize;
        OctreeMarchingCubesVbDesc.pData = bufferData;
        OctreeMarchingCubesVbDesc.ppBuffer = &pOctreeMarchingCubesVertexBuffer[i];
        addResource(&OctreeMarchingCubesVbDesc, nullptr);

        BufferLoadDesc OctreeMarchingCubesIbDesc = {};
        OctreeMarchingCubesIbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
        OctreeMarchingCubesIbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        OctreeMarchingCubesIbDesc.mDesc.mSize = sizeof(uint32_t) * octreeMarchingCubesIndicesCount[i];
        OctreeMarchingCubesIbDesc.pData = octreeTriVertexIndicesLists[i];
        OctreeMarchingCubesIbDesc.ppBuffer = &pOctreeMarchingCubesIndexBuffer[i];
        addResource(&OctreeMarchingCubesIbDesc, nullptr);

        waitForAllResourceLoads();
        tf_free(bufferData);

        // upload vertex buffer of wireframes for each geometry

      //  octreeWireframesVertexCount[i] = Octree::lineListSize[i];

        bufferSize = octreeWireframesVertexCount[i] * gOctreeMarchingCubesVertexLayout.mBindings[0].mStride;
        bufferData = tf_calloc(1, bufferSize);
        

        // wireframe uses the same vertex layout as the marching cubes surfaces
        copy_attribute(&gOctreeMarchingCubesVertexLayout, bufferData, 0, 12, octreeWireframesVertexCount[i], octreeWireframesPositions[i]);
        copy_attribute(&gOctreeMarchingCubesVertexLayout, bufferData, 12, 12, octreeWireframesVertexCount[i], octreeWireframesColors[i]);

        BufferLoadDesc OctreeWireframesVbDesc = {};
        OctreeWireframesVbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        OctreeWireframesVbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        OctreeWireframesVbDesc.mDesc.mSize = bufferSize;
        OctreeWireframesVbDesc.pData = bufferData;
        OctreeWireframesVbDesc.ppBuffer = &pOctreeWireframesVertexBuffer[i];
        addResource(&OctreeWireframesVbDesc, nullptr);

        waitForAllResourceLoads();

        tf_free(bufferData);
    }


    // free the memory
    for (size_t i = 0; i < octreeMarchingCubesGeometryCount; ++i)
    {
        
            tf_free(octreeTriangleVertexLists[i]);
            tf_free(octreeTriVertexUniqueLists[i]);
            tf_free(octreeTriVertexIndicesLists[i]);
            tf_free(lineLists[i]);
            tf_free(SDFValues[i]);
            tf_free(octreeWireframesColors[i]);
            tf_free(octreeWireframesPositions[i]);
            tf_free(octreeTriangleVerticesPositions[i]);
            tf_free(octreeTriangleVerticesNormals[i]);
    }
    tf_free(octreeTriangleVertexLists);
    tf_free(octreeTriVertexUniqueLists);
    tf_free(octreeTriVertexIndicesLists);
    tf_free(lineLists);
    tf_free(SDFValues);
}


class Transformations: public IApp
{
public:
    bool Init()
    {
        // window and renderer setup
        RendererDesc settings;
        memset(&settings, 0, sizeof(settings));
        initGPUConfiguration(settings.pExtendedSettings);
        initRenderer(GetName(), &settings, &pRenderer);
        // check for init success
        if (!pRenderer)
        {
            ShowUnsupportedMessage("Failed To Initialize renderer!");
            return false;
        }
        setupGPUConfigurationPlatformParameters(pRenderer, settings.pExtendedSettings);

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryPoolDesc poolDesc = {};
            poolDesc.mQueryCount = 3; // The count is 3 due to quest & multi-view use otherwise 2 is enough as we use 2 queries.
            poolDesc.mType = QUERY_TYPE_PIPELINE_STATISTICS;
            for (uint32_t i = 0; i < gDataBufferCount; ++i)
            {
                initQueryPool(pRenderer, &poolDesc, &pPipelineStatsQueryPool[i]);
            }
        }

        QueueDesc queueDesc = {};
        queueDesc.mType = QUEUE_TYPE_GRAPHICS;
        queueDesc.mFlag = QUEUE_FLAG_INIT_MICROPROFILE;
        initQueue(pRenderer, &queueDesc, &pGraphicsQueue);

        GpuCmdRingDesc cmdRingDesc = {};
        cmdRingDesc.pQueue = pGraphicsQueue;
        cmdRingDesc.mPoolCount = gDataBufferCount;
        cmdRingDesc.mCmdPerPoolCount = 1;
        cmdRingDesc.mAddSyncPrimitives = true;
        initGpuCmdRing(pRenderer, &cmdRingDesc, &gGraphicsCmdRing);

        initSemaphore(pRenderer, &pImageAcquiredSemaphore);

        initResourceLoaderInterface(pRenderer);

        RootSignatureDesc rootDesc = {};
        INIT_RS_DESC(rootDesc, "default.rootsig", "compute.rootsig");
        initRootSignature(pRenderer, &rootDesc);

        SamplerDesc samplerDesc = { FILTER_LINEAR,
                                    FILTER_LINEAR,
                                    MIPMAP_MODE_LINEAR,
                                    ADDRESS_MODE_CLAMP_TO_EDGE,
                                    ADDRESS_MODE_CLAMP_TO_EDGE,
                                    ADDRESS_MODE_CLAMP_TO_EDGE };
        addSampler(pRenderer, &samplerDesc, &pSkyBoxSampler);

        // Loads Skybox Textures
        for (int i = 0; i < 6; ++i)
        {
            TextureLoadDesc textureDesc = {};
            textureDesc.pFileName = pSkyBoxImageFileNames[i];
            textureDesc.ppTexture = &pSkyBoxTextures[i];
            // Textures representing color should be stored in SRGB or HDR format
            textureDesc.mCreationFlag = TEXTURE_CREATION_FLAG_SRGB;
            addResource(&textureDesc, NULL);
        }

        uint64_t       skyBoxDataSize = 4 * 6 * 6 * sizeof(float);
        BufferLoadDesc skyboxVbDesc = {};
        skyboxVbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        skyboxVbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        skyboxVbDesc.mDesc.mSize = skyBoxDataSize;
        skyboxVbDesc.pData = gSkyBoxPoints;
        skyboxVbDesc.ppBuffer = &pSkyBoxVertexBuffer;
        addResource(&skyboxVbDesc, NULL);

        BufferLoadDesc ubDesc = {};
        ubDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        ubDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        ubDesc.pData = NULL;
        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            ubDesc.mDesc.pName = "UniformBuffer";
            ubDesc.mDesc.mSize = sizeof(UniformBlock);
            ubDesc.ppBuffer = &pUniformBuffer[i];
            addResource(&ubDesc, NULL);
        }

        // add octree marching cubes uniform buffer
        BufferLoadDesc ubDescOMC = {};
        ubDescOMC.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubDescOMC.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        ubDescOMC.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        ubDescOMC.pData = NULL;
        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            ubDescOMC.mDesc.pName = "OctreeMarchingCubesUniformBuffer";
            ubDescOMC.mDesc.mSize = sizeof(OctreeMarchingCubesUniformBlock);
            ubDescOMC.ppBuffer = &pOctreeMarchingCubesUniformBuffer[i];
            addResource(&ubDescOMC, NULL);
        }


        // Load fonts
        FontDesc font = {};
        font.pFontPath = "TitilliumText/TitilliumText-Bold.otf";
        fntDefineFonts(&font, 1, &gFontID);

        FontSystemDesc fontRenderDesc = {};
        fontRenderDesc.pRenderer = pRenderer;
        if (!initFontSystem(&fontRenderDesc))
            return false; // report?

        // Initialize Forge User Interface Rendering
        UserInterfaceDesc uiRenderDesc = {};
        uiRenderDesc.pRenderer = pRenderer;
        initUserInterface(&uiRenderDesc);

        // Initialize micro profiler and its UI.
        ProfilerDesc profiler = {};
        profiler.pRenderer = pRenderer;
        initProfiler(&profiler);

        // Gpu profiler can only be added after initProfile.
        gGpuProfileToken = initGpuProfiler(pRenderer, pGraphicsQueue, "Graphics");

        const uint32_t numScripts = TF_ARRAY_COUNT(gWindowTestScripts);
        LuaScriptDesc  scriptDescs[numScripts] = {};
        uint32_t       numScriptsFinal = numScripts;
        // For reload server test, use reload server test scripts
        if (!mSettings.mBenchmarking)
            numScriptsFinal = TF_ARRAY_COUNT(gReloadServerTestScripts);
        for (uint32_t i = 0; i < numScriptsFinal; ++i)
            scriptDescs[i].pScriptFileName = mSettings.mBenchmarking ? gWindowTestScripts[i] : gReloadServerTestScripts[i];
        DEFINE_LUA_SCRIPTS(scriptDescs, numScriptsFinal);

        waitForAllResourceLoads();
       
        //CameraMotionParameters cmp{ 160.0f, 600.0f, 200.0f };
        CameraMotionParameters cmp{ 80.0f, 300.0f, 100.0f };
        vec3                   camPos{ 60.0f, 45.0f, -60.0f };
        vec3                   lookAt{ vec3(0) };

        pCameraController = initFpsCameraController(camPos, lookAt);

        pCameraController->setMotionParameters(cmp);

        omcUniformData.La = float4(1.0f, 1.0f, 1.0f, 1.0f);
        omcUniformData.Ld = float4(1.2f, 1.2f, 1.2f, 1.0f);
        omcUniformData.Ls = float4(1.0f, 1.0f, 1.0f, 1.0f);
        omcUniformData.ka = float4(0.1f, 0.1f, 0.1f, 1.0f);
        omcUniformData.kd = float4(1.0f, 1.0f, 1.0f, 1.0f);
        omcUniformData.ks = float4(0.2f, 0.2f, 0.2f, 1.0f);
        omcUniformData.lightDirection = float4(0.0f, 1.0f, 0.0f, 0.0f);
        omcUniformData.alpha = 1.0f;
        omcUniformData.padding = float3(0.f, 0.f, 0.f);

        // Add mouse and keyboard input
        AddCustomInputBindings();

        initScreenshotCapturer(pRenderer, pGraphicsQueue, GetName());
        gFrameIndex = 0;

        return true;
    }

    void Exit()
    {
        exitScreenshotCapturer();

        exitCameraController(pCameraController);

        exitUserInterface();

        exitFontSystem();

        // Exit profile
        exitProfiler();

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            removeResource(pUniformBuffer[i]);
            removeResource(pOctreeMarchingCubesUniformBuffer[i]);
            if (pRenderer->pGpu->mPipelineStatsQueries)
            {
                exitQueryPool(pRenderer, pPipelineStatsQueryPool[i]);
            }
        }

        removeResource(pSkyBoxVertexBuffer);
        removeSampler(pRenderer, pSkyBoxSampler);

        for (uint i = 0; i < 6; ++i)
            removeResource(pSkyBoxTextures[i]);

        exitGpuCmdRing(pRenderer, &gGraphicsCmdRing);
        exitSemaphore(pRenderer, pImageAcquiredSemaphore);

        exitRootSignature(pRenderer);
        exitResourceLoaderInterface(pRenderer);

        exitQueue(pRenderer, pGraphicsQueue);

        exitRenderer(pRenderer);
        exitGPUConfiguration();
        pRenderer = NULL;
    }

    bool Load(ReloadDesc* pReloadDesc)
    {
        if (pReloadDesc->mType & RELOAD_TYPE_SHADER)
        {
            addShaders();
            addDescriptorSets();
        }

        if (pReloadDesc->mType & (RELOAD_TYPE_RESIZE | RELOAD_TYPE_RENDERTARGET))
        {
            // we only need to reload gui when the size of window changed
            
            loadProfilerUI(mSettings.mWidth, mSettings.mHeight);

            UIComponentDesc guiDesc = {};
            guiDesc.mStartPosition = vec2(mSettings.mWidth * 0.01f, mSettings.mHeight * 0.2f);
            uiAddComponent(GetName(), &guiDesc, &pGuiWindow);


             // set GUI for controlling wireframes and geometryies
            UIComponentDesc guiDescControl = {};
            guiDescControl.mStartPosition = vec2(mSettings.mWidth * 0.01f, mSettings.mHeight * 0.7f);
            uiAddComponent("Wireframes and Geometry Control", &guiDescControl, &pGuiControlBar);

            CheckboxWidget wireframeEnabled = {};
            wireframeEnabled.pData = (bool*)&showWireframe;
            luaRegisterWidget(uiAddComponentWidget(pGuiControlBar, "show wireframes", &wireframeEnabled, WIDGET_TYPE_CHECKBOX));

            SliderUintWidget switchGeometryWidget;
            switchGeometryWidget.mMin = 0;
            switchGeometryWidget.mMax = 4;
            switchGeometryWidget.mStep = 1;
            switchGeometryWidget.pData = &currentGeometry;
            luaRegisterWidget(uiAddComponentWidget(pGuiControlBar, "Current Geometry", &switchGeometryWidget, WIDGET_TYPE_SLIDER_UINT));

            
            SliderUintWidget vertexLayoutWidget;
            vertexLayoutWidget.mMin = 0;
            vertexLayoutWidget.mMax = 1;
            vertexLayoutWidget.mStep = 1;
            vertexLayoutWidget.pData = &gSphereLayoutType;
            UIWidget* pVLw = uiAddComponentWidget(pGuiWindow, "Vertex Layout", &vertexLayoutWidget, WIDGET_TYPE_SLIDER_UINT);
            uiSetWidgetOnEditedCallback(pVLw, nullptr, reloadRequest);
            

            if (pRenderer->pGpu->mPipelineStatsQueries)
            {
                static float4     color = { 1.0f, 1.0f, 1.0f, 1.0f };
                DynamicTextWidget statsWidget;
                statsWidget.pText = &gPipelineStats;
                statsWidget.pColor = &color;
                uiAddComponentWidget(pGuiWindow, "Pipeline Stats", &statsWidget, WIDGET_TYPE_DYNAMIC_TEXT);
            }
            
            if (!addSwapChain())
                return false;

            if (!addDepthBuffer())
                return false;
        }

        if (pReloadDesc->mType & (RELOAD_TYPE_SHADER | RELOAD_TYPE_RENDERTARGET))
        {
            generate_complex_mesh();
            generate_Octree_MarchingCubes_mesh();
            addPipelines();
        }

        prepareDescriptorSets();
        prepareOctreeMarchingCubesDescriptorSets();
        
        UserInterfaceLoadDesc uiLoad = {};
        uiLoad.mColorFormat = pSwapChain->ppRenderTargets[0]->mFormat;
        uiLoad.mHeight = mSettings.mHeight;
        uiLoad.mWidth = mSettings.mWidth;
        uiLoad.mLoadType = pReloadDesc->mType;
        loadUserInterface(&uiLoad);

        FontSystemLoadDesc fontLoad = {};
        fontLoad.mColorFormat = pSwapChain->ppRenderTargets[0]->mFormat;
        fontLoad.mHeight = mSettings.mHeight;
        fontLoad.mWidth = mSettings.mWidth;
        fontLoad.mLoadType = pReloadDesc->mType;
        loadFontSystem(&fontLoad);
        
        return true;
    }

    void Unload(ReloadDesc* pReloadDesc)
    {
        waitQueueIdle(pGraphicsQueue);

        unloadFontSystem(pReloadDesc->mType);
        unloadUserInterface(pReloadDesc->mType);

        if (pReloadDesc->mType & (RELOAD_TYPE_SHADER | RELOAD_TYPE_RENDERTARGET))
        {
            removePipelines();
            removeResource(pSphereVertexBuffer);
            removeResource(pSphereIndexBuffer);

            for (int i = 0; i < octreeMarchingCubesGeometryCount; i++)
            {
                removeResource(pOctreeMarchingCubesVertexBuffer[i]);
                removeResource(pOctreeMarchingCubesIndexBuffer[i]);
                removeResource(pOctreeWireframesVertexBuffer[i]);
            }
          
        //    removeResource(pOctreeWireframesIndexBuffer);
        }

        if (pReloadDesc->mType & (RELOAD_TYPE_RESIZE | RELOAD_TYPE_RENDERTARGET))
        {
            removeSwapChain(pRenderer, pSwapChain);
            removeRenderTarget(pRenderer, pDepthBuffer);
            uiRemoveComponent(pGuiWindow);
            uiRemoveComponent(pGuiControlBar);
            unloadProfilerUI();
        }

        if (pReloadDesc->mType & RELOAD_TYPE_SHADER)
        {
            removeDescriptorSets();
            removeShaders();
        }
    }

    void Update(float deltaTime)
    {
        if (!uiIsFocused())
        {
            pCameraController->onMove({ inputGetValue(0, CUSTOM_MOVE_X), inputGetValue(0, CUSTOM_MOVE_Y) });
            pCameraController->onRotate({ inputGetValue(0, CUSTOM_LOOK_X), inputGetValue(0, CUSTOM_LOOK_Y) });
            pCameraController->onMoveY(inputGetValue(0, CUSTOM_MOVE_UP));
            if (inputGetValue(0, CUSTOM_RESET_VIEW))
            {
                pCameraController->resetView(); // don't need the reset view at the moment
                
               // turn on the wireframe or turn off
             //  showWireframe = !showWireframe; 
            }
            if (inputGetValue(0, CUSTOM_TOGGLE_FULLSCREEN))
            {
                toggleFullscreen(pWindow);
            }
            if (inputGetValue(0, CUSTOM_TOGGLE_UI))
            {
                uiToggleActive();
               // switch the geometry being shown
               // currentGeometry = (currentGeometry + 1) % octreeMarchingCubesGeometryCount;

            }
            if (inputGetValue(0, CUSTOM_DUMP_PROFILE))
            {
                dumpProfileData(GetName());
            }
            if (inputGetValue(0, CUSTOM_EXIT))
            {
                requestShutdown();
            }
        }

        pCameraController->update(deltaTime);
        /************************************************************************/
        // Scene Update
        /************************************************************************/
        static float currentTime = 0.0f;
        currentTime += deltaTime * 1000.0f;

        // update camera with time
        mat4 viewMat = pCameraController->getViewMatrix();

        const float  aspectInverse = (float)mSettings.mHeight / (float)mSettings.mWidth;
        const float  horizontal_fov = PI / 2.0f;
        CameraMatrix projMat = CameraMatrix::perspectiveReverseZ(horizontal_fov, aspectInverse, 0.1f, 1000.0f);
        gUniformData.mProjectView = projMat * viewMat;

        // project and view martix for octree marching cubes pipeline
        omcUniformData.mView = viewMat;
        omcUniformData.mProject = projMat;

        // set the model matrix for rotating the octree marcing cubes model
         angleMarchingCubes += 1.0f * deltaTime;
        if (angleMarchingCubes > 360.0f)
        {
            angleMarchingCubes -= 360.0f;
        }
        mat4 translationToOrigin = mat4::translation(vec3(-20.0f, -20.0f, 20.0f));
        mat4 translationBack = mat4::translation(vec3(20.0f, 20.0f, -20.0f));
        mat4 rotationModel = mat4::rotationY(angleMarchingCubes);

        omcUniformData.modelMatrix = translationBack * rotationModel * translationToOrigin;
       // omcUniformData.modelMatrix = mat4::identity();
        omcUniformData.color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
      //  omcUniformData.normalMatrix = transpose(inverse(omcUniformData.mView * omcUniformData.modelMatrix));
        omcUniformData.normalMatrix = omcUniformData.mView * omcUniformData.modelMatrix; // no non-uniform scaling, so don't need inverse and transpose
        // point light parameters
        gUniformData.mLightPosition = vec4(0, 0, 0, 0);
        gUniformData.mLightColor = vec4(0.9f, 0.9f, 0.7f, 1.0f); // Pale Yellow

        viewMat.setTranslation(vec3(0));
        gUniformData.mSkyProjectView = projMat * viewMat;
    }

    void Draw()
    {
        if ((bool)pSwapChain->mEnableVsync != mSettings.mVSyncEnabled)
        {
            waitQueueIdle(pGraphicsQueue);
            ::toggleVSync(pRenderer, &pSwapChain);
        }

        uint32_t swapchainImageIndex;
        acquireNextImage(pRenderer, pSwapChain, pImageAcquiredSemaphore, NULL, &swapchainImageIndex);

        RenderTarget*     pRenderTarget = pSwapChain->ppRenderTargets[swapchainImageIndex];
        GpuCmdRingElement elem = getNextGpuCmdRingElement(&gGraphicsCmdRing, true, 1);

        // Stall if CPU is running "gDataBufferCount" frames ahead of GPU
        FenceStatus fenceStatus;
        getFenceStatus(pRenderer, elem.pFence, &fenceStatus);
        if (fenceStatus == FENCE_STATUS_INCOMPLETE)
            waitForFences(pRenderer, 1, &elem.pFence);

        
        // Update UNIFORM BUFFER HERE, need to take attention
        // Update uniform buffers
        BufferUpdateDesc viewProjCbv = { pUniformBuffer[gFrameIndex] };
        beginUpdateResource(&viewProjCbv);
        memcpy(viewProjCbv.pMappedData, &gUniformData, sizeof(gUniformData));
        endUpdateResource(&viewProjCbv);

        // Update octree marching cubes uniform buffer
        BufferUpdateDesc viewProjCbv2 = { pOctreeMarchingCubesUniformBuffer[gFrameIndex] };
        beginUpdateResource(&viewProjCbv2);
        memcpy(viewProjCbv2.pMappedData, &omcUniformData, sizeof(omcUniformData));
        endUpdateResource(&viewProjCbv2);

        // Reset cmd pool for this frame
        resetCmdPool(pRenderer, elem.pCmdPool);

        // Collects GPU performance data
        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryData data3D = {};
            QueryData data2D = {};
            getQueryData(pRenderer, pPipelineStatsQueryPool[gFrameIndex], 0, &data3D);
            getQueryData(pRenderer, pPipelineStatsQueryPool[gFrameIndex], 1, &data2D);
            bformat(&gPipelineStats,
                    "\n"
                    "Pipeline Stats 3D:\n"
                    "    VS invocations:      %u\n"
                    "    PS invocations:      %u\n"
                    "    Clipper invocations: %u\n"
                    "    IA primitives:       %u\n"
                    "    Clipper primitives:  %u\n"
                    "\n"
                    "Pipeline Stats 2D UI:\n"
                    "    VS invocations:      %u\n"
                    "    PS invocations:      %u\n"
                    "    Clipper invocations: %u\n"
                    "    IA primitives:       %u\n"
                    "    Clipper primitives:  %u\n",
                    data3D.mPipelineStats.mVSInvocations, data3D.mPipelineStats.mPSInvocations, data3D.mPipelineStats.mCInvocations,
                    data3D.mPipelineStats.mIAPrimitives, data3D.mPipelineStats.mCPrimitives, data2D.mPipelineStats.mVSInvocations,
                    data2D.mPipelineStats.mPSInvocations, data2D.mPipelineStats.mCInvocations, data2D.mPipelineStats.mIAPrimitives,
                    data2D.mPipelineStats.mCPrimitives);
        }

       /******************************************************************************************************************************************/

        // start recording commands
        Cmd* cmd = elem.pCmds[0];
        beginCmd(cmd);
      
        // Tracks performance of each step
        cmdBeginGpuFrameProfile(cmd, gGpuProfileToken);
        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            cmdResetQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], 0, 2);
            QueryDesc queryDesc = { 0 };
            cmdBeginQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
        }

        // Synchronize GPU Resources
        // transit the render target's state from present to rendering 
        RenderTargetBarrier barriers[] = {
            { pRenderTarget, RESOURCE_STATE_PRESENT, RESOURCE_STATE_RENDER_TARGET },
        };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Skybox/Planets");

        // simply record the screen cleaning command
        BindRenderTargetsDesc bindRenderTargets = {};
        bindRenderTargets.mRenderTargetCount = 1;
        bindRenderTargets.mRenderTargets[0] = { pRenderTarget, LOAD_ACTION_CLEAR };
        bindRenderTargets.mDepthStencil = { pDepthBuffer, LOAD_ACTION_CLEAR };
        cmdBindRenderTargets(cmd, &bindRenderTargets);
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 0.0f, 1.0f);
        cmdSetScissor(cmd, 0, 0, pRenderTarget->mWidth, pRenderTarget->mHeight);

        const uint32_t skyboxVbStride = sizeof(float) * 4;
        
        // draw skybox
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Skybox");
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 1.0f, 1.0f);
        cmdBindPipeline(cmd, pSkyBoxDrawPipeline);
        cmdBindDescriptorSet(cmd, 0, pDescriptorSetTexture);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetUniforms);
        cmdBindVertexBuffer(cmd, 1, &pSkyBoxVertexBuffer, &skyboxVbStride, NULL);
        cmdDraw(cmd, 36, 0);
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 0.0f, 1.0f);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        /*
        // draw planet
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Planets");
        cmdBindPipeline(cmd, pSpherePipeline);
        cmdBindVertexBuffer(cmd, 1, &pSphereVertexBuffer, &gSphereVertexLayout.mBindings[0].mStride, nullptr);
        cmdBindIndexBuffer(cmd, pSphereIndexBuffer, INDEX_TYPE_UINT16, 0);
        cmdDrawIndexedInstanced(cmd, gSphereIndexCount, 0, gNumPlanets, 0, 0);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        */

        // draw octree marching cubes surfaces
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw octree marching cubes surfaces");
        cmdBindPipeline(cmd, pOctreeMarchingCubesPipeline);
        cmdBindDescriptorSet(cmd, gFrameIndex, pOctreeMarchingCubesDescriptorSetUniforms);
        cmdBindVertexBuffer(cmd, 1, &pOctreeMarchingCubesVertexBuffer[currentGeometry], &gOctreeMarchingCubesVertexLayout.mBindings[0].mStride, nullptr);
        cmdBindIndexBuffer(cmd, pOctreeMarchingCubesIndexBuffer[currentGeometry], INDEX_TYPE_UINT32, 0);
        cmdDrawIndexedInstanced(cmd, octreeMarchingCubesIndicesCount[currentGeometry], 0, 1, 0, 0);   
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        
         // draw octree Wireframes
        if (showWireframe)
        {
            cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw octree Wireframes");
            cmdBindPipeline(cmd, pOctreeWireframesPipeline);
            cmdBindVertexBuffer(cmd, 1, &pOctreeWireframesVertexBuffer[currentGeometry], &gOctreeMarchingCubesVertexLayout.mBindings[0].mStride, nullptr);
            cmdDraw(cmd, octreeWireframesVertexCount[currentGeometry], 0);
            cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        }

        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken); // Draw Skybox/Planets
        cmdBindRenderTargets(cmd, NULL); //Unbinds the currently active render targets


    /************************************************************************************************************************/

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryDesc queryDesc = { 0 };
            cmdEndQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);

            queryDesc = { 1 };
            cmdBeginQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
        }

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw UI");

        // Render UI element
        bindRenderTargets = {};
        bindRenderTargets.mRenderTargetCount = 1;
        bindRenderTargets.mRenderTargets[0] = { pRenderTarget, LOAD_ACTION_LOAD };
        bindRenderTargets.mDepthStencil = { NULL, LOAD_ACTION_DONTCARE };
        cmdBindRenderTargets(cmd, &bindRenderTargets);

        gFrameTimeDraw.mFontColor = 0xff00ffff;
        gFrameTimeDraw.mFontSize = 18.0f;
        gFrameTimeDraw.mFontID = gFontID;
        float2 txtSizePx = cmdDrawCpuProfile(cmd, float2(8.f, 15.f), &gFrameTimeDraw);
        cmdDrawGpuProfile(cmd, float2(8.f, txtSizePx.y + 75.f), gGpuProfileToken, &gFrameTimeDraw);

        cmdDrawUserInterface(cmd);

        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        cmdBindRenderTargets(cmd, NULL);

        // transitions the pRenderTarget from RENDER_TARGET state to PRESENT state, preparing the image for display
        barriers[0] = { pRenderTarget, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_PRESENT };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);

        cmdEndGpuFrameProfile(cmd, gGpuProfileToken);

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryDesc queryDesc = { 1 };
            cmdEndQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
            cmdResolveQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], 0, 2);
        }

        endCmd(cmd);  // the command recording is end here

        // Ensures all GPU resource updates (buffers, textures) are completed before rendering
        FlushResourceUpdateDesc flushUpdateDesc = {};
        flushUpdateDesc.mNodeIndex = 0;
        flushResourceUpdates(&flushUpdateDesc); // submitted semaphore can be obtained from flushUpdateDesc directly
        Semaphore* waitSemaphores[2] = { flushUpdateDesc.pOutSubmittedSemaphore, pImageAcquiredSemaphore };

        // Submit the command buffer to the GPU queue
        QueueSubmitDesc submitDesc = {};
        submitDesc.mCmdCount = 1;
        submitDesc.mSignalSemaphoreCount = 1;
        submitDesc.mWaitSemaphoreCount = TF_ARRAY_COUNT(waitSemaphores);
        submitDesc.ppCmds = &cmd;
        submitDesc.ppSignalSemaphores = &elem.pSemaphore;
        submitDesc.ppWaitSemaphores = waitSemaphores;
        submitDesc.pSignalFence = elem.pFence;
        queueSubmit(pGraphicsQueue, &submitDesc);

        // Present the frame to screen
        QueuePresentDesc presentDesc = {};
        presentDesc.mIndex = (uint8_t)swapchainImageIndex;
        presentDesc.mWaitSemaphoreCount = 1;
        presentDesc.pSwapChain = pSwapChain;
        presentDesc.ppWaitSemaphores = &elem.pSemaphore;
        presentDesc.mSubmitDone = true;

        queuePresent(pGraphicsQueue, &presentDesc);
        flipProfiler();

        gFrameIndex = (gFrameIndex + 1) % gDataBufferCount;
    }

    const char* GetName() { return "99_XinjieInterviewProject"; }


    // don't need to change for octree marching cubes (may need change if using compute shader)
    bool addSwapChain()
    {
        SwapChainDesc swapChainDesc = {};
        swapChainDesc.mWindowHandle = pWindow->handle;
        swapChainDesc.mPresentQueueCount = 1;
        swapChainDesc.ppPresentQueues = &pGraphicsQueue;
        swapChainDesc.mWidth = mSettings.mWidth;
        swapChainDesc.mHeight = mSettings.mHeight;
        swapChainDesc.mImageCount = getRecommendedSwapchainImageCount(pRenderer, &pWindow->handle);
        swapChainDesc.mColorFormat = getSupportedSwapchainFormat(pRenderer, &swapChainDesc, COLOR_SPACE_SDR_SRGB);
        swapChainDesc.mColorSpace = COLOR_SPACE_SDR_SRGB;
        swapChainDesc.mEnableVsync = mSettings.mVSyncEnabled;
        swapChainDesc.mFlags = SWAP_CHAIN_CREATION_FLAG_ENABLE_FOVEATED_RENDERING_VR;
        ::addSwapChain(pRenderer, &swapChainDesc, &pSwapChain);

        return pSwapChain != NULL;
    }

    // don't need to change for octree marching cubes
    bool addDepthBuffer()
    {
        // Add depth buffer
        RenderTargetDesc depthRT = {};
        depthRT.mArraySize = 1;
        depthRT.mClearValue.depth = 0.0f;
        depthRT.mClearValue.stencil = 0;
        depthRT.mDepth = 1;
        depthRT.mFormat = TinyImageFormat_D32_SFLOAT;
        depthRT.mStartState = RESOURCE_STATE_DEPTH_WRITE;
        depthRT.mHeight = mSettings.mHeight;
        depthRT.mSampleCount = SAMPLE_COUNT_1;
        depthRT.mSampleQuality = 0;
        depthRT.mWidth = mSettings.mWidth;
        depthRT.mFlags = TEXTURE_CREATION_FLAG_ON_TILE | TEXTURE_CREATION_FLAG_VR_MULTIVIEW;
        addRenderTarget(pRenderer, &depthRT, &pDepthBuffer);

        return pDepthBuffer != NULL;
    }

    void addDescriptorSets()
    {
        DescriptorSetDesc descPersisent = SRT_SET_DESC(SrtData, Persistent, 1, 0);
        addDescriptorSet(pRenderer, &descPersisent, &pDescriptorSetTexture);
        DescriptorSetDesc descUniforms = SRT_SET_DESC(SrtData, PerFrame, gDataBufferCount, 0);
        addDescriptorSet(pRenderer, &descUniforms, &pDescriptorSetUniforms);

        // octree marching cubes descriptor set
        DescriptorSetDesc descUniformsOMC = SRT_SET_DESC(SrtData, PerFrame, gDataBufferCount, 0);
        addDescriptorSet(pRenderer, &descUniformsOMC, &pOctreeMarchingCubesDescriptorSetUniforms);
    }

    void removeDescriptorSets()
    {
        removeDescriptorSet(pRenderer, pDescriptorSetUniforms);
        removeDescriptorSet(pRenderer, pDescriptorSetTexture);
    
        removeDescriptorSet(pRenderer, pOctreeMarchingCubesDescriptorSetUniforms);
    }

    void addShaders()
    {
        ShaderLoadDesc skyShader = {};
        skyShader.mVert.pFileName = "skybox.vert";
        skyShader.mFrag.pFileName = "skybox.frag";

        ShaderLoadDesc basicShader = {};
        basicShader.mVert.pFileName = "basic.vert";
        basicShader.mFrag.pFileName = "basic.frag";

        ShaderLoadDesc octreeMarchingCubesShader = {};
        octreeMarchingCubesShader.mVert.pFileName = "octreeMarchingCubes.vert";
        octreeMarchingCubesShader.mFrag.pFileName = "octreeMarchingCubes.frag";

        ShaderLoadDesc octreeWireframesShader = {};
        octreeWireframesShader.mVert.pFileName = "octreeWireframes.vert";
        octreeWireframesShader.mFrag.pFileName = "octreeWireframes.frag";

        addShader(pRenderer, &skyShader, &pSkyBoxDrawShader);
        addShader(pRenderer, &basicShader, &pSphereShader);
        addShader(pRenderer, &octreeMarchingCubesShader, &pOctreeMarchingCubesShader);
        addShader(pRenderer, &octreeWireframesShader, &pOctreeWireframesShader);
    }

    void removeShaders()
    {
        removeShader(pRenderer, pSphereShader);
        removeShader(pRenderer, pSkyBoxDrawShader);
        removeShader(pRenderer, pOctreeMarchingCubesShader);
        removeShader(pRenderer, pOctreeWireframesShader);
    }

    void addPipelines()
    {
        RasterizerStateDesc rasterizerStateDesc = {};
        rasterizerStateDesc.mCullMode = CULL_MODE_NONE;

        RasterizerStateDesc sphereRasterizerStateDesc = {};
        sphereRasterizerStateDesc.mCullMode = CULL_MODE_FRONT;

         RasterizerStateDesc octreeMarchingCubesRasterizerStateDesc = {};
         octreeMarchingCubesRasterizerStateDesc.mFrontFace = FRONT_FACE_CW;
         octreeMarchingCubesRasterizerStateDesc.mCullMode = CULL_MODE_BACK;
        // octreeMarchingCubesRasterizerStateDesc.mFillMode = FILL_MODE_wireframe;

        DepthStateDesc depthStateDesc = {};
        depthStateDesc.mDepthTest = true;
        depthStateDesc.mDepthWrite = true;
        depthStateDesc.mDepthFunc = CMP_GEQUAL;

        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_GRAPHICS;
        PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(SrtData, Persistent), SRT_LAYOUT_DESC(SrtData, PerFrame), NULL, NULL);
        GraphicsPipelineDesc& pipelineSettings = desc.mGraphicsDesc;
        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        pipelineSettings.mRenderTargetCount = 1;
        pipelineSettings.pDepthState = &depthStateDesc;
        pipelineSettings.pColorFormats = &pSwapChain->ppRenderTargets[0]->mFormat;
        pipelineSettings.mSampleCount = pSwapChain->ppRenderTargets[0]->mSampleCount;
        pipelineSettings.mSampleQuality = pSwapChain->ppRenderTargets[0]->mSampleQuality;
        pipelineSettings.mDepthStencilFormat = pDepthBuffer->mFormat;
        pipelineSettings.pShaderProgram = pSphereShader;
        pipelineSettings.pVertexLayout = &gSphereVertexLayout;
        pipelineSettings.pRasterizerState = &sphereRasterizerStateDesc;
        pipelineSettings.mVRFoveatedRendering = true;
        addPipeline(pRenderer, &desc, &pSpherePipeline);

        
        // pipeline for octree marching cubes
        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        pipelineSettings.mRenderTargetCount = 1;
        pipelineSettings.pDepthState = &depthStateDesc;
        pipelineSettings.pColorFormats = &pSwapChain->ppRenderTargets[0]->mFormat;
        pipelineSettings.mSampleCount = pSwapChain->ppRenderTargets[0]->mSampleCount;
        pipelineSettings.mSampleQuality = pSwapChain->ppRenderTargets[0]->mSampleQuality;
        pipelineSettings.mDepthStencilFormat = pDepthBuffer->mFormat;
        pipelineSettings.pShaderProgram = pOctreeMarchingCubesShader;
        pipelineSettings.pVertexLayout = &gOctreeMarchingCubesVertexLayout;
        pipelineSettings.pRasterizerState = &octreeMarchingCubesRasterizerStateDesc;
        pipelineSettings.mVRFoveatedRendering = true;
        addPipeline(pRenderer, &desc, &pOctreeMarchingCubesPipeline);


         // pipeline for octree marching cubes
        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_LINE_LIST;
        pipelineSettings.pShaderProgram = pOctreeWireframesShader;
        addPipeline(pRenderer, &desc, &pOctreeWireframesPipeline);

        // layout and pipeline for skybox draw
        VertexLayout vertexLayout = {};
        vertexLayout.mBindingCount = 1;
        vertexLayout.mBindings[0].mStride = sizeof(float4);
        vertexLayout.mAttribCount = 1;
        vertexLayout.mAttribs[0].mSemantic = SEMANTIC_POSITION;
        vertexLayout.mAttribs[0].mFormat = TinyImageFormat_R32G32B32A32_SFLOAT;
        vertexLayout.mAttribs[0].mBinding = 0;
        vertexLayout.mAttribs[0].mLocation = 0;
        vertexLayout.mAttribs[0].mOffset = 0;
        pipelineSettings.pVertexLayout = &vertexLayout;
        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        pipelineSettings.pDepthState = NULL;
        pipelineSettings.pRasterizerState = &rasterizerStateDesc;
        pipelineSettings.pShaderProgram = pSkyBoxDrawShader; //-V519
        addPipeline(pRenderer, &desc, &pSkyBoxDrawPipeline);



    }

    void addOctreeMarchingCubesPipeline() {

    }

    void removePipelines()
    {
        removePipeline(pRenderer, pSkyBoxDrawPipeline);
        removePipeline(pRenderer, pSpherePipeline);
        removePipeline(pRenderer, pOctreeMarchingCubesPipeline);
        removePipeline(pRenderer, pOctreeWireframesPipeline);
    }

    void prepareDescriptorSets()
    {
        // Prepare descriptor sets
        DescriptorData params[7] = {};
        params[0].mIndex = SRT_RES_IDX(SrtData, Persistent, gRightTexture);
        params[0].ppTextures = &pSkyBoxTextures[0];
        params[1].mIndex = SRT_RES_IDX(SrtData, Persistent, gLeftTexture);
        params[1].ppTextures = &pSkyBoxTextures[1];
        params[2].mIndex = SRT_RES_IDX(SrtData, Persistent, gTopTexture);
        params[2].ppTextures = &pSkyBoxTextures[2];
        params[3].mIndex = SRT_RES_IDX(SrtData, Persistent, gBotTexture);
        params[3].ppTextures = &pSkyBoxTextures[3];
        params[4].mIndex = SRT_RES_IDX(SrtData, Persistent, gFrontTexture);
        params[4].ppTextures = &pSkyBoxTextures[4];
        params[5].mIndex = SRT_RES_IDX(SrtData, Persistent, gBackTexture);
        params[5].ppTextures = &pSkyBoxTextures[5];
        params[6].mIndex = SRT_RES_IDX(SrtData, Persistent, gSampler);
        params[6].ppSamplers = &pSkyBoxSampler;
        updateDescriptorSet(pRenderer, 0, pDescriptorSetTexture, TF_ARRAY_COUNT(params), params);

        // create descriptor set for uniform buffer
        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            DescriptorData uParams[1] = {};
            uParams[0].mIndex = SRT_RES_IDX(SrtData, PerFrame, gUniformBlock);
            uParams[0].ppBuffers = &pUniformBuffer[i];
            updateDescriptorSet(pRenderer, i, pDescriptorSetUniforms, 1, uParams);
        }
    }


    void prepareOctreeMarchingCubesDescriptorSets() {

          // add the uniform buffers to the descriptor set
        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            DescriptorData uParams[1] = {};
            uParams[0].mIndex = SRT_RES_IDX(SrtData, PerFrame, gOMCUniformBlock); // search the binding index
            uParams[0].ppBuffers = &pOctreeMarchingCubesUniformBuffer[i];
            updateDescriptorSet(pRenderer, i, pOctreeMarchingCubesDescriptorSetUniforms, 1, uParams);
        }

    }
};
DEFINE_APPLICATION_MAIN(Transformations)
