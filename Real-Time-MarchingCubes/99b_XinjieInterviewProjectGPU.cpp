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

#define MAX_PLANETS 20 // Does not affect test, just for allocating space in uniform block. Must match with shader.


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


struct UniformBlock
{
    CameraMatrix mProjectView;
    CameraMatrix mSkyProjectView;
    mat4         mToWorldMat[MAX_PLANETS];
    vec4         mColor[MAX_PLANETS];
    float        mGeometryWeight[MAX_PLANETS][4];

    // Point Light Information
    vec4 mLightPosition;
    vec4 mLightColor;
};

struct MarchingCubesUniformBlock
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
    float  alpha;
    float3 padding;
    bool   ifMetaBalls;
    float3 padding2;
};

// But we only need Two sets of resources (one in flight and one being used on CPU)
const uint32_t gDataBufferCount = 2;

Renderer*  pRenderer = NULL;
Queue*     pGraphicsQueue = NULL;
GpuCmdRing gGraphicsCmdRing = {};

SwapChain*    pSwapChain = NULL;
RenderTarget* pDepthBuffer = NULL;
Semaphore*    pImageAcquiredSemaphore = NULL;


Shader*        pSkyBoxDrawShader = NULL;
Buffer*        pSkyBoxVertexBuffer = NULL;
Pipeline*      pSkyBoxDrawPipeline = NULL;
Texture*       pSkyBoxTextures[6];
Sampler*       pSkyBoxSampler = {};
DescriptorSet* pDescriptorSetTexture = { NULL };
DescriptorSet* pDescriptorSetUniforms = { NULL };

// marching cubes compute shaders setting
Shader*        pComputeShader = NULL;
Pipeline*      pComputePipelineMCOne = NULL;
Pipeline*      pComputePipelineMCTwo = NULL;
Buffer*        pInputBuffer[gDataBufferCount] = { NULL };
Buffer*        pOutputBuffer[gDataBufferCount] = { NULL };
DescriptorSet* pDescriptorSetCompute = { NULL };
float*         pOutputData = NULL;

Shader*        pComputeShaderMC1 = NULL;
Shader*        pComputeShaderMC2 = NULL;
Buffer*        pSDFBuffer[gDataBufferCount] = { NULL };
Buffer*        pTriCountBuffer[gDataBufferCount] = { NULL };

Pipeline*      pPrefixThreadSumPipeline = NULL;
Pipeline*      pPrefixBlockSumPipeline = NULL;
Pipeline*      pPrefixFinalSumPipeline = NULL;
Buffer*        pPrefixSumBuffer[gDataBufferCount] = { NULL };
Buffer*        pBlockSumBuffer[gDataBufferCount] = { NULL };
Buffer*        pScannedBlockSumBuffer[gDataBufferCount] = { NULL };
Buffer*        pScratchBuffer[gDataBufferCount] = { NULL };
Buffer*        pNumVerticesBuffer[gDataBufferCount] = { NULL };
Buffer*        pIndirectDrawArgBuffer[gDataBufferCount] = { NULL };


Shader*        pComputeShaderPrefixSum = NULL;
Shader*        pComputeShaderPrefixBlockSum = NULL;
Shader*        pComputeShaderPrefixFinalSum = NULL;

// marching cubes rendering setting
Buffer*        pTriangleBuffer[gDataBufferCount] = { NULL };

Pipeline*      pMarchingCubesGraphicsPipeline = NULL;
Shader*        pMarchingCubesGraphicsShader = NULL;

MarchingCubesUniformBlock mcUniformData;
Buffer*        pMarchingCubesGraphicsUniformBuffer[gDataBufferCount] = { NULL };

Buffer*        pUniformBuffer[gDataBufferCount] = { NULL };

uint32_t     gFrameIndex = 0;
ProfileToken gGpuProfileToken = PROFILE_INVALID_TOKEN;

UniformBlock     gUniformData;

ICameraController* pCameraController = NULL;

UIComponent* pGuiWindow = NULL;

UIComponent* pGuiSliderBar = NULL;

uint32_t gFontID = 0;

QueryPool* pPipelineStatsQueryPool[gDataBufferCount] = {};

//const char* pSkyBoxImageFileNames[] = { "Skybox_right1.tex",  "Skybox_left2.tex",  "Skybox_top3.tex",
                         //               "Skybox_bottom4.tex", "Skybox_front5.tex", "Skybox_back6.tex" };

const char*  pSkyBoxImageFileNames[] = { "skybox/hw_sahara/sahara_rt.tex", "skybox/hw_sahara/sahara_lf.tex",
                                         "skybox/hw_sahara/sahara_up.tex", "skybox/hw_sahara/sahara_dn.tex",
                                         "skybox/hw_sahara/sahara_ft.tex", "skybox/hw_sahara/sahara_bk.tex" };
float       frequency = 0.1f;
float        amplitude = 16.0f;
float*       SDFData = nullptr;
float        currentTime = 0.0f;
bool         ifEnableAnimation = false;
bool         ifEnable3DNoise = false;
float        animationSpeed = 3.0f;


const int numMetaballs = 4;

// Centers and radiuses of the metaballs (x, y, z)
float3 centers[4] = { 
     float3(42.0f, 32.0f, 15.0f),
    float3(32.0f, 47.0f, 32.0f),
    float3(22.0f, 12.0f, 32.0f), 
    float3(32.0f, 22.0f, 32.0f) };
float radiuses[4] = { 8.0f, 8.0f, 8.0f, 5.0f };
float metaballThreshold = 1.0f;
bool  ifMetaballs = false;

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

float MultiMetaballField(float x, float y, float z, const float3* centers, const float* radiuses, int numMetaballs, float threshold)
{
    float3 p = float3(x, y, z);
    float  sumVal = 0.0f;

    for (int i = 0; i < numMetaballs; i++)
    {
        float3 d = p - centers[i];
        float  distSq = d.x * d.x + d.y * d.y + d.z * d.z;
        sumVal += (radiuses[i] * radiuses[i]) / (distSq + 1e-6f);
    }

    return sumVal - threshold;
}

float computeTerrianHeight(float x, float y, float z, float frequency, float amplitude, float time)
{
    // Frequency and amplitude are tunable parameters

    float total = 0.0f;
    float noise = 0.0f;
 
    for (int i = 0; i < 2; i++)
    {   
        // use 3D noise for 3D terrain
        if (ifEnable3DNoise) 
        noise = db::perlin(x * frequency, y * frequency,  z * frequency + 0.02f * time);
        // use 2D noise for terrain heightmap
        else noise = db::perlin(x * frequency, z * frequency + 0.02f * time);
        
        noise = clamp(noise, -0.3f, 0.3f);
        total += amplitude * noise;
        frequency *= 2.0f; // Double the frequency each octave
        amplitude *= 0.5f; // Halve the amplitude each octave
    }
      return total;
  //    return amplitude * abs(noise);
  //  return amplitude * abs(db::perlin(x * frequency, z * frequency + 0.02f * time));
    //return amplitude * (db::perlin(x * frequency, z * frequency)) + sin(time * 0.5f) * 5.0f;
}

float TerrainSDF(float x, float y, float z, float frequency, float amplitude, float time)
{
    // 20.0 is the offset, otherwise some xz coordinates will have no surfaces
    return y - 20.0f - computeTerrianHeight(x, y,  z, frequency, amplitude, time) - 0.05f;
}

float* generateSDF(float freq, float amplitude, float time) {

    uint32_t width = 64;
    uint32_t length = 64;
    uint32_t height = 64;

    size_t bufferSize = sizeof(float) * width * length * height;
    float* SDFValues = (float*)tf_calloc(1, bufferSize);

    for (int i = 0; i < height; i++)    // y axis
        for (int k = 0; k < width; k++) // x axis
            for (int j = 0; j < length; j++) // z axis, we use -z axis
            { 
                SDFValues[i * width * length + k * length + j] = TerrainSDF(k, i, j, freq, amplitude, time);
            }
    return SDFValues;

}

float* generateMetaBallsSDF(const float3* centers, const float* radiuses, int numMetaballs, float threshold)
{
    uint32_t width = 64;
    uint32_t length = 64;
    uint32_t height = 64;

    size_t bufferSize = sizeof(float) * width * length * height;
    float* SDFValues = (float*)tf_calloc(1, bufferSize);

    for (int i = 0; i < height; i++)         // y axis
        for (int k = 0; k < width; k++)      // x axis
            for (int j = 0; j < length; j++) // z axis, we use -z axis
            {
                SDFValues[i * width * length + k * length + j] = MultiMetaballField( k, i, j, centers, radiuses, numMetaballs, threshold);
            }
    return SDFValues;
}


void initBuffers() {

    
    float  inputData[256] = {};
    for (int i = 0; i < 256; i++)
    {
        inputData[i] = (float)i; // Fill buffer with numbers 0, 1, 2, ..., 255
    }
    
    // size of marching cubes grids
    uint32_t width = 64;
    uint32_t length = 64;
    uint32_t height = 64;

    size_t bufferSize = sizeof(float) * width * length * height;
    float* SDFValues = (float*) tf_calloc(1, bufferSize);

    for (int i = 0; i < gDataBufferCount; i++)
    {
        // Create Input Buffer (initial test buffer for compute shader)
        BufferLoadDesc inputBufferDesc = {};
        inputBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER | DESCRIPTOR_TYPE_BUFFER;
        inputBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        inputBufferDesc.mDesc.mSize = 256 * sizeof(float);
        inputBufferDesc.mDesc.mElementCount = 256;
        inputBufferDesc.mDesc.mStructStride = sizeof(float);
        inputBufferDesc.pData = &inputData; // Bind input data to the buffer
        inputBufferDesc.ppBuffer = &pInputBuffer[i];
     // inputBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
       // inputBufferDesc.mDesc.pName = "inputBuffer";
        addResource(&inputBufferDesc, NULL);

        // Create Output Buffer (empty) (initial test buffer for compute shader)
        BufferLoadDesc outputBufferDesc = {};
        outputBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER;
        outputBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
        outputBufferDesc.mDesc.mSize = 256 * sizeof(float);
        outputBufferDesc.mDesc.mElementCount = 256;
        outputBufferDesc.mDesc.mStructStride = sizeof(float);
        outputBufferDesc.mDesc.mStartState = RESOURCE_STATE_COMMON;
     // outputBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        outputBufferDesc.pData = nullptr; // Output buffer starts empty
        outputBufferDesc.ppBuffer = &pOutputBuffer[i];
        addResource(&outputBufferDesc, NULL);



        // Create SDF Input Buffer
        BufferLoadDesc inputSDFBufferDesc = {};
        inputSDFBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER | DESCRIPTOR_TYPE_BUFFER;
        inputSDFBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        inputSDFBufferDesc.mDesc.mSize = bufferSize;
        inputSDFBufferDesc.mDesc.mElementCount = width * height * length;
        inputSDFBufferDesc.mDesc.mStructStride = sizeof(float);
        inputSDFBufferDesc.pData = SDFValues; // Bind input data to the buffer
        inputSDFBufferDesc.ppBuffer = &pSDFBuffer[i];
        inputSDFBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT; 
        addResource(&inputSDFBufferDesc, NULL);

        // Create Output triangle count Buffer (empty)
        BufferLoadDesc outputTriCountBufferDesc = {};
        outputTriCountBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER;
        outputTriCountBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        outputTriCountBufferDesc.mDesc.mSize = (width - 1) * (height - 1) * (length - 1) * sizeof(uint);
        outputTriCountBufferDesc.mDesc.mElementCount = (width - 1) * (height - 1) * (length - 1);
        outputTriCountBufferDesc.mDesc.mStructStride = sizeof(uint);
        outputTriCountBufferDesc.mDesc.mStartState = RESOURCE_STATE_COMMON;
     // outputTriCountBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        outputTriCountBufferDesc.pData = nullptr; // Output buffer starts empty
        outputTriCountBufferDesc.ppBuffer = &pTriCountBuffer[i];
        addResource(&outputTriCountBufferDesc, NULL);

        // Create Output prefix sum Buffer (empty)
        BufferLoadDesc prefixSumBufferDesc = {};
        prefixSumBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER;
        prefixSumBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        prefixSumBufferDesc.mDesc.mSize = (width - 1) * (height - 1) * (length - 1) * sizeof(uint);
        prefixSumBufferDesc.mDesc.mElementCount = (width - 1) * (height - 1) * (length - 1);
        prefixSumBufferDesc.mDesc.mStructStride = sizeof(uint);
        prefixSumBufferDesc.mDesc.mStartState = RESOURCE_STATE_COMMON;
        prefixSumBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        prefixSumBufferDesc.pData = nullptr; // Output buffer starts empty
        prefixSumBufferDesc.ppBuffer = &pPrefixSumBuffer[i];
        addResource(&prefixSumBufferDesc, NULL);

         // Create block sum Buffer (empty)
        BufferLoadDesc blockSumBufferDesc = {};
        blockSumBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER;
        blockSumBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        blockSumBufferDesc.mDesc.mSize = ((width * height * length) / 256) * sizeof(uint);
        blockSumBufferDesc.mDesc.mElementCount = (width * height * length) / 256;
        blockSumBufferDesc.mDesc.mStructStride = sizeof(uint);
        blockSumBufferDesc.mDesc.mStartState = RESOURCE_STATE_COMMON;
    //  blockSumBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        blockSumBufferDesc.pData = nullptr; 
        blockSumBufferDesc.ppBuffer = &pBlockSumBuffer[i];
        addResource(&blockSumBufferDesc, NULL);

         // Create scanned block sum Buffer (empty)
        BufferLoadDesc scannedBlockSumBufferDesc = {};
        scannedBlockSumBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER;
        scannedBlockSumBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        scannedBlockSumBufferDesc.mDesc.mSize = ((width * height * length) / 256) * sizeof(uint);
        scannedBlockSumBufferDesc.mDesc.mElementCount = (width * height * length) / 256;
        scannedBlockSumBufferDesc.mDesc.mStructStride = sizeof(uint);
        scannedBlockSumBufferDesc.mDesc.mStartState = RESOURCE_STATE_COMMON;
     // scannedBlockSumBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        scannedBlockSumBufferDesc.pData = nullptr; 
        scannedBlockSumBufferDesc.ppBuffer = &pScannedBlockSumBuffer[i];
        addResource(&scannedBlockSumBufferDesc, NULL);

        prefixSumBufferDesc.ppBuffer = &pScratchBuffer[i];
        prefixSumBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        addResource(&prefixSumBufferDesc, NULL);


        // contain only one element (which is the number of vertices of the surfaces)
        BufferLoadDesc numVerticesBufferDesc = {};
        numVerticesBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER;
        numVerticesBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
        numVerticesBufferDesc.mDesc.mSize = sizeof(uint);
        numVerticesBufferDesc.mDesc.mElementCount = 1;
        numVerticesBufferDesc.mDesc.mStructStride = sizeof(uint);
        numVerticesBufferDesc.mDesc.mStartState = RESOURCE_STATE_COMMON;
        numVerticesBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        numVerticesBufferDesc.pData = nullptr;
        numVerticesBufferDesc.ppBuffer = &pNumVerticesBuffer[i];
        addResource(&numVerticesBufferDesc, nullptr);

        BufferLoadDesc triangleBufferDesc = {};
        triangleBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER;
        triangleBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        triangleBufferDesc.mDesc.mSize = (width - 1) * (height - 1) * (length - 1) * 5 * 3 * 2 * sizeof(float);
        triangleBufferDesc.mDesc.mElementCount = (width - 1) * (height - 1) * (length - 1) * 5 * 3 * 2;
        triangleBufferDesc.mDesc.mStructStride = sizeof(float);
        triangleBufferDesc.mDesc.mStartState = RESOURCE_STATE_COMMON;
       // triangleBufferDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        triangleBufferDesc.pData = nullptr;
        triangleBufferDesc.ppBuffer = &pTriangleBuffer[i];
        addResource(&triangleBufferDesc, nullptr);
        

        BufferLoadDesc indirectBufferDesc = {};
        indirectBufferDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDIRECT_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER;
        indirectBufferDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        indirectBufferDesc.mDesc.mElementCount = 4; // or more if you store multiple commands
        indirectBufferDesc.mDesc.mStructStride = sizeof(uint32_t);
        indirectBufferDesc.mDesc.mSize = sizeof(uint32_t) * 4;
        indirectBufferDesc.mDesc.mStartState = RESOURCE_STATE_UNORDERED_ACCESS; // so a compute shader can fill it
        indirectBufferDesc.mDesc.pName = "Indirect Draw Buffer";
        indirectBufferDesc.pData = nullptr;
        indirectBufferDesc.ppBuffer = &pIndirectDrawArgBuffer[i];
        addResource(&indirectBufferDesc, nullptr);
    }
    waitForAllResourceLoads();
    tf_free(SDFValues);

}




class RealTimeMarchingCubes: public IApp
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

        // marching cubes graphics uniform buffer
        BufferLoadDesc ubDescMC = {};
        ubDescMC.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubDescMC.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        ubDescMC.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        ubDescMC.pData = NULL;
        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            ubDescMC.mDesc.pName = "MarchingCubesGraphicsUniformBuffer";
            ubDescMC.mDesc.mSize = sizeof(MarchingCubesUniformBlock);
            ubDescMC.ppBuffer = &pMarchingCubesGraphicsUniformBuffer[i];
            addResource(&ubDescMC, NULL);
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

        CameraMotionParameters cmp{ 80.0f, 300.0f, 100.0f };
      //  CameraMotionParameters cmp{ 160.0f, 600.0f, 200.0f };
        vec3                   camPos{ 90.0f, 60.0f, -90.0f };
        vec3                   lookAt{ vec3(0) };

        pCameraController = initFpsCameraController(camPos, lookAt);

        pCameraController->setMotionParameters(cmp);

        mcUniformData.La = float4(1.0f, 1.0f, 1.0f, 1.0f);
        mcUniformData.Ld = float4(1.2f, 1.2f, 1.2f, 1.0f);
        mcUniformData.Ls = float4(1.0f, 1.0f, 1.0f, 1.0f);
        mcUniformData.ka = float4(0.1f, 0.1f, 0.1f, 1.0f);
        mcUniformData.kd = float4(1.0f, 1.0f, 1.0f, 1.0f);
        mcUniformData.ks = float4(0.2f, 0.2f, 0.2f, 1.0f);
        mcUniformData.lightDirection = float4(0.0f, 1.0f, 0.0f, 0.0f);
        mcUniformData.alpha = 1.0f;
        mcUniformData.padding = float3(0.f, 0.f, 0.f);
        mcUniformData.ifMetaBalls = ifMetaballs;
        mcUniformData.padding2 = float3(0.f, 0.f, 0.f);

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
            removeResource(pInputBuffer[i]);
            removeResource(pOutputBuffer[i]);

            removeResource(pSDFBuffer[i]);
            removeResource(pTriCountBuffer[i]);

            removeResource(pPrefixSumBuffer[i]);
            removeResource(pBlockSumBuffer[i]);
            removeResource(pScannedBlockSumBuffer[i]);
            removeResource(pScratchBuffer[i]);
            removeResource(pTriangleBuffer[i]);
            removeResource(pNumVerticesBuffer[i]);
            removeResource(pIndirectDrawArgBuffer[i]);

            removeResource(pMarchingCubesGraphicsUniformBuffer[i]);

            removeResource(pUniformBuffer[i]);
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
            addComputeDescriptorSets();
        }

        if (pReloadDesc->mType & (RELOAD_TYPE_RESIZE | RELOAD_TYPE_RENDERTARGET))
        {
            // we only need to reload gui when the size of window changed
            loadProfilerUI(mSettings.mWidth, mSettings.mHeight);

            UIComponentDesc guiDesc = {};
            guiDesc.mStartPosition = vec2(mSettings.mWidth * 0.01f, mSettings.mHeight * 0.35f);
            uiAddComponent(GetName(), &guiDesc, &pGuiWindow);

            // set GUI for controlling terrain parameters
            UIComponentDesc guiDesc2 = {};
            guiDesc2.mStartPosition = vec2(mSettings.mWidth * 0.01f, mSettings.mHeight * 0.68f);
            uiAddComponent("Terrain Parameters", &guiDesc2, &pGuiSliderBar);

            /*
            SliderUintWidget vertexLayoutWidget;
            vertexLayoutWidget.mMin = 0;
            vertexLayoutWidget.mMax = 1;
            vertexLayoutWidget.mStep = 1;
            vertexLayoutWidget.pData = &gSphereLayoutType;
            UIWidget* pVLw = uiAddComponentWidget(pGuiWindow, "Vertex Layout", &vertexLayoutWidget, WIDGET_TYPE_SLIDER_UINT);
            uiSetWidgetOnEditedCallback(pVLw, nullptr, reloadRequest);
            */

            // set float widget for controlling terrain parameters
            SliderFloatWidget terrainFrequencyControl = {};
            terrainFrequencyControl.pData = &frequency;
            terrainFrequencyControl.mMin = 0.0f;
            terrainFrequencyControl.mMax = 0.2f;
            terrainFrequencyControl.mStep = 0.0001f;

            SliderFloatWidget terrainAmplitudeControl = {};
            terrainAmplitudeControl.pData = &amplitude;
            terrainAmplitudeControl.mMin = 0.0f;
            terrainAmplitudeControl.mMax = 30.0f;
            terrainAmplitudeControl.mStep = 0.0001f;

            SliderFloatWidget animationSpeedControl = {};
            animationSpeedControl.pData = &animationSpeed;
            animationSpeedControl.mMin = 1.0f;
            animationSpeedControl.mMax = 10.0f;
            animationSpeedControl.mStep = 0.0001f;

            CheckboxWidget animationEnabled = {};
            animationEnabled.pData = (bool*)&ifEnableAnimation;

            CheckboxWidget threeDNoiseEnabled = {};
            threeDNoiseEnabled.pData = (bool*)&ifEnable3DNoise;

            CheckboxWidget switchToMetaBall = {};
            switchToMetaBall.pData = (bool*)&ifMetaballs;

            luaRegisterWidget(uiAddComponentWidget(pGuiSliderBar, "Terrain Frequency", &terrainFrequencyControl, WIDGET_TYPE_SLIDER_FLOAT));
            luaRegisterWidget(uiAddComponentWidget(pGuiSliderBar, "Terrain Amplitude", &terrainAmplitudeControl, WIDGET_TYPE_SLIDER_FLOAT));
            luaRegisterWidget(uiAddComponentWidget(pGuiSliderBar, "Animation Speed", &animationSpeedControl, WIDGET_TYPE_SLIDER_FLOAT));
            luaRegisterWidget(uiAddComponentWidget(pGuiSliderBar, "Enable Animation (surfaces changes over time)", &animationEnabled, WIDGET_TYPE_CHECKBOX));
            luaRegisterWidget(uiAddComponentWidget(pGuiSliderBar, "Enable 3D Noises", &threeDNoiseEnabled, WIDGET_TYPE_CHECKBOX));
            luaRegisterWidget(uiAddComponentWidget(pGuiSliderBar, "Meta Ball examples", &switchToMetaBall, WIDGET_TYPE_CHECKBOX));

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

            initBuffers();
            addPipelines();
            addComputePipelines();
            addMarchingCubesGraphicsPipeline();
        }

        prepareDescriptorSets();
        prepareComputeDescriptorSet();

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
        
        // Following are the debugging code for compute shaders
        /*
         pOutputData = (float*)tf_malloc(256 * sizeof(float));
        // pOutputData = (float*)pOutputBuffer[gFrameIndex]->pCpuMappedAddress;

        memcpy(pOutputData, pOutputBuffer[gFrameIndex]->pCpuMappedAddress, sizeof(float) * 256);

        for (int i = 0; i < 256; i++)
            LOGF(LogLevel::eINFO, "ii : %f", pOutputData[i]);

        tf_free(pOutputData);
        */
        
      //   uint* pOutputData = (uint*)tf_malloc(63 * 63 * 63 * sizeof(uint));
    //     pOutputData = (float*)pOutputBuffer[gFrameIndex]->pCpuMappedAddress;
      //   uint* pOutputData2 = (uint*)tf_malloc(1024 * sizeof(uint));
       
        // memcpy(pOutputData, pPrefixSumBuffer[gFrameIndex]->pCpuMappedAddress, 63 * 63 * 63 * sizeof(uint));
    //     memcpy(pOutputData2, pScannedBlockSumBuffer[gFrameIndex]->pCpuMappedAddress, 1024 * sizeof(uint));
    //     memcpy(pOutputData, pPrefixSumBuffer[gFrameIndex]->pCpuMappedAddress, 63 * 63 * 63 * sizeof(uint));

     //   for (int i = 0; i < 63 * 63 * 63; i++)
      //   for (int i = 0; i < 63 * 63 * 63; i++)
      //      LOGF(LogLevel::eINFO, "%u : %u", i, pOutputData[i]);

       //  for (int i = 0; i < 1024; i++)
            //     LOGF(LogLevel::eINFO, "%u : %u", i, pOutputData2[i]);

      //   tf_free(pOutputData);
      //   tf_free(pOutputData2);

     //    float* pOutputData = (float*)tf_malloc(48000 * 15 * sizeof(float));
     //   memcpy(pOutputData, pTriangleBuffer[gFrameIndex]->pCpuMappedAddress, 48000 * 15 * sizeof(float));

       //  for (int i = 0; i < 48000 * 15; i++)
          //      LOGF(LogLevel::eINFO, "%u : %f", i, pOutputData[i]);

     //    tf_free(pOutputData);

        unloadFontSystem(pReloadDesc->mType);
        unloadUserInterface(pReloadDesc->mType);

        if (pReloadDesc->mType & (RELOAD_TYPE_SHADER | RELOAD_TYPE_RENDERTARGET))
        {
            removePipelines();
        }

        if (pReloadDesc->mType & (RELOAD_TYPE_RESIZE | RELOAD_TYPE_RENDERTARGET))
        {
            removeSwapChain(pRenderer, pSwapChain);
            removeRenderTarget(pRenderer, pDepthBuffer);
            uiRemoveComponent(pGuiWindow);
            uiRemoveComponent(pGuiSliderBar);
            unloadProfilerUI();
        }

        if (pReloadDesc->mType & RELOAD_TYPE_SHADER)
        {
            removeDescriptorSets();
            removeComputeDescriptorSets();
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
                pCameraController->resetView();
            }
            if (inputGetValue(0, CUSTOM_TOGGLE_FULLSCREEN))
            {
                toggleFullscreen(pWindow);
            }
            if (inputGetValue(0, CUSTOM_TOGGLE_UI))
            {
                uiToggleActive();
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
  

        // update camera with time
        mat4 viewMat = pCameraController->getViewMatrix();

        const float  aspectInverse = (float)mSettings.mHeight / (float)mSettings.mWidth;
        const float  horizontal_fov = PI / 2.0f;
        CameraMatrix projMat = CameraMatrix::perspectiveReverseZ(horizontal_fov, aspectInverse, 0.1f, 1000.0f);
        gUniformData.mProjectView = projMat * viewMat;

        // point light parameters
        gUniformData.mLightPosition = vec4(0, 0, 0, 0);
        gUniformData.mLightColor = vec4(0.9f, 0.9f, 0.7f, 1.0f); // Pale Yellow

        // project and view martix for octree marching cubes pipeline
        mcUniformData.mView = viewMat;
        mcUniformData.mProject = projMat;
        mcUniformData.modelMatrix = mat4::identity();

        mcUniformData.color = vec4(0.3f, 0.8f, 0.5f, 1.0f);
        mcUniformData.normalMatrix = mcUniformData.mView * mcUniformData.modelMatrix; // no non-uniform scaling, so don't need inverse and transpose
        mcUniformData.ifMetaBalls = ifMetaballs;
  

        // if the animation is enabled, the SDF values and terrain surfaces will change with time
        if (ifEnableAnimation) currentTime += deltaTime * animationSpeed;


        // update SDF values (every frame)
        if (!ifMetaballs)
        {
            SDFData = generateSDF(frequency, amplitude, currentTime);
        }
        else
        {
            float offset = 5.0f * sin(currentTime);
            centers[0] = float3(42.0f - offset, 32.0f, 15.0f);
            centers[1] = float3(32.0f, 47.0f - offset, 32.0f);
            centers[2] = float3(22.0f + offset, 12.0f, 32.0f);
            centers[3] = float3(32.0f, 22.0f + offset, 32.0f);

            radiuses[0] = 8.0f + 2.0f * sin(currentTime * 2.0f);
            radiuses[1] = 8.0f + 1.5f * cos(currentTime * 1.5f);
            radiuses[2] = 8.0f + 1.0f * sin(currentTime * 3.0f);
            radiuses[3] = 5.0f + 1.0f * sin(currentTime * 3.5f);

            SDFData = generateMetaBallsSDF(centers, radiuses, numMetaballs, metaballThreshold);
        }

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

        // Update uniform buffers
        BufferUpdateDesc viewProjCbv = { pUniformBuffer[gFrameIndex] };
        beginUpdateResource(&viewProjCbv);
        memcpy(viewProjCbv.pMappedData, &gUniformData, sizeof(gUniformData));
        endUpdateResource(&viewProjCbv);

        // Update marching cubes uniform buffer
        BufferUpdateDesc viewProjCbv2 = { pMarchingCubesGraphicsUniformBuffer[gFrameIndex] };
        beginUpdateResource(&viewProjCbv2);
        memcpy(viewProjCbv2.pMappedData, &mcUniformData, sizeof(mcUniformData));
        endUpdateResource(&viewProjCbv2);

      
        // Update SDF buffer
        BufferUpdateDesc SDFDataCbv = { pSDFBuffer[gFrameIndex] };
        beginUpdateResource(&SDFDataCbv);
        memcpy(SDFDataCbv.pMappedData, SDFData, sizeof(float) * 64 * 64 * 64);
        endUpdateResource(&SDFDataCbv);
        tf_free(SDFData);
       

        // Reset cmd pool for this frame
        resetCmdPool(pRenderer, elem.pCmdPool);

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

        Cmd* cmd = elem.pCmds[0];
        beginCmd(cmd);

        cmdBeginGpuFrameProfile(cmd, gGpuProfileToken);
        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            cmdResetQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], 0, 2);
            QueryDesc queryDesc = { 0 };
            cmdBeginQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
        }

        RenderTargetBarrier barriers[] = {
            { pRenderTarget, RESOURCE_STATE_PRESENT, RESOURCE_STATE_RENDER_TARGET },
        };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Compute and draw Marching Cubes Surfaces");

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
        
        BufferBarrier bufBarrier = {};
      //  bufBarrier.pBuffer = pOutputBuffer[gFrameIndex];
        bufBarrier.pBuffer = pTriCountBuffer[gFrameIndex];
        bufBarrier.mCurrentState = RESOURCE_STATE_COPY_SOURCE;
        bufBarrier.mNewState = RESOURCE_STATE_UNORDERED_ACCESS;
       
        cmdResourceBarrier(cmd, 1, &bufBarrier, 0, NULL, 0, NULL);

        // compute pipeline
        // Marching Cubes First pass   
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Compute1: TriNums");
        cmdBindPipeline(cmd, pComputePipelineMCOne);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetCompute);
        cmdDispatch(cmd, 8, 8, 8);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);


        // UAV barrier to ensure Marching Cubes First pass  finishing writing before next pass reads the data in pTriCountBuffer 
        BufferBarrier barrier[] = { { pTriCountBuffer[gFrameIndex], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS }
          };
        cmdResourceBarrier(cmd, 1, barrier, 0, nullptr, 0, nullptr);

        // Prefix Sum subpass 1
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "PrefixSum subpass 1");
        cmdBindPipeline(cmd, pPrefixThreadSumPipeline);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetCompute);
        cmdDispatch(cmd, 1024, 1, 1);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        BufferBarrier barriersPrefixSum[] = { 
            { pPrefixSumBuffer[gFrameIndex], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS }, 
            { pBlockSumBuffer[gFrameIndex], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS },
            { pScannedBlockSumBuffer[gFrameIndex], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS },
            { pScratchBuffer[gFrameIndex], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS } 
        
        };
        cmdResourceBarrier(cmd, 4, barriersPrefixSum, 0, nullptr, 0, nullptr);


        // Prefix Sum subpass 2
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "PrefixSum subpass 2");
        cmdBindPipeline(cmd, pPrefixBlockSumPipeline);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetCompute);
        cmdDispatch(cmd, 1, 1, 1);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

         cmdResourceBarrier(cmd, 4, barriersPrefixSum, 0, nullptr, 0, nullptr);

        // Prefix Sum subpass 3
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "PrefixSum subpass 3");
        cmdBindPipeline(cmd, pPrefixFinalSumPipeline);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetCompute);
        cmdDispatch(cmd, 1024, 1, 1);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        
        cmdResourceBarrier(cmd, 4, barriersPrefixSum, 0, nullptr, 0, nullptr);

        // Marching Cubes Second pass
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Compute2: surface vertices");
        cmdBindPipeline(cmd, pComputePipelineMCTwo);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetCompute);
        cmdDispatch(cmd, 8, 8, 8);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        BufferBarrier barrierForRendering[] = { 
        { pTriangleBuffer[gFrameIndex], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS }, 
        { pNumVerticesBuffer[gFrameIndex], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_GENERIC_READ },
        { pIndirectDrawArgBuffer[gFrameIndex], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_INDIRECT_ARGUMENT }
        };
        cmdResourceBarrier(cmd, 3, barrierForRendering, 0, nullptr, 0, nullptr);

        // get the numVertices from GPU for cmdDraw
         uint* numVertices = (uint*)tf_malloc(sizeof(uint));
        

         memcpy(numVertices, pNumVerticesBuffer[gFrameIndex]->pCpuMappedAddress, sizeof(uint));
       

        // Marching Cubes surfaces rendering
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Rendering");
        cmdBindPipeline(cmd, pMarchingCubesGraphicsPipeline);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetCompute);
        // Here we can use cmdDraw on CPU or GPU indirect draw
        // However, indirect draw is better because cmdDraw will cause aliasing
        // Because the numVertices needs to be map from GPU and CPU, hard to synchronize
        cmdDraw(cmd, *numVertices, 0);
        cmdExecuteIndirect(cmd, INDIRECT_DRAW, 1, pIndirectDrawArgBuffer[gFrameIndex], 0, NULL, 0);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        tf_free(numVertices);
        bufBarrier.mCurrentState = RESOURCE_STATE_UNORDERED_ACCESS;
        bufBarrier.mNewState = RESOURCE_STATE_GENERIC_READ;
        
         cmdResourceBarrier(cmd, 1, &bufBarrier, 0, NULL, 0, NULL);

        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken); // Draw Skybox/Planets
        cmdBindRenderTargets(cmd, NULL);

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryDesc queryDesc = { 0 };
            cmdEndQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);

            queryDesc = { 1 };
            cmdBeginQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
        }

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw UI");

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

        barriers[0] = { pRenderTarget, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_PRESENT };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);

        cmdEndGpuFrameProfile(cmd, gGpuProfileToken);

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryDesc queryDesc = { 1 };
            cmdEndQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
            cmdResolveQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], 0, 2);
        }

        endCmd(cmd);


        FlushResourceUpdateDesc flushUpdateDesc = {};
        flushUpdateDesc.mNodeIndex = 0;
        flushResourceUpdates(&flushUpdateDesc);
        Semaphore* waitSemaphores[2] = { flushUpdateDesc.pOutSubmittedSemaphore, pImageAcquiredSemaphore };

        QueueSubmitDesc submitDesc = {};
        submitDesc.mCmdCount = 1;
        submitDesc.mSignalSemaphoreCount = 1;
        submitDesc.mWaitSemaphoreCount = TF_ARRAY_COUNT(waitSemaphores);
        submitDesc.ppCmds = &cmd;
        submitDesc.ppSignalSemaphores = &elem.pSemaphore;
        submitDesc.ppWaitSemaphores = waitSemaphores;
        submitDesc.pSignalFence = elem.pFence;
        queueSubmit(pGraphicsQueue, &submitDesc);

        QueuePresentDesc presentDesc = {};
        presentDesc.mIndex = (uint8_t)swapchainImageIndex;
        presentDesc.mWaitSemaphoreCount = 1;
        presentDesc.pSwapChain = pSwapChain;
        presentDesc.ppWaitSemaphores = &elem.pSemaphore;
        presentDesc.mSubmitDone = true;

        queuePresent(pGraphicsQueue, &presentDesc);
        flipProfiler();

 
        waitForFences(pRenderer, 1, &elem.pFence);

            
        gFrameIndex = (gFrameIndex + 1) % gDataBufferCount;
    }

    const char* GetName() { return "99b_XinjieInterviewProjectGPU"; }

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
    }

     void addComputeDescriptorSets()
    {
        DescriptorSetDesc descCompute = SRT_SET_DESC(SrtComputeData, PerFrame, gDataBufferCount, 0);
        addDescriptorSet(pRenderer, &descCompute, &pDescriptorSetCompute);
        
    }


    void removeDescriptorSets()
    {
        removeDescriptorSet(pRenderer, pDescriptorSetUniforms);
        removeDescriptorSet(pRenderer, pDescriptorSetTexture);
    }

    void removeComputeDescriptorSets()
    { 
        removeDescriptorSet(pRenderer, pDescriptorSetCompute);
    }


    void addShaders()
    {
        ShaderLoadDesc skyShader = {};
        skyShader.mVert.pFileName = "skybox.vert";
        skyShader.mFrag.pFileName = "skybox.frag";

        ShaderLoadDesc computeShaderDesc = {};
        computeShaderDesc.mComp.pFileName = "example.comp";

        ShaderLoadDesc computeShaderMC1Desc = {};
        computeShaderMC1Desc.mComp.pFileName = "marchingCubesPassOne.comp";

        ShaderLoadDesc computeShaderPrefixSumDesc = {};
        computeShaderPrefixSumDesc.mComp.pFileName = "prefixThreadSum.comp";

        ShaderLoadDesc computeShaderPrefixBlockSumDesc = {};
        computeShaderPrefixBlockSumDesc.mComp.pFileName = "prefixBlockSum.comp";
        
        ShaderLoadDesc computeShaderPrefixFinalSumDesc = {};
        computeShaderPrefixFinalSumDesc.mComp.pFileName = "prefixFinalSum.comp";

        ShaderLoadDesc computeShaderMC2Desc = {};
        computeShaderMC2Desc.mComp.pFileName = "marchingCubesPassTwo.comp";

        ShaderLoadDesc marchingCubesShaderDesc = {};
        marchingCubesShaderDesc.mVert.pFileName = "marchingCubes.vert";
        marchingCubesShaderDesc.mFrag.pFileName = "marchingCubes.frag";

        addShader(pRenderer, &skyShader, &pSkyBoxDrawShader);
    

        addShader(pRenderer, &computeShaderDesc, &pComputeShader);

        addShader(pRenderer, &computeShaderMC1Desc, &pComputeShaderMC1);
        addShader(pRenderer, &computeShaderPrefixSumDesc, &pComputeShaderPrefixSum);
        addShader(pRenderer, &computeShaderPrefixBlockSumDesc, &pComputeShaderPrefixBlockSum);
        addShader(pRenderer, &computeShaderPrefixFinalSumDesc, &pComputeShaderPrefixFinalSum);
        addShader(pRenderer, &computeShaderMC2Desc, &pComputeShaderMC2);

        addShader(pRenderer, &marchingCubesShaderDesc, &pMarchingCubesGraphicsShader);
    }

    void removeShaders()
    {
        
        removeShader(pRenderer, pSkyBoxDrawShader);

        removeShader(pRenderer, pComputeShader);

        removeShader(pRenderer, pComputeShaderMC1);
        removeShader(pRenderer, pComputeShaderPrefixSum);
        removeShader(pRenderer, pComputeShaderPrefixBlockSum);
        removeShader(pRenderer, pComputeShaderPrefixFinalSum);
        removeShader(pRenderer, pComputeShaderMC2);

        removeShader(pRenderer, pMarchingCubesGraphicsShader);
    }

    void addPipelines()
    {
        RasterizerStateDesc rasterizerStateDesc = {};
        rasterizerStateDesc.mCullMode = CULL_MODE_NONE;

        RasterizerStateDesc sphereRasterizerStateDesc = {};
        sphereRasterizerStateDesc.mCullMode = CULL_MODE_FRONT;

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
        pipelineSettings.pRasterizerState = &sphereRasterizerStateDesc;
        pipelineSettings.mVRFoveatedRendering = true;
     

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

        pipelineSettings.pDepthState = NULL;
        pipelineSettings.pRasterizerState = &rasterizerStateDesc;
        pipelineSettings.pShaderProgram = pSkyBoxDrawShader; //-V519
        addPipeline(pRenderer, &desc, &pSkyBoxDrawPipeline);
    }

    void addComputePipelines() {
        PipelineDesc pipelineDesc = {};
        pipelineDesc.mType = PIPELINE_TYPE_COMPUTE;
        ComputePipelineDesc& cpDesc = pipelineDesc.mComputeDesc;
     //   cpDesc.pShaderProgram = pComputeShader;
        cpDesc.pShaderProgram = pComputeShaderMC1;
        addPipeline(pRenderer, &pipelineDesc, &pComputePipelineMCOne);

        cpDesc.pShaderProgram = pComputeShaderPrefixSum;
        addPipeline(pRenderer, &pipelineDesc, &pPrefixThreadSumPipeline);

        cpDesc.pShaderProgram = pComputeShaderPrefixBlockSum;
        addPipeline(pRenderer, &pipelineDesc, &pPrefixBlockSumPipeline);

        cpDesc.pShaderProgram = pComputeShaderPrefixFinalSum;
        addPipeline(pRenderer, &pipelineDesc, &pPrefixFinalSumPipeline);
    
        cpDesc.pShaderProgram = pComputeShaderMC2;
        addPipeline(pRenderer, &pipelineDesc, &pComputePipelineMCTwo);
    }



    void addMarchingCubesGraphicsPipeline() {


        RasterizerStateDesc marchingCubesRasterizerStateDesc = {};
        marchingCubesRasterizerStateDesc.mCullMode = CULL_MODE_NONE;
        //marchingCubesRasterizerStateDesc.mFillMode = FILL_MODE_WIREFRAME;

        DepthStateDesc depthStateDesc = {};
        depthStateDesc.mDepthTest = true;
        depthStateDesc.mDepthWrite = true;
        depthStateDesc.mDepthFunc = CMP_GEQUAL;

        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_GRAPHICS;
        PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(SrtComputeData, Persistent), SRT_LAYOUT_DESC(SrtComputeData, PerFrame), NULL, NULL);
        GraphicsPipelineDesc& pipelineSettings = desc.mGraphicsDesc;
        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        pipelineSettings.mRenderTargetCount = 1;
        pipelineSettings.pDepthState = &depthStateDesc;
        pipelineSettings.pColorFormats = &pSwapChain->ppRenderTargets[0]->mFormat;
        pipelineSettings.mSampleCount = pSwapChain->ppRenderTargets[0]->mSampleCount;
        pipelineSettings.mSampleQuality = pSwapChain->ppRenderTargets[0]->mSampleQuality;
        pipelineSettings.mDepthStencilFormat = pDepthBuffer->mFormat;
        pipelineSettings.pShaderProgram = pMarchingCubesGraphicsShader;
      //  pipelineSettings.pVertexLayout = &gSphereVertexLayout;
        pipelineSettings.pRasterizerState = &marchingCubesRasterizerStateDesc;
       // pipelineSettings.mVRFoveatedRendering = true;
        addPipeline(pRenderer, &desc, &pMarchingCubesGraphicsPipeline);

    }

    void removePipelines()
    {
        removePipeline(pRenderer, pSkyBoxDrawPipeline);
        removePipeline(pRenderer, pComputePipelineMCOne);
        removePipeline(pRenderer, pPrefixThreadSumPipeline);
        removePipeline(pRenderer, pPrefixBlockSumPipeline);
        removePipeline(pRenderer, pPrefixFinalSumPipeline);
        removePipeline(pRenderer, pComputePipelineMCTwo);

        removePipeline(pRenderer, pMarchingCubesGraphicsPipeline);

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

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            DescriptorData uParams[1] = {};
            uParams[0].mIndex = SRT_RES_IDX(SrtData, PerFrame, gUniformBlock);
            uParams[0].ppBuffers = &pUniformBuffer[i];
            updateDescriptorSet(pRenderer, i, pDescriptorSetUniforms, 1, uParams);
        }
    }

    void prepareComputeDescriptorSet() {

         for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            DescriptorData params[12] = {};
       
            params[0].ppBuffers = &pInputBuffer[i];
            params[0].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gInputBuffer);
            params[1].ppBuffers = &pOutputBuffer[i];
            params[1].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gOutputBuffer);
          
        
            params[2].ppBuffers = &pSDFBuffer[i];
            params[2].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gSDF);
            params[3].ppBuffers = &pTriCountBuffer[i];
            params[3].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gTriangleCountBuffer);
            params[4].ppBuffers = &pPrefixSumBuffer[i];
            params[4].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gPrefixSumBuffer);
            params[5].ppBuffers = &pBlockSumBuffer[i];
            params[5].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gBlockSums);
            params[6].ppBuffers = &pScannedBlockSumBuffer[i];
            params[6].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gScannedBlockSums);
            params[7].ppBuffers = &pScratchBuffer[i];
            params[7].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gScratchBuffer);
            params[8].ppBuffers = &pTriangleBuffer[i];
            params[8].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gTriangleBuffer);
            params[9].ppBuffers = &pNumVerticesBuffer[i];
            params[9].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gNumVerticesBuffer);
            params[10].ppBuffers = &pMarchingCubesGraphicsUniformBuffer[i];
            params[10].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gMCUniformBlock);
            params[11].ppBuffers = &pIndirectDrawArgBuffer[i];
            params[11].mIndex = SRT_RES_IDX(SrtComputeData, PerFrame, gIndirectDrawArgBuffer);

            updateDescriptorSet(pRenderer, i, pDescriptorSetCompute, 12, params);
         
         }


    }

};
DEFINE_APPLICATION_MAIN(RealTimeMarchingCubes)
