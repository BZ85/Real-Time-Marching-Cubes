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

#pragma once


// for low end iOS devices, do not use Argument buffers
BEGIN_SRT_NO_AB(SrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_TEXTURE(Persistent, Tex2D(float4), gRightTexture)
        DECL_TEXTURE(Persistent, Tex2D(float4), gLeftTexture)
        DECL_TEXTURE(Persistent, Tex2D(float4), gTopTexture)
        DECL_TEXTURE(Persistent, Tex2D(float4), gBotTexture)
        DECL_TEXTURE(Persistent, Tex2D(float4), gFrontTexture)
        DECL_TEXTURE(Persistent, Tex2D(float4), gBackTexture)
        DECL_SAMPLER(Persistent, SamplerState, gSampler)

    END_SRT_SET(Persistent)

    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(UniformData), gUniformBlock)
     //   DECL_RWBUFFER(PerFrame, RWBuffer<float>, gInputBuffer)
    //    DECL_RWBUFFER(PerFrame, RWBuffer<float>, gOutputBuffer)
    END_SRT_SET(PerFrame)
END_SRT(SrtData)


BEGIN_SRT_NO_AB(SrtComputeData)
BEGIN_SRT_SET(PerFrame)
  DECL_RWBUFFER(PerFrame, RWBuffer<float>, gInputBuffer)
  DECL_RWBUFFER(PerFrame, RWBuffer<float>, gOutputBuffer)
  DECL_RWBUFFER(PerFrame, RWBuffer<float>, gSDF)
  DECL_RWBUFFER(PerFrame, RWBuffer<uint>, gTriangleCountBuffer)
  DECL_RWBUFFER(PerFrame, RWBuffer<uint>, gPrefixSumBuffer)
  DECL_RWBUFFER(PerFrame, RWBuffer<uint>, gBlockSums)
  DECL_RWBUFFER(PerFrame, RWBuffer<uint>, gScannedBlockSums)
  DECL_RWBUFFER(PerFrame, RWBuffer<uint>, gScratchBuffer)
  DECL_RWBUFFER(PerFrame, RWBuffer<float>, gTriangleBuffer)
  DECL_RWBUFFER(PerFrame, RWBuffer<uint>, gNumVerticesBuffer)
  DECL_CBUFFER(PerFrame, CBUFFER(MCUniformData), gMCUniformBlock)
  DECL_RWBUFFER(PerFrame, RWBuffer<uint>, gIndirectDrawArgBuffer)
 END_SRT_SET(PerFrame)

END_SRT(SrtComputeData)
