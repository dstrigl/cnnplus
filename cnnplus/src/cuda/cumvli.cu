/**************************************************************************//**
 *
 * \file   cumvli.cu
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 08 2009
 *
 * $Id: cumvli.cu 3560 2010-11-22 20:47:19Z klaus $
 *
 * \brief  Implementation of cnnplus::cumvli.
 *
 *****************************************************************************/

#include "cudautils.hh"

///////////////////////////////////////////////////////////////////////////////

template<bool accumulate, size_t blockSize>
__global__ void
sumrow_kernel(float const * src, size_t const stride, float * dst,
              size_t const rows, size_t const cols)
{
    if (blockIdx.x >= rows) // !WORKAROUND! for "unspecified launch failure"
        return;             //   on a Quadro FX360M and Geforce 8600M GT

    __shared__ float sdata[blockSize];

    float const * const row = src + CNN_UIMUL(blockIdx.x, stride);
    size_t        const tid = threadIdx.x;

    // Reduce multiple elements per thread
    float tmp = 0;
    for (size_t i = tid; i < cols; i += blockSize)
        tmp += row[i];
    sdata[tid] = tmp;
    __syncthreads();

    // Do reduction in shared memory
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads();
    }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; EMUSYNC; }
    }

    // Write result to global memory
    if (tid == 0) {
        if (accumulate) dst[blockIdx.x] += sdata[0];
        else dst[blockIdx.x] = sdata[0];
    }
}

///////////////////////////////////////////////////////////////////////////////

template<size_t blockSize>
__global__ void
gemv_kernel(float const * src1, size_t const strideSrc1,
            size_t const rows, size_t const cols,
            float const * src2, float const * src3, float * dst,
            float const alpha, float const beta)
{
    if (blockIdx.x >= rows) // !WORKAROUND! for "unspecified launch failure"
        return;             //   on a Quadro FX360M and Geforce 8600M GT

    __shared__ float sdata[blockSize];

    float  const * const row = src1 + CNN_UIMUL(blockIdx.x, strideSrc1);
    size_t const         tid = threadIdx.x;

    // Reduce multiple elements per thread
    float tmp = 0;
    for (size_t i = tid; i < cols; i += blockSize)
        tmp += row[i] * src2[i];
    sdata[tid] = tmp;
    __syncthreads();

    // Do reduction in shared memory
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads();
    }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; EMUSYNC; }
    }

    // Write result to global memory
    if (tid == 0) {
        dst[blockIdx.x] = alpha * sdata[0] + beta * src3[blockIdx.x];
    }
}

///////////////////////////////////////////////////////////////////////////////

// y = alpha * Ax + beta * z
// A: m-by-n matrix, x: n elements vector, y, z: m elements vector
// m and n are arbitrary positive integers
//
// Copyright (C) 2008 Noriyuki Fujimoto, All Rights Reserved
// fujimoto@mi.s.osakafu-u.ac.jp
//
// Please refer the paper below if you use my algorithm in your published work:
//
// Noriyuki Fujimoto, Faster Matrix-Vector Multiplication on GeForce 8800GTX,
// In the Proceedings of the 22nd IEEE International Parallel and
// Distributed Processing Symposium (IPDPS), LSPP-402, pp.1-8, April 2008
//
// http://www.mi.s.osakafu-u.ac.jp/~fujimoto/CUDA/

#define bx blockIdx.x
#define tx threadIdx.x
#define ty threadIdx.y

__global__ void
mv_kernel(float const * A, size_t const strideA, size_t const m, size_t const n,
          float const * x, float const * z, float * y, float const alpha, float const beta)
{
    __shared__ float xs[16][16];
    __shared__ float Ps[16][16];

    float4 a;
    float *Psptr = (float *) Ps + (ty << 4) + tx;
    size_t const ay = (bx << 4) + ty;
    float const *Aptr = A + CNN_UIMUL(min((unsigned int)ay, (unsigned int)(m - 1)), strideA);
    float const *xptr = x + (ty << 4) + tx;
    float *xsptr = (float *) xs + (tx << 2);

    *Psptr = 0.0f;
    size_t i = 0;
    for (; i < (n & ~255); i += 256, xptr += 256) {
        xs[ty][tx] = *xptr;
        __syncthreads();
        size_t const ax = (tx << 2) + i; //= tx + (i >> 2);
        //a = tex2D(texRefA, ax     , ay);
        a = *(float4 *)(Aptr + ax      );
        *Psptr += a.x * *xsptr         + a.y * *(xsptr +   1) + a.z * *(xsptr +   2) + a.w * *(xsptr +   3);
        //a = tex2D(texRefA, ax + 16, ay);
        a = *(float4 *)(Aptr + ax +  64);
        *Psptr += a.x * *(xsptr +  64) + a.y * *(xsptr +  65) + a.z * *(xsptr +  66) + a.w * *(xsptr +  67);
        //a = tex2D(texRefA, ax + 32, ay);
        a = *(float4 *)(Aptr + ax + 128);
        *Psptr += a.x * *(xsptr + 128) + a.y * *(xsptr + 129) + a.z * *(xsptr + 130) + a.w * *(xsptr + 131);
        //a = tex2D(texRefA, ax + 48, ay);
        a = *(float4 *)(Aptr + ax + 192);
        *Psptr += a.x * *(xsptr + 192) + a.y * *(xsptr + 193) + a.z * *(xsptr + 194) + a.w * *(xsptr + 195);
        __syncthreads();
    }

    if (i + (ty << 4) + tx < n) {
        xs[ty][tx] = *xptr;
    }
    __syncthreads();
    size_t j = 0;
    for (; j < ((n - i) >> 6); j++, xsptr += 61) {
        //a = tex2D(texRefA, tx + (i >> 2) + (j << 4), ay);
        a = *(float4 *)(Aptr + (tx << 2) + i + (j << 6));
        *Psptr += a.x * *xsptr++ + a.y * *xsptr++ + a.z * *xsptr++ + a.w * *xsptr;
    }
    __syncthreads();
    size_t const remain = (n - i) & 63;
    Aptr += (tx << 2) + i + (j << 6);
    if ((tx << 2)     < remain) *Psptr += *Aptr++ * *xsptr++;
    if ((tx << 2) + 1 < remain) *Psptr += *Aptr++ * *xsptr++;
    if ((tx << 2) + 2 < remain) *Psptr += *Aptr++ * *xsptr++;
    if ((tx << 2) + 3 < remain) *Psptr += *Aptr   * *xsptr;
    //if ((tx << 2) < remain) {
    //    //a = tex2D(texRefA, tx + (i >> 2) + (j << 4), ay);
    //    a = *(float4 *)(Aptr + (tx << 2) + i + (j << 6));
    //    *Psptr += a.x * *xsptr++;
    //}
    //if ((tx << 2) + 1 < remain) *Psptr += a.y * *xsptr++;
    //if ((tx << 2) + 2 < remain) *Psptr += a.z * *xsptr++;
    //if ((tx << 2) + 3 < remain) *Psptr += a.w * *xsptr;
    __syncthreads();

    if (tx < 8) *Psptr += *(Psptr + 8);
    if (tx < 4) *Psptr += *(Psptr + 4);
    if (tx < 2) *Psptr += *(Psptr + 2);
    if (tx < 1) *Psptr += *(Psptr + 1);

    __syncthreads();
    if (ty == 0 && (bx << 4) + tx < m)
        y[(bx << 4) + tx] = alpha * Ps[tx][0] + beta * z[(bx << 4) + tx];
}

#undef bx
#undef tx
#undef ty

///////////////////////////////////////////////////////////////////////////////

__global__ void
pmulmm_kernel(float const * src1, size_t const strideSrc1,
              float const * src2, size_t const strideSrc2,
              float * dst, size_t const strideDst,
              size_t const rows, size_t const cols)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: dst = src1 .* src2
    dst[CNN_UIMUL(r, strideDst) + c] =
        src1[CNN_UIMUL(r, strideSrc1) + c] * src2[CNN_UIMUL(r, strideSrc2) + c];
}

///////////////////////////////////////////////////////////////////////////////

__global__ void
axpy_kernel(float const * src, size_t const strideSrc,
            size_t const rows, size_t const cols,
            float * srcDst, size_t const strideSrcDst,
            float const alpha)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: srcDst += alpha * src
    srcDst[CNN_UIMUL(r, strideSrcDst) + c] += alpha * src[CNN_UIMUL(r, strideSrc) + c];
}

///////////////////////////////////////////////////////////////////////////////

__global__ void
setv_kernel(float * dst, size_t const len, float const val)
{
    size_t const i = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (i >= len)
        return;

    dst[i] = val;
}

///////////////////////////////////////////////////////////////////////////////

__global__ void
setm_kernel(float * dst, size_t const stride,
            size_t const rows, size_t const cols,
            float const val)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    dst[CNN_UIMUL(r, stride) + c] = val;
}

///////////////////////////////////////////////////////////////////////////////

__global__ void
mulv_kernel(float const * src1, float const * src2, float * dst, size_t const len)
{
    size_t const i = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (i >= len)
        return;

    // Compute: dst = src1 .* src2
    dst[i] = src1[i] * src2[i];
}

///////////////////////////////////////////////////////////////////////////////

__global__ void
setcol_kernel(float * dst, size_t stride, size_t rows, size_t cols, float const * src)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    dst[r * stride + c] = src[r];
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include "common.hh"

CNNPLUS_NS_BEGIN

namespace cumvli {

void cu_sumrow(float const * src, size_t stride, float * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    sumrow_kernel<false, THREADS><<<rows, THREADS>>>(src, stride, dst, rows, cols);
    CUDA_CHECK_ERROR("Kernel call 'sumrow_kernel' failed");
}

void cu_sumrowacc(float const * src, size_t stride, float * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    sumrow_kernel<true, THREADS><<<rows, THREADS>>>(src, stride, dst, rows, cols);
    CUDA_CHECK_ERROR("Kernel call 'sumrow_kernel' failed");
}

void cu_pmulmm(float const * src1, size_t strideSrc1,
               float const * src2, size_t strideSrc2,
               float * dst, size_t strideDst,
               size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= cols);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                       (rows + dimBlock.y - 1) / dimBlock.y);
    pmulmm_kernel<<<dimGrid, dimBlock>>>
        (src1, strideSrc1, src2, strideSrc2, dst, strideDst, rows, cols);
    CUDA_CHECK_ERROR("Kernel call 'pmulmm_kernel' failed");
}

void cu_axpy(float const * src, size_t strideSrc,
             size_t rows, size_t cols,
             float * srcDst, size_t strideSrcDst,
             float alpha)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                       (rows + dimBlock.y - 1) / dimBlock.y);
    axpy_kernel<<<dimGrid, dimBlock>>>
        (src, strideSrc, rows, cols, srcDst, strideSrcDst, alpha);
    CUDA_CHECK_ERROR("Kernel call 'axpy_kernel' failed");
}

void cu_gemv(float const * src1, size_t strideSrc1,
             size_t rowsSrc1, size_t colsSrc1,
             float const * src2, size_t lenSrc2,
             float * srcDst, float alpha, float beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == colsSrc1);
    CNNPLUS_ASSERT(srcDst);
#if 1
    gemv_kernel<THREADS><<<rowsSrc1, THREADS>>>
        (src1, strideSrc1, rowsSrc1, colsSrc1, src2, srcDst, srcDst, alpha, beta);
    CUDA_CHECK_ERROR("Kernel call 'gemv_kernel' failed");
#else
    size_t const numBlk = (rowsSrc1 >> 4) + ((rowsSrc1 & 15) ? 1 : 0);
    dim3 const threads(16, 16);
    dim3 const grid(numBlk, 1);

    mv_kernel<<<grid, threads>>>
        (src1, strideSrc1, rowsSrc1, colsSrc1, src2, srcDst, srcDst, alpha, beta);
    CUDA_CHECK_ERROR("Kernel call 'mv_kernel' failed");
#endif
}

void cu_gemv(float const * src1, size_t strideSrc1,
             size_t rowsSrc1, size_t colsSrc1,
             float const * src2, size_t lenSrc2,
             float const * src3, size_t lenSrc3,
             float * dst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == colsSrc1);
    CNNPLUS_ASSERT(src3 && lenSrc3 == rowsSrc1);
    CNNPLUS_ASSERT(dst);
#if 1
    gemv_kernel<THREADS><<<rowsSrc1, THREADS>>>
        (src1, strideSrc1, rowsSrc1, colsSrc1, src2, src3, dst, 1, 1);
    CUDA_CHECK_ERROR("Kernel call 'gemv_kernel' failed");
#else
    size_t const numBlk = (rowsSrc1 >> 4) + ((rowsSrc1 & 15) ? 1 : 0);
    dim3 const threads(16, 16);
    dim3 const grid(numBlk, 1);

    mv_kernel<<<grid, threads>>>
        (src1, strideSrc1, rowsSrc1, colsSrc1, src2, src3, dst, 1, 1);
    CUDA_CHECK_ERROR("Kernel call 'mv_kernel' failed");
#endif
}

void cu_setv(float * dst, size_t len, float val)
{
    CNNPLUS_ASSERT(dst && len > 0);

    setv_kernel<<<(len + THREADS - 1) / THREADS, THREADS>>>(dst, len, val);
    CUDA_CHECK_ERROR("Kernel call 'setv_kernel' failed");
}

void cu_setm(float * dst, size_t stride, size_t rows, size_t cols, float val)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                       (rows + dimBlock.y - 1) / dimBlock.y);
    setm_kernel<<<dimGrid, dimBlock>>>(dst, stride, rows, cols, val);
    CUDA_CHECK_ERROR("Kernel call 'setm_kernel' failed");
}

void cu_mulv(float const * src1, float const * src2, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);

    mulv_kernel<<<(len + THREADS - 1) / THREADS, THREADS>>>(src1, src2, dst, len);
    CUDA_CHECK_ERROR("Kernel call 'mulv_kernel' failed");
}

void cu_setcol(float * dst, size_t stride, size_t rows, size_t cols, float const * src)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    CNNPLUS_ASSERT(src);

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                       (rows + dimBlock.y - 1) / dimBlock.y);
    setcol_kernel<<<dimGrid, dimBlock>>>(dst, stride, rows, cols, src);
    CUDA_CHECK_ERROR("Kernel call 'setcol_kernel' failed");
}

}; // namespace cumvli

CNNPLUS_NS_END
