#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include "nv12_cuda.cuh"  // 声明函数头部

__global__ void nv12_to_bgr_kernel(const uint8_t* y_plane, const uint8_t* uv_plane,
                                  uchar3* bgr_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // 1. 读取 Y 分量（和之前一致）
    int y_val = y_plane[y * width + x];

    // 2. 计算 UV 位置（NV12 是 UV 交错存储）
    int uv_x = x / 2;              // UV 横向分辨率是 Y 的一半
    int uv_y = y / 2;              // UV 纵向分辨率是 Y 的一半
    int uv_index = uv_y * width + uv_x * 2;  // UV 数据是 U、V、U、V... 交替存储

    // 3. 读取 U 和 V（注意 NV12 的 UV 是交错存储的）
    int u_val = uv_plane[uv_index] - 128;     // U 分量
    int v_val = uv_plane[uv_index + 1] - 128; // V 分量

    // 4. YUV → RGB 转换（使用 BT.601 标准系数）
    // 注意：这里假设 Y 范围是 16-235（电视标准），U/V 是 16-240
    int y_clipped = max(0, y_val - 16);  // 限制 Y 范围
    int r = (298 * y_clipped + 409 * v_val + 128) >> 8;
    int g = (298 * y_clipped - 100 * u_val - 208 * v_val + 128) >> 8;
    int b = (298 * y_clipped + 516 * u_val + 128) >> 8;

    // 5. 限制 RGB 范围 0-255
    r = min(max(r, 0), 255);
    g = min(max(g, 0), 255);
    b = min(max(b, 0), 255);

    // 6. 存储为 BGR 格式
    bgr_image[y * width + x] = make_uchar3(b, g, r);
}


void nv12_to_bgr_gpu(const uint8_t* nv12_data, cv::Mat& dst, int width, int height) {
    uint8_t* d_y_plane;
    uint8_t* d_uv_plane;
    uchar3* d_bgr_image;

    size_t y_size = width * height;
    size_t uv_size = width * height / 2;

    cudaMalloc(&d_y_plane, y_size);
    cudaMalloc(&d_uv_plane, uv_size);
    cudaMalloc(&d_bgr_image, sizeof(uchar3) * width * height);

    cudaMemcpy(d_y_plane, nv12_data, y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uv_plane, nv12_data + y_size, uv_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    nv12_to_bgr_kernel<<<grid, block>>>(d_y_plane, d_uv_plane, d_bgr_image, width, height);
    cudaDeviceSynchronize();

    dst.create(height, width, CV_8UC3);
    cudaMemcpy(dst.data, d_bgr_image, sizeof(uchar3) * width * height, cudaMemcpyDeviceToHost);

    cudaFree(d_y_plane);
    cudaFree(d_uv_plane);
    cudaFree(d_bgr_image);
}
