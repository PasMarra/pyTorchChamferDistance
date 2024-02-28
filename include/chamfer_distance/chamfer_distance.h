//#ifndef CHAMFER_DISTANCE_HPP
//#define CHAMFER_DISTANCE_HPP
//
#include <torch/torch.h>

// CUDA forward declarations

void ChamferDistanceKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const int m,
    const float* xyz2,
    float* result,
    int* result_i,
    float* result2,
    int* result2_i);


void ChamferDistanceGradKernelLauncher(
    const int b, const int n,
    const float* xyz1,
    const int m,
    const float* xyz2,
    const float* grad_dist1,
    const int* idx1,
    const float* grad_dist2,
    const int* idx2,
    float* grad_xyz1,
    float* grad_xyz2);


void chamfer_distance_forward_cuda(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2);


void chamfer_distance_backward_cuda(
    const at::Tensor xyz1,
    const at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2);


void nnsearch(
    const int b, const int n, const int m,
    const float* xyz1,
    const float* xyz2,
    float* dist,
    int* idx);


void chamfer_distance_forward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2);


void chamfer_distance_backward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2);

//#endif
