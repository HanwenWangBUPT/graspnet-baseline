#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#endif



int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{

    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data<float>();
    float *query_dev = query.data<float>();
    long *idx_dev = idx.data<long>();




  if (ref.type().is_cuda()) {
#ifdef WITH_CUDA
    auto stream = at::cuda::getCurrentCUDAStream();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor dist_dev_tensor = torch::empty({ref_nb * query_nb}, options);
    float *dist_dev = dist_dev_tensor.data_ptr<float>();

    for (int b = 0; b < batch; b++) {
        knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
                   dist_dev, idx_dev + b * k * query_nb, stream);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in knn: %s\n", cudaGetErrorString(err));
        AT_ERROR("aborting");
    }
    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }


    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}


