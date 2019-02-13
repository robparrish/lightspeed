make
/global/software/CUDA/8.0.44/bin/nvcc -Xcompiler -rdynamic -lineinfo -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 --ptxas-options=-v --std=c++11 -O3 --compiler-options '-fPIC -O3' -I/global/user_software/lightspeed/1.01/build/include -c gpu_aiscat.cu
make


# /global/software/CUDA/8.0.44/bin/nvcc -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 --ptxas-options=-v --std=c++11 -O3 --compiler-options '-fPIC -O3' -I/global/user_software/lightspeed/1.01/build/include -c gpu_aiscat.cu
# /global/software/CUDA/9.0.103/bin/nvcc -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 --ptxas-options=-v --std=c++11 -O3 --compiler-options '-fPIC -O3' -I/global/user_software/lightspeed/1.01/build/include -c gpu_aiscat.cu
