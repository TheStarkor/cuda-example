# Vector add 

## Execute nvcc
실행 결과 에러가 `.c` 파일은 에러가 발생하고 `.cu` 파일은 제대로 작동하지 않는다. 이는 CPU와 GPU가 각자의 메모리 공간을 가지고 있어서 직접 접근이 불가능하기 때문이다. 즉, GPU(device) 메모리로 CPU(host) 메모리의 값을 옮겨주어야 된다.  

따라서 CUDA 프로그램을 사용하기 위해서는 아래와 같은 과정을 거쳐야 한다.
1. Allocate host memory and initialized host data
2. Allocate device memory
3. Transfer input data from host to device memory
4. Execute kernels
5. Transfer output from device memory to host

```c
#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *out;

    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);

    // Initialize array
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f; b[i] = 2.0f;
    }

    vector_add<<<1,1>>>(out, a, b, N);
}
```

현재 코드에서는 1, 4만 진행이 되었기 때문에 제대로 작동하지 않았다. 즉 2, 3, 5 과정을 추가해 주어야 한다.

## Modify it

### Memory Management
2, 3, 5 과정을 해결하기 위해서 CUDA에서 제공하는 `cudaMalloc()`, `cudaFree()` 함수를 사용 할 수 있다.

```c
cudaMalloc(void **devPtr, size_t count);
cudaFree(void *devPtr);
```

위 두 함수를 사용하면 기존의 `malloc()`, `free()`와 유사하게 device pointer `devPtr`을 device 메모리에 alloc 및 free를 할 수 있고 이를 통해 2번 과정을 해결 할 수 있게 된다.  

### Memory Transfer
그리고 이렇게 할당한 device 메모리에 host 메모리의 data를 복사해야 한다. 이를 위해 `cudaMemcpy()`함수를 사용 할 수 있다.

```c
cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind);
```

기존의 `memcpy()`함수와 유사하고 마지막에 `cudaMemcpyKind` 파라미터를 추가로 받는다. 다양한 종류가 있지만, `cudaMemcpyHostToDevice`와 `cudaMemcpyDeviceToHost`를 사용하면 이름처럼 host to device와 device to host를 진행 할 수 있다.

### Answer
```c
// nvcc vector_add.cu -o vector_add

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);

    // Initialize array
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f; b[i] = 2.0f;
    }

    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    vector_add<<<1,1>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
```

## Time and Performance

### Time
```
$ time ./vector_add
out[0] = 3.000000
PASSED

real    0m5.150s
user    0m0.536s
sys     0m4.372s
```

### nvprof
CUDA는 `nvprof`이라는 commandline profiler tool을 제공해준다. 이를 활용해서 `time`보다 자세한 정보를 얻을 수 있다.
```
$ sudo /usr/local/cuda/bin/nvprof ./vector_add
==14683== Profiling application: ./vector_add
==14683== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   93.18%  537.54ms         1  537.54ms  537.54ms  537.54ms  vector_add(float*, float*, float*, int)
                    4.42%  25.503ms         1  25.503ms  25.503ms  25.503ms  [CUDA memcpy DtoH]
                    2.40%  13.861ms         2  6.9304ms  6.9046ms  6.9562ms  [CUDA memcpy HtoD]
      API calls:   64.22%  578.02ms         3  192.67ms  7.0590ms  563.82ms  cudaMemcpy
                   34.82%  313.38ms         3  104.46ms  239.09us  312.89ms  cudaMalloc
                    0.40%  3.5957ms         4  898.94us  890.90us  906.74us  cuDeviceTotalMem
                    0.28%  2.4860ms       344  7.2260us     285ns  281.74us  cuDeviceGetAttribute
                    0.24%  2.1979ms         3  732.64us  377.94us  916.37us  cudaFree
                    0.03%  245.86us         4  61.465us  55.803us  77.386us  cuDeviceGetName
                    0.01%  53.623us         1  53.623us  53.623us  53.623us  cudaLaunch
                    0.00%  24.965us         4  6.2410us  2.9370us  14.343us  cuDeviceGetPCIBusId
                    0.00%  9.9770us         4  2.4940us     295ns  8.4450us  cudaSetupArgument
                    0.00%  3.7810us         1  3.7810us  3.7810us  3.7810us  cudaConfigureCall
                    0.00%  3.2340us         8     404ns     255ns  1.0610us  cuDeviceGet
                    0.00%  2.1480us         3     716ns     282ns  1.5010us  cuDeviceGetCount
```

### nsys
그리고 `nvprof`보다 더 자세한 정보와 GUI를 제공하는 `nsys`라는 profiler tool을 사용 할 수 있다.
```
$ nsys profile ./vector_add
```