# cuda-example

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

# thread

vector_add를 진행하면서 설명을 하지 않은 <<<...>>> 표기에 대해 알아본다. CPU 대신 GPU를 사용하는 이유는 결국 GPU가 병렬 프로그래밍에 장점이 있기 때문이고 CUDA는 당연하게 이를 위한 API를 제공한다. CUDA는 threads를 "thread block"으로 관리하고 Kernel은 이러한 thread block 을 여러 개 실행 할 수 있는데 이 때 "grid"를 사용하여 관리한다. 

```<<<M,T>>>```

해당 표기법에서 `M`은 thread blocks를 `T`는 각 thread block에 속하는 thread의 수를 의미한다.

## Terminology

- `threadIdx.x`: the index of the thread within the block
- `blockIdx.x`: the index of the block within the grid
- `blockDim.x`: the size of thread block (number of threads in the thread block)
- `gridDim.x`: the size of the grid

## Using Multithread

`vector_add<<<1,256>>>(d_out, d_a, d_b, N);`으로 함수 실행 코드가 바뀐 것을 `vector_add_thread.cu` 파일에서 확인 할 수 있다. 그러면 `threadIdx.x`는 0~255의 값을 가지게 되고 `blockDim.x`는 256이 된다. 이 값을 활용해서 병렬 연산으로 바꾸는 과정은 아래와 같다.

- **thread 별로 N/thread 개의 연산을 처리한다.**

```c
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}
```

### Performance
```
$ time ./vector_add_thread
real    0m4.617s
user    0m0.240s
sys     0m4.144s
```

```
$ nvprof ./vector_add_thread
==18852== Profiling application: ./vector_add_thread
==18852== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.17%  32.451ms         1  32.451ms  32.451ms  32.451ms  [CUDA memcpy DtoH]
                   24.20%  13.979ms         1  13.979ms  13.979ms  13.979ms  vector_add(float*, float*, float*, int)
                   19.63%  11.340ms         2  5.6700ms  5.5543ms  5.7856ms  [CUDA memcpy HtoD]
      API calls:   83.01%  333.71ms         3  111.24ms  250.49us  333.21ms  cudaMalloc
                   14.68%  59.020ms         3  19.673ms  5.6930ms  47.426ms  cudaMemcpy
                    0.88%  3.5367ms         4  884.18us  828.12us  923.52us  cuDeviceTotalMem
                    0.72%  2.9063ms         3  968.76us  407.00us  1.2543ms  cudaFree
                    0.62%  2.4858ms       344  7.2260us     265ns  285.39us  cuDeviceGetAttribute
                    0.06%  253.76us         4  63.440us  55.076us  85.427us  cuDeviceGetName
                    0.01%  53.461us         1  53.461us  53.461us  53.461us  cudaLaunch
                    0.01%  22.031us         4  5.5070us  2.5320us  11.786us  cuDeviceGetPCIBusId
                    0.00%  9.2370us         4  2.3090us     352ns  7.4060us  cudaSetupArgument
                    0.00%  3.6840us         3  1.2280us     322ns  1.7220us  cuDeviceGetCount
                    0.00%  3.6340us         8     454ns     254ns  1.1040us  cuDeviceGet
                    0.00%  2.9640us         1  2.9640us  2.9640us  2.9640us  cudaConfigureCall
```

## Using Multiprocess

위 과정에서 우리는 multithread를 사용하여 연산을 진행하였다. 맨 처음 언급했듯 CUDA는 "grid"를 통해 이러한 "thread block"을 여러 개 관리 할 수 있으므로 이제 thread block의 개수를 늘려본다. 즉 앞의 예제에서는 `BlockIdx.x`가 0 하나였다면 이제는 이 값이 늘어나는 것이다.

- **모든 연산을 thread에 하나씩 할당하고 병렬적으로 진행한다.**

### modify call operation

```c
int block_size = 256;
int grid_size = ((N + block_size) / block_size);
vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
```

### modify vector_add operation

```c
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}
```

## Performance
```
$ time ./vector_add_grid
real    0m4.482s
user    0m0.204s
sys     0m4.032s
```

```
$ nvprof ./vector_add_grid
==19103== Profiling application: ./vector_add_grid
==19103== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   71.37%  32.575ms         1  32.575ms  32.575ms  32.575ms  [CUDA memcpy DtoH]
                   28.17%  12.858ms         2  6.4288ms  6.4212ms  6.4364ms  [CUDA memcpy HtoD]
                    0.47%  212.77us         1  212.77us  212.77us  212.77us  vector_add(float*, float*, float*, int)
      API calls:   84.44%  301.93ms         3  100.64ms  242.88us  301.44ms  cudaMalloc
                   13.14%  46.969ms         3  15.656ms  6.5576ms  33.777ms  cudaMemcpy
                    0.86%  3.0921ms         4  773.03us  768.73us  779.09us  cuDeviceTotalMem
                    0.83%  2.9518ms         3  983.94us  409.99us  1.3025ms  cudaFree
                    0.64%  2.2853ms       344  6.6430us     249ns  264.27us  cuDeviceGetAttribute
                    0.06%  219.72us         4  54.928us  50.543us  67.722us  cuDeviceGetName
                    0.02%  54.158us         1  54.158us  54.158us  54.158us  cudaLaunch
                    0.01%  19.444us         4  4.8610us  1.5700us  12.783us  cuDeviceGetPCIBusId
                    0.00%  10.926us         8  1.3650us     223ns  8.7940us  cuDeviceGet
                    0.00%  10.831us         4  2.7070us     379ns  8.8550us  cudaSetupArgument
                    0.00%  3.5030us         1  3.5030us  3.5030us  3.5030us  cudaConfigureCall
                    0.00%  1.9570us         3     652ns     240ns  1.0850us  cuDeviceGetCount
```

## 결론
CUDA를 통해 쉽게 GPU 병렬 계산을 진행 할 수 있다. 그리고 이 결과 아래 표에서 확인 할 수 있듯이 상당한 성능 향상이 이루어졌다.

| Version | Execution Time(ms) | Speedup |
|---|:---:|---:|
| 1 thread | 192.67 | 1.00x |
| 1 block | 13.98 | 13.78x |
| Multiple bloocks | 0.21 | 917.48x |
