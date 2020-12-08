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
