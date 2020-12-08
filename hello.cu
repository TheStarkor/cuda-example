// nvcc hello.cu -o hello

#include <stdio.h>

// indicates a function that runs on device (GPU)
__global__ void cuda_hello(void) {
    printf("Hello, World!\n");
}

int main(void) {
    // <<< ... >>> syntax later
    cuda_hello<<<1, 1>>> ();
    return 0;
}