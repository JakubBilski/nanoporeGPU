#include <algorithm>
#include <iostream>
#include <cstring>

// for mmap:
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cuda_runtime.h>

#include "defines.cuh"
#include "utils.cuh"
#include "graphCreationK.cuh"

const char* map_file(const char* fname, size_t& length);

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
      printf("Usage: reader file_path\n");
      return 0;
    }
    size_t length;
    auto f = map_file(argv[1], length);
    auto l = f + length;
    char* d_file;
    int* d_out_numLines;
    int cudaNumLines=0;
    gpuErrchk(cudaMalloc(&d_file, length*sizeof(char)));
    gpuErrchk(cudaMemcpy(d_file, f, length*sizeof(char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_out_numLines, sizeof(int)));
    gpuErrchk(cudaMemset(&d_out_numLines, sizeof(int), 0));
    AddFileToGraph<<<1, BLOCK_SIZE>>>(d_file, length, d_out_numLines);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    gpuErrchk(cudaMemcpy(&cudaNumLines, d_out_numLines, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_file));
    gpuErrchk(cudaFree(d_out_numLines));
    uintmax_t cpuNumLines = 0;
    while (f && f!=l)
        if ((f = static_cast<const char*>(memchr(f, '\n', l-f))))
            cpuNumLines++, f++;

    std::cout << "cpuNumLines = " << cpuNumLines << "\n";
    std::cout << "cudaNumLines = " << cudaNumLines << "\n";
}

void handle_error(const char* msg) {
    perror(msg);
    exit(255);
}

const char* map_file(const char* fname, size_t& length)
{
    int fd = open(fname, O_RDONLY);
    if (fd == -1)
        handle_error("open");

    // obtain file size
    struct stat sb;
    if (fstat(fd, &sb) == -1)
        handle_error("fstat");

    length = sb.st_size;

    const char* addr = static_cast<const char*>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0u));
    if (addr == MAP_FAILED)
        handle_error("mmap");

    // TODO close fd at some point in time, call munmap(...)
    return addr;
}
