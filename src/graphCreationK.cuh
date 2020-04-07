#include <cuda_runtime.h>

__global__
void AddFileToGraph(char* file, int length, int* out_numLines)
{
  int thid = threadIdx.x;
  while(thid < length)
  {
    if (file[thid] == '\n')
    {
      atomicAdd(out_numLines, 1);
    }
    thid += BLOCK_SIZE;
  }
}
