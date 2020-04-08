#include <cuda_runtime.h>

__global__
void AddChunkToGraph(char* file, int length, int* out_numLines)	//TODO
{
  int thid = threadIdx.x;
  while(thid < length)
  {
    if (file[thid] == 'A')
    {
      atomicAdd(out_numLines, 1);
    }
    thid += BLOCK_SIZE;
  }
}

__global__
void AddPrecleanedChunkToGraph(char* file, int length, int* out_numLines)
{
	int thid = threadIdx.x;
	while (thid < length)
	{
		if (file[thid] == 'A')
		{
			atomicAdd(out_numLines, 1);
		}
		thid += BLOCK_SIZE;
	}
}