#include <cuda_runtime.h>

__global__
void CompressFileToReads(int noBlocks, char* file, int fileLettersReadBefore, int fileLength, int* compressedReads, int* holesBetweenReads, int* noHolesBetweenReads)
{
	int fileIndex = 16*(threadIdx.x + blockIdx.x * BLOCK_SIZE);
	int cReadsIndex = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	int k;	//index of letter being processed
	int currentNode;	//index of node correspoding to already processed letters' string, always multiple of 4
	bool isStringValid;
	while (fileIndex +16 < fileLength)
	{
		int mask = 3;
		int compressedString = 0;
		char letter;
		for (size_t i = 15; i >= 0; i--)
		{
			letter = file[fileIndex + i];
			if (letter == '\n')
			{
				holesBetweenReads[atomicAdd(noHolesBetweenReads, 1)] = fileLettersReadBefore + fileIndex + i;
			}
		}
		compressedReads[cReadsIndex] = 
		fileIndex += 16 * BLOCK_SIZE * noBlocks;	//go to the next k-mer
		cReadsIndex += BLOCK_SIZE * noBlocks;	//go to the next k-mer
	}
}