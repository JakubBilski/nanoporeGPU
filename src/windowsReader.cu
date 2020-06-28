#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include <time.h>

#include <cuda_runtime.h>

#include "defines.cuh"
#include "utils.cuh"
#include "graphCreationK.cuh"
#include "weakLeavesDeletionK.cuh"
#include "debugTools.cuh"
#include "tempOperations.cuh"

#include "defines.h"
#include "Correct.h"

template <int TNoBlocks> int* precleanedJumpGPU(std::fstream& fs, std::fstream& ts);

int main(int argc, char* argv[])
{
#ifdef DEBUG
	TestCorrect();
	return 0;
#else
    if(argc != 2 && argc != 3)
    {
      printf("Usage: reader file_path temp_file_path\n");
      return 0;
    }
	char* tempFilePath = "./tempFile.txt";
	if (argc == 3)
	{
		tempFilePath = argv[2];
	}
	char* inputFilePath = argv[1];
	printf("Machine:\n\t%d MB process memory\n", (sizeof(char)*HOST_CHUNK_SIZE) / (1024 * 1024));
	printf("\t%d MB device data memory\n", (sizeof(char)*DEVICE_CHUNK_SIZE) / (1024 * 1024));
	printf("\t%d MB device tree memory\n", (sizeof(int)*DEVICE_TREE_SIZE) / (1024 * 1024));
	printf("Starting\n");

	const int noTests = 1;
	for (size_t test = 0; test < noTests; test++)
	{
		printf(inputFilePath);
		printf(", %d-mers, run %d\n", MER_LENGHT, (int)test);

		std::fstream fs(inputFilePath, std::ios::in | std::ios::binary);
		assertOpenFile(fs, inputFilePath);
		std::fstream ts(tempFilePath, std::ios::in | std::ios::out | std::ios::binary | std::ofstream::trunc);
		assertOpenFile(ts, tempFilePath);
		clock_t start = clock();

		int* DBG = precleanedJumpGPU<10>(fs, ts);
		printf("%25s = %11f\n", "precleanedJumpGPU<10>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();
		ts.close();

		char* tempFilePath2 = "result_file_aa.txt";
		fs.open(tempFilePath, std::ios::in | std::ios::binary);
		assertOpenFile(fs, tempFilePath);
		ts.open(tempFilePath2, std::ios::in | std::ios::out | std::ios::binary | std::ofstream::trunc);
		assertOpenFile(ts, tempFilePath2);
		start = clock();

		Correct(fs, ts, DBG);
		printf("%25s = %11f\n", "precleanedJumpGPU<10>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();
		ts.close();

		printf("\n");
	}
	return 0;
#endif
}

template <int TNoBlocks>
int* precleanedJumpGPU(std::fstream& fs, std::fstream& ts)
{
	char* d_chunk;
	int* d_tree;
	int* d_treeLength;
	char* chunk = (char*)malloc(sizeof(char)*HOST_CHUNK_SIZE);
	char* clearedChunk = (char*)malloc(sizeof(char)*DEVICE_CHUNK_SIZE);
	int clearedChunkSize = 0;
	int chunkOffset = 0;
	int cutPhase = 2;
	gpuErrchk(cudaMalloc(&d_chunk, DEVICE_CHUNK_SIZE * sizeof(char)));
	gpuErrchk(cudaMalloc(&d_tree, DEVICE_TREE_SIZE * sizeof(int)));
	gpuErrchk(cudaMemset(d_tree, 0, DEVICE_TREE_SIZE * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_treeLength, sizeof(int)));
	const int startingTreeLength = 4;
	gpuErrchk(cudaMemcpy(d_treeLength, &startingTreeLength, sizeof(int), cudaMemcpyHostToDevice));
	int lettersLength;
	while (!fs.eof())
	{
		fs.read(chunk + chunkOffset, HOST_CHUNK_SIZE - chunkOffset);
		int chunkLength = chunkOffset + fs.gcount();
		int i = 0;
		for (int j = 0; j < 3 - cutPhase; j++)
		{
			while (chunk[i] != '\n')
			{
				i++;
			}
			i++;
		}
		bool endOfChunk = false;
		while (!endOfChunk)
		{
			int startOfLetters = i;
			while (chunk[i] != '\n')
			{
				i++;
				if (clearedChunkSize + i - startOfLetters == DEVICE_CHUNK_SIZE)
				{
					gpuErrchk(cudaMemcpy(d_chunk, clearedChunk, clearedChunkSize * sizeof(char), cudaMemcpyHostToDevice));
					AddPrecleanedChunkToGraph<MER_LENGHT> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_tree, d_treeLength);
					AddPrecleanedChunkToTemp(ts, clearedChunk, clearedChunkSize);
					kernelErrchk();
					clearedChunkSize = 0;
				}
				if (i == chunkLength)
				{
					memcpy(chunk, chunk + startOfLetters, sizeof(char)*(chunkLength - startOfLetters));
					chunkOffset = chunkLength - startOfLetters;
					cutPhase = 3;
					endOfChunk = true;
					break;
				}
			}
			i++;
			if (!endOfChunk)
			{
				lettersLength = i - startOfLetters;
				memcpy(clearedChunk + clearedChunkSize, chunk + startOfLetters, lettersLength);
				clearedChunkSize += lettersLength;
				if (i + 2 >= chunkLength)
				{
					cutPhase = 0;
					break;
				}
				if (i + 2 + lettersLength >= chunkLength)
				{
					cutPhase = 1;
					break;
				}
				i += 2 + lettersLength;
				while (chunk[i] != '\n')
				{
					i++;
					if (i == chunkLength)
					{
						cutPhase = 2;
						endOfChunk = true;
						break;
					}
				}
				i++;
				if (i == chunkLength)
				{
					cutPhase = 3;
					chunkOffset = 0;
					break;
				}
			}
		}
	}
	gpuErrchk(cudaMemcpy(d_chunk, clearedChunk, clearedChunkSize * sizeof(char), cudaMemcpyHostToDevice));
	AddPrecleanedChunkToGraph<MER_LENGHT> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_tree, d_treeLength);
	kernelErrchk();
	DeleteWeakLeaves<MER_LENGHT> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_tree);
	kernelErrchk();
	int finalTreeLength = 0;
	gpuErrchk(cudaMemcpy(&finalTreeLength, d_treeLength, sizeof(int), cudaMemcpyDeviceToHost));
	int* finalTree = (int*)malloc(sizeof(int)*finalTreeLength);
	gpuErrchk(cudaMemcpy(finalTree, d_tree, finalTreeLength * sizeof(int), cudaMemcpyDeviceToHost));
	//DisplaySizeInfo(finalTreeLength, MER_LENGHT);
	DisplayTree(finalTree);
	//DisplayTable(finalTree, finalTreeLength);
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_tree));
	gpuErrchk(cudaFree(d_treeLength));
	free(chunk);
	free(clearedChunk);
	return finalTree;
}