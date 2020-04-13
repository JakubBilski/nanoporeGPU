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
#include "debugTools.cuh"

template <int TNoBlocks> void precleanedGPU(std::ifstream& fs);
template <int TNoBlocks> void precleanedJumpGPU(std::ifstream& fs);
void simpleCPU(std::ifstream& fs);
void jumpCPU(std::ifstream& fs);

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
      printf("Usage: reader file_path\n");
      return 0;
    }

	printf("Machine:\n\t%d MB process memory\n", (sizeof(char)*HOST_CHUNK_SIZE) / (1024 * 1024));
	printf("\t%d MB device data memory\n", (sizeof(char)*DEVICE_CHUNK_SIZE) / (1024 * 1024));
	printf("\t%d MB device tree memory\n", (sizeof(int)*DEVICE_TREE_SIZE) / (1024 * 1024));
	printf("Starting\n");

	const int noTests = 1;
	for (size_t test = 0; test < noTests; test++)
	{
		printf(argv[1]);
		printf(", run %d\n", (int)test);

		std::ifstream fs(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		clock_t start = clock();
		precleanedGPU<5>(fs);
		printf("%25s = %11f\n", "precleanedGPU<5>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedGPU<5>(fs);
		//printf("%25s = %11f\n", "precleanedGPU<5>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedGPU<10>(fs);
		//printf("%25s = %11f\n", "precleanedGPU<10>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedGPU<20>(fs);
		//printf("%25s = %11f\n", "precleanedGPU<20>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedJumpGPU<1>(fs);
		//printf("%25s = %11f\n", "precleanedJumpGPU<1>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedJumpGPU<5>(fs);
		//printf("%25s = %11f\n", "precleanedJumpGPU<5>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedJumpGPU<10>(fs);
		//printf("%25s = %11f\n", "precleanedJumpGPU<10>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedJumpGPU<20>(fs);
		//printf("%25s = %11f\n", "precleanedJumpGPU<20>", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//simpleCPU(fs);
		//printf("%25s = %11f\n", "simpleCPU", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//jumpCPU(fs);
		//printf("%25s = %11f\n", "jumpCPU", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		printf("\n");
	}
	return 0;
}

template <int TNoBlocks>
void precleanedGPU(std::ifstream& fs)
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
				memcpy(clearedChunk + clearedChunkSize, chunk + startOfLetters, i - startOfLetters);
				clearedChunkSize += i - startOfLetters;
			}
			if (!endOfChunk)
			{
				for (int j = 0; j < 3 && !endOfChunk; j++)
				{
					if (i == chunkLength)
					{
						cutPhase = j;
						break;
					}
					while (chunk[i] != '\n')
					{
						i++;
						if (i == chunkLength)
						{
							cutPhase = j;
							endOfChunk = true;
							break;
						}
					}
					i++;
				}
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
	int finalTreeLength = 0;
	gpuErrchk(cudaMemcpy(&finalTreeLength, d_treeLength, sizeof(int), cudaMemcpyDeviceToHost));
	int* finalTree = (int*)malloc(sizeof(int)*finalTreeLength);
	gpuErrchk(cudaMemcpy(finalTree, d_tree, finalTreeLength*sizeof(int), cudaMemcpyDeviceToHost));
	printf("Treelength: %d\n", finalTreeLength);
	//DisplayTree(finalTree);
	//DisplayTable(finalTree, finalTreeLength);
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_tree));
	gpuErrchk(cudaFree(d_treeLength));
	free(chunk);
	free(clearedChunk);
}

template <int TNoBlocks>
void precleanedJumpGPU(std::ifstream& fs)
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
	int finalTreeLength = 0;
	gpuErrchk(cudaMemcpy(&finalTreeLength, d_treeLength, sizeof(int), cudaMemcpyDeviceToHost));
	int* finalTree = (int*)malloc(sizeof(int)*finalTreeLength);
	gpuErrchk(cudaMemcpy(finalTree, d_tree, finalTreeLength * sizeof(int), cudaMemcpyDeviceToHost));
	DisplayTree(finalTree);
	//DisplayTable(finalTree, finalTreeLength);
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_tree));
	gpuErrchk(cudaFree(d_treeLength));
	free(chunk);
	free(clearedChunk);
}

void jumpCPU(std::ifstream& fs)
{
	int noAs = 0;
	char* chunk = (char*)malloc(sizeof(char)*HOST_CHUNK_SIZE);
	int cutPhase = 2;
	int noSequenceLetters = 0;
	int localStartOfLetters;
	while (!fs.eof())
	{
		fs.read(chunk, HOST_CHUNK_SIZE);
		int i = 0;
		for (int j=0; j < 3-cutPhase; j++)
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
			localStartOfLetters = i;
			while (chunk[i] != '\n')
			{
				if (chunk[i] == 'A')
				{
					noAs++;
				}
				i++;
				if (i == HOST_CHUNK_SIZE)
				{
					//w jakis sposob musimy tu zapamietac to, co bedzie potrzebne do dalszego liczenia w kolejnym kawalku
					//moze to zapewne zajac troche czasu
					//ponizsze zakomentowane jest bledne
					//memcpy(chunk, chunk + startOfLetters, sizeof(char)*(INPUT_CHUNK_SIZE - startOfLetters));
					//chunkOffset = INPUT_CHUNK_SIZE - startOfLetters;
					cutPhase = 3;
					endOfChunk = true;
					noSequenceLetters = i - localStartOfLetters;
					break;
				}
			}
			if (!endOfChunk)
			{
				i++;
				if (i == HOST_CHUNK_SIZE)
				{
					cutPhase = 0;
					noSequenceLetters = 0;
					break;
				}
				noSequenceLetters += i - localStartOfLetters;
				if (i + 2 >= HOST_CHUNK_SIZE)
				{
					cutPhase = 0;
					noSequenceLetters = 0;
					break;
				}
				if (i + 2 + noSequenceLetters >= HOST_CHUNK_SIZE)
				{
					cutPhase = 1;
					noSequenceLetters = 0;
					break;
				}
				i += 2 + noSequenceLetters;
				while (chunk[i] != '\n')
				{
					i++;
					if (i == HOST_CHUNK_SIZE)
					{
						cutPhase = 2;
						endOfChunk = true;
						noSequenceLetters = 0;
						break;
					}
				}
				i++;
				noSequenceLetters = 0;
				if (i == HOST_CHUNK_SIZE)
				{
					cutPhase = 3;
					break;
				}
			}
		}
	}
	free(chunk);
	if (noAs != DEBUG_A_COUNT)
	{
		throw std::runtime_error("invalid value calculated by tested function");
	}
}

void simpleCPU(std::ifstream& fs)
{
	int noAs = 0;
	char* chunk = (char*)malloc(sizeof(char)*HOST_CHUNK_SIZE);
	int cutPhase = 2;
	while (!fs.eof())
	{
		fs.read(chunk, HOST_CHUNK_SIZE);
		int i = 0;
		for (int j=0; j < 3-cutPhase; j++)
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
			while (chunk[i] != '\n')
			{
				if (chunk[i] == 'A')
				{
					noAs++;
				}
				i++;
				if (i == HOST_CHUNK_SIZE)
				{
					//w jakis sposob musimy tu zapamietac to, co bedzie potrzebne do dalszego liczenia w kolejnym kawalku
					//moze to zapewne zajac troche czasu
					//ponizsze zakomentowane jest bledne
					//memcpy(chunk, chunk + startOfLetters, sizeof(char)*(INPUT_CHUNK_SIZE - startOfLetters));
					//chunkOffset = INPUT_CHUNK_SIZE - startOfLetters;
					cutPhase = 3;
					endOfChunk = true;
					break;
				}
			}
			i++;
			if (i == HOST_CHUNK_SIZE)
			{
				cutPhase = 0;
				break;
			}
			if (!endOfChunk)
			{
				for (int j = 0; j < 3 && !endOfChunk; j++)
				{
					while (chunk[i] != '\n')
					{
						i++;
						if (i == HOST_CHUNK_SIZE)
						{
							cutPhase = j;
							endOfChunk = true;
							break;
						}
					}
					i++;
					if (i == HOST_CHUNK_SIZE)
					{
						cutPhase = j+1;
						endOfChunk = true;
						break;
					}
				}
			}
		}
	}
	free(chunk);
	if (noAs != DEBUG_A_COUNT)
	{
		throw std::runtime_error("invalid value calculated by tested function");
	}
}