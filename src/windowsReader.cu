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

template <int TNoBlocks, int merLength>
int* precleanedToGraph(std::fstream& fs, int& out_treeLength);

template <int TNoBlocks, int merLength>
int* fastqToGraphAndPrecleaned(std::fstream& fs, std::fstream& ts, int& out_treeLength);

void performRun(char* inputFilePath, char* resultFile1, char* resultFile2, char* resultFile3, char* resultFile4);

int main(int argc, char* argv[])
{
#ifdef DEBUG
	TestCorrect();
	return 0;
#else
    if(argc != 6)
    {
      printf("Usage: reader file_path result_file_1 result_file_2 result_file_3 result_file_4\n");
      return 0;
    }
	char* inputFilePath = argv[1];
	char* resultFile1 = argv[2];
	char* resultFile2 = argv[3];
	char* resultFile3 = argv[4];
	char* resultFile4 = argv[5];
	printf("Machine:\n\t%d MB process memory\n", (sizeof(char)*HOST_CHUNK_SIZE) / (1024 * 1024));
	printf("\t%d MB device data memory\n", (sizeof(char)*DEVICE_CHUNK_SIZE) / (1024 * 1024));
	printf("\t%d MB device tree memory\n", (sizeof(int)*DEVICE_TREE_SIZE) / (1024 * 1024));
	printf("Starting\n");

	const int noTests = 1;
	for (size_t test = 0; test < noTests; test++)
	{
		printf("\n\nRun %d\n", (int)test);
		performRun(inputFilePath, resultFile1, resultFile2, resultFile3, resultFile4);
	}
	return 0;
#endif
}

void performRun(char* inputFilePath, char* resultFile1, char* resultFile2, char* resultFile3, char* resultFile4)
{
	std::fstream fs(inputFilePath, std::ios::in | std::ios::binary);
	assertOpenFile(fs, inputFilePath);
	std::fstream ts(resultFile1, std::ios::in | std::ios::out | std::ios::binary | std::ofstream::trunc);
	assertOpenFile(ts, resultFile1);
	clock_t start = clock();

	int DBG_size = 0;
	int* DBG = fastqToGraphAndPrecleaned<NO_BLOCKS, MER_LENGTH_1>(fs, ts, DBG_size);
	printf("%25s = %11f\n", "fastqToGraphAndPrecleaned", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
	fs.close();
	ts.close();

	fs.open(resultFile1, std::ios::in | std::ios::binary);
	assertOpenFile(fs, resultFile1);
	ts.open(resultFile2, std::ios::in | std::ios::out | std::ios::binary | std::ofstream::trunc);
	assertOpenFile(ts, resultFile2);
	start = clock();

	std::vector<int> DBG_v(DBG, DBG + DBG_size);
	Correct(fs, ts, DBG_v, MER_LENGTH_1);
	printf("%25s = %11f\n", "Correct", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
	fs.close();
	ts.close();

	free(DBG);

	fs.open(resultFile2, std::ios::in | std::ios::binary);
	assertOpenFile(fs, resultFile2);
	start = clock();

	DBG = precleanedToGraph<NO_BLOCKS, MER_LENGTH_2>(fs, DBG_size);
	printf("%25s = %11f\n", "precleanedToGraph", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
	fs.close();

	fs.open(resultFile2, std::ios::in | std::ios::binary);
	assertOpenFile(fs, resultFile2);
	ts.open(resultFile3, std::ios::in | std::ios::out | std::ios::binary | std::ofstream::trunc);
	assertOpenFile(ts, resultFile3);
	start = clock();

	DBG_v = std::vector<int>(DBG, DBG + DBG_size);
	Correct(fs, ts, DBG_v, MER_LENGTH_2);
	printf("%25s = %11f\n", "Correct", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
	fs.close();
	ts.close();

	free(DBG);

	fs.open(resultFile3, std::ios::in | std::ios::binary);
	assertOpenFile(fs, resultFile3);
	start = clock();

	DBG = precleanedToGraph<NO_BLOCKS, MER_LENGTH_3>(fs, DBG_size);
	printf("%25s = %11f\n", "precleanedToGraph", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
	fs.close();

	fs.open(resultFile3, std::ios::in | std::ios::binary);
	assertOpenFile(fs, resultFile3);
	ts.open(resultFile4, std::ios::in | std::ios::out | std::ios::binary | std::ofstream::trunc);
	assertOpenFile(ts, resultFile4);
	start = clock();

	DBG_v = std::vector<int>(DBG, DBG + DBG_size);
	Correct(fs, ts, DBG_v, MER_LENGTH_3);
	printf("%25s = %11f\n", "Correct", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
	fs.close();
	ts.close();

	printf("\n");
}

template <int TNoBlocks, int merLength>
int* precleanedToGraph(std::fstream& fs, int& out_treeLength)
{
	char* d_chunk;
	int* d_tree;
	int* d_treeLength;
	char* chunk = (char*)malloc(sizeof(char)*HOST_CHUNK_SIZE);
	gpuErrchk(cudaMalloc(&d_chunk, DEVICE_CHUNK_SIZE * sizeof(char)));
	gpuErrchk(cudaMalloc(&d_tree, DEVICE_TREE_SIZE * sizeof(int)));
	gpuErrchk(cudaMemset(d_tree, 0, DEVICE_TREE_SIZE * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_treeLength, sizeof(int)));
	const int startingTreeLength = 4;
	gpuErrchk(cudaMemcpy(d_treeLength, &startingTreeLength, sizeof(int), cudaMemcpyHostToDevice));
	int chunkOffset = 0;
	int lastNewline = 0;
	int loadedToGPU = 0;
	fs.read(chunk + chunkOffset, HOST_CHUNK_SIZE - chunkOffset);
	int chunkLength = chunkOffset + fs.gcount();
	bool readEnded = false;
	while (!readEnded)
	{
		//w gpu zmiesci sie wiecej niz jest w buforze
		if (chunkLength - 1 < lastNewline + (DEVICE_CHUNK_SIZE - loadedToGPU))
		{
			int i = chunkLength - 1;
			while (chunk[i] != '\n')
			{
				i--;
			}
			//skopiuj dane do gpu, ale nie odpalaj kernela
			gpuErrchk(cudaMemcpy(d_chunk + loadedToGPU, chunk + lastNewline, (i - lastNewline) * sizeof(char), cudaMemcpyHostToDevice));
			loadedToGPU += (i - lastNewline);

			//skopiuj ostatni urwany read w buforze na poczatek bufora
			memcpy(chunk, chunk + i, sizeof(char)*(chunkLength - i));
			chunkOffset = chunkLength - i;

			if (!fs.eof())
			{
				//wczytaj z pliku az do wypelnienia bufora
				fs.read(chunk + chunkOffset, HOST_CHUNK_SIZE - chunkOffset);
				chunkLength = chunkOffset + fs.gcount();
				lastNewline = 0;
			}
			else
			{
				//plik sie skonczyl, odpalamy kernel pomimo niewypelnienia do konca pamieci gpu
				AddPrecleanedChunkToGraph<merLength> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, loadedToGPU, d_tree, d_treeLength);
				kernelErrchk();
				readEnded = true;
			}
		}
		else  //w gpu zmiesci sie mniej niz jest w buforze
		{
			int i = lastNewline + (DEVICE_CHUNK_SIZE - loadedToGPU);
			while (chunk[i] != '\n')
			{
				i--;
			}
			//skopiuj ile sie zmiesci do gpu i odpal kernel
			gpuErrchk(cudaMemcpy(d_chunk + loadedToGPU, chunk + lastNewline, (i - lastNewline) * sizeof(char), cudaMemcpyHostToDevice));
			AddPrecleanedChunkToGraph<merLength> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, loadedToGPU + i - lastNewline, d_tree, d_treeLength);
			kernelErrchk();
			loadedToGPU = 0;
			lastNewline = i;
		}
	}
	int* d_noDeletedDebug;
	gpuErrchk(cudaMalloc(&d_noDeletedDebug, sizeof(int)));
	gpuErrchk(cudaMemset(d_noDeletedDebug, 0, sizeof(int)));
	DeleteWeakLeaves<merLength> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_tree, d_noDeletedDebug);
	kernelErrchk();

	int finalTreeLength = 0;
	gpuErrchk(cudaMemcpy(&finalTreeLength, d_treeLength, sizeof(int), cudaMemcpyDeviceToHost));
	int* finalTree = (int*)malloc(sizeof(int)*finalTreeLength);
	gpuErrchk(cudaMemcpy(finalTree, d_tree, finalTreeLength * sizeof(int), cudaMemcpyDeviceToHost));
	int noDeletedDebug = 0;
	gpuErrchk(cudaMemcpy(&noDeletedDebug, d_noDeletedDebug, sizeof(int), cudaMemcpyDeviceToHost));
	printf("Deleted %d weak %d-mers\n", noDeletedDebug, merLength);

	DisplaySizeInfo(finalTreeLength, merLength);
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_tree));
	gpuErrchk(cudaFree(d_treeLength));
	free(chunk);
	out_treeLength = finalTreeLength;
	return finalTree;
}

template <int TNoBlocks, int merLength>
int* fastqToGraphAndPrecleaned(std::fstream& fs, std::fstream& ts, int& out_treeLength)
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
					AddPrecleanedChunkToGraph<merLength> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_tree, d_treeLength);
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
	AddPrecleanedChunkToGraph<merLength> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_tree, d_treeLength);
	kernelErrchk();
	int* d_noDeletedDebug;
	gpuErrchk(cudaMalloc(&d_noDeletedDebug, sizeof(int)));
	gpuErrchk(cudaMemset(d_noDeletedDebug, 0, sizeof(int)));
	DeleteWeakLeaves<merLength> << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_tree, d_noDeletedDebug);
	kernelErrchk();
	int finalTreeLength = 0;
	gpuErrchk(cudaMemcpy(&finalTreeLength, d_treeLength, sizeof(int), cudaMemcpyDeviceToHost));
	int* finalTree = (int*)malloc(sizeof(int)*finalTreeLength);
	gpuErrchk(cudaMemcpy(finalTree, d_tree, finalTreeLength * sizeof(int), cudaMemcpyDeviceToHost));
	int noDeletedDebug = 0;
	gpuErrchk(cudaMemcpy(&noDeletedDebug, d_noDeletedDebug, sizeof(int), cudaMemcpyDeviceToHost));
	printf("Deleted %d weak %d-mers\n", noDeletedDebug, merLength);

	DisplaySizeInfo(finalTreeLength, merLength);
	//DisplayTree(finalTree);
	//DisplayTable(finalTree, finalTreeLength);
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_tree));
	gpuErrchk(cudaFree(d_treeLength));
	free(chunk);
	free(clearedChunk);
	out_treeLength = finalTreeLength;
	return finalTree;
}