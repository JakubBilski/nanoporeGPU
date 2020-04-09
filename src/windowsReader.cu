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

void simpleGPU(std::ifstream& fs);
void stringCPU(std::ifstream& fs);
void precleanedStreamGPU(std::ifstream& fs);
void simpleCPU(std::ifstream& fs);

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
      printf("Usage: reader file_path\n");
      return 0;
    }

	printf("Starting\n");

	std::ifstream fs(argv[1], std::ios::in | std::ios::binary);
	clock_t start = clock();
	stringCPU(fs);
	printf("stringCPU done in %f seconds\n", 0.001f * (clock() - start)*1000 / CLOCKS_PER_SEC);
	fs.close();

	fs.open(argv[1], std::ios::in | std::ios::binary);
	start = clock();
	precleanedStreamGPU(fs);
	printf("precleanedStreamGPU done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
	fs.close();

	fs.open(argv[1], std::ios::in | std::ios::binary);
	start = clock();
	simpleCPU(fs);
	printf("simpleCPU done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
	fs.close();

	return 0;
}

void simpleGPU(std::ifstream& fs)
{
	char* d_chunk;
	int* d_out_numAs;
	int noAs = 0;
	char* chunk = (char*)malloc(sizeof(char)*INPUT_CHUNK_SIZE);
	gpuErrchk(cudaMalloc(&d_chunk, INPUT_CHUNK_SIZE * sizeof(char)));
	gpuErrchk(cudaMalloc(&d_out_numAs, sizeof(int)));
	gpuErrchk(cudaMemset(&d_out_numAs, sizeof(int), 0));
	int chunkSize = 0;
	do
	{
		fs.read(chunk, INPUT_CHUNK_SIZE);
		chunkSize = fs.gcount();
		gpuErrchk(cudaMemcpy(d_chunk, chunk, chunkSize * sizeof(char), cudaMemcpyHostToDevice));
		AddChunkToGraph << <1, BLOCK_SIZE >> > (d_chunk, chunkSize, d_out_numAs);
		kernelErrchk();
	} while (chunkSize == INPUT_CHUNK_SIZE);
	gpuErrchk(cudaMemcpy(&noAs, d_out_numAs, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_out_numAs));
	free(chunk);
	if (noAs != DEBUG_A_COUNT)
	{
		throw std::runtime_error("invalid value calculated by tested function");
	}
}

void stringCPU(std::ifstream& fs)
{
	std::string line;
	int noAs = 0;
	do
	{
		fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::getline(fs, line);
		for (auto c : line)
		{
			if (c == 'A')
			{
				noAs++;
			}
		}
		fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	} while (!fs.eof());
	if (noAs != DEBUG_A_COUNT)
	{
		throw std::runtime_error("invalid value calculated by tested function");
	}
}

void precleanedStreamGPU(std::ifstream& fs)
{
	char* d_chunk;
	int* d_out_numAs;
	char* clearedChunk = (char*)malloc(sizeof(char)*INPUT_CHUNK_SIZE);
	int clearedChunkSize = 0;
	int noAs = 0;
	gpuErrchk(cudaMalloc(&d_chunk, INPUT_CHUNK_SIZE * sizeof(char)));
	gpuErrchk(cudaMalloc(&d_out_numAs, sizeof(int)));
	gpuErrchk(cudaMemset(&d_out_numAs, sizeof(int), 0));
	while(!fs.eof())
	{
		fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		int freeSpace = INPUT_CHUNK_SIZE - clearedChunkSize;
		fs.getline(&(clearedChunk[clearedChunkSize]), freeSpace);
		if (fs.fail())	//if line didn't fit into chunk
		{
			if (fs.eof())
			{
				break;
			}
			//printf("Kernel launch\n");
			gpuErrchk(cudaMemcpy(d_chunk, clearedChunk, clearedChunkSize * sizeof(char), cudaMemcpyHostToDevice));
			AddPrecleanedChunkToGraph << <NO_BLOCKS, BLOCK_SIZE >> > (d_chunk, clearedChunkSize, d_out_numAs);
			kernelErrchk();
			int savedLen = fs.gcount() - 1;
			fs.clear();
			memcpy(clearedChunk, clearedChunk + clearedChunkSize, sizeof(char)*savedLen);
			clearedChunkSize = savedLen;
			fs.getline(&(clearedChunk[clearedChunkSize]), INPUT_CHUNK_SIZE);	
			if (fs.fail())
			{
				printf("Unhandled error\n");	//case when some line is longer than INPUT_CHUNK_SIZE
				exit(0);
			}
			clearedChunkSize += fs.gcount();
		}
		else
		{
			clearedChunkSize += fs.gcount();
		}
		fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	//printf("Kernel launch\n");
	gpuErrchk(cudaMemcpy(d_chunk, clearedChunk, clearedChunkSize * sizeof(char), cudaMemcpyHostToDevice));
	AddPrecleanedChunkToGraph << <NO_BLOCKS, BLOCK_SIZE >> > (d_chunk, clearedChunkSize, d_out_numAs);
	kernelErrchk();
	gpuErrchk(cudaMemcpy(&noAs, d_out_numAs, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_out_numAs));
	free(clearedChunk);
	if (noAs != DEBUG_A_COUNT)
	{
		throw std::runtime_error("invalid value calculated by tested function");
	}
}

void simpleCPU(std::ifstream& fs)
{
	int noAs = 0;
	char* chunk = (char*)malloc(sizeof(char)*INPUT_CHUNK_SIZE);
	int cutPhase = 2;
	while (!fs.eof())
	{
		fs.read(chunk, INPUT_CHUNK_SIZE);
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
			int startOfLetters = i;
			while (chunk[i] != '\n')
			{
				if (chunk[i] == 'A')
				{
					noAs++;
				}
				i++;
				if (i == INPUT_CHUNK_SIZE)
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
			if (!endOfChunk)
			{
				for (int j = 0; j < 3 && !endOfChunk; j++)
				{
					while (chunk[i] != '\n')
					{
						i++;
						if (i == INPUT_CHUNK_SIZE)
						{
							cutPhase = j;
							endOfChunk = true;
							break;
						}
					}
					i++;
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