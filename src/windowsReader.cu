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

template <int TNoBlocks> void simpleGPU(std::ifstream& fs);
template <int TNoBlocks> void precleanedStreamGPU(std::ifstream& fs);
template <int TNoBlocks> void precleanedGPU(std::ifstream& fs);
template <int TNoBlocks> void precleanedJumpGPU(std::ifstream& fs);
void stringCPU(std::ifstream& fs);
void simpleCPU(std::ifstream& fs);
void jumpCPU(std::ifstream& fs);

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
      printf("Usage: reader file_path\n");
      return 0;
    }

	printf("Starting\n");

	const int noTests = 2;
	for (size_t test = 0; test < noTests; test++)
	{
		printf(argv[1]);
		printf(", run %d\n", (int)test);

		std::ifstream fs(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		clock_t start = clock();
		simpleCPU(fs);
		printf("precleanedGPU<1> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		simpleCPU(fs);
		printf("simpleCPU done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		jumpCPU(fs);
		printf("jumpCPU done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		precleanedGPU<5>(fs);
		printf("precleanedGPU<5> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		precleanedGPU<10>(fs);
		printf("precleanedGPU<10> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		precleanedGPU<20>(fs);
		printf("precleanedGPU<20> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		precleanedJumpGPU<1>(fs);
		printf("precleanedJumpGPU<1> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		precleanedJumpGPU<5>(fs);
		printf("precleanedJumpGPU<5> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		precleanedJumpGPU<10>(fs);
		printf("precleanedJumpGPU<10> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		precleanedJumpGPU<20>(fs);
		printf("precleanedJumpGPU<20> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		start = clock();
		precleanedStreamGPU<1>(fs);
		printf("precleanedStreamGPU<1> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedStreamGPU<5>(fs);
		//printf("precleanedStreamGPU<5> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		//fs.open(argv[1], std::ios::in | std::ios::binary);
		//assertOpenFile(fs, argv[1]);
		//start = clock();
		//precleanedStreamGPU<10>(fs);
		//printf("precleanedStreamGPU<10> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		//fs.close();

		fs.open(argv[1], std::ios::in | std::ios::binary);
		assertOpenFile(fs, argv[1]);
		start = clock();
		precleanedStreamGPU<20>(fs);
		printf("precleanedStreamGPU<20> done in %f seconds\n", 0.001f * (clock() - start) * 1000 / CLOCKS_PER_SEC);
		fs.close();

		printf("\n");
	}
	return 0;
}

template <int TNoBlocks>
void simpleGPU(std::ifstream& fs)
{
	char* d_chunk;
	int* d_out_numAs;
	int noAs = 0;
	//char* chunk = (char*)malloc(sizeof(char)*INPUT_CHUNK_SIZE);
	//gpuErrchk(cudaMalloc(&d_chunk, INPUT_CHUNK_SIZE * sizeof(char)));
	//gpuErrchk(cudaMalloc(&d_out_numAs, sizeof(int)));
	//gpuErrchk(cudaMemset(d_out_numAs, 0, sizeof(int)));
	//int chunkSize = 0;
	//do
	//{
	//	fs.read(chunk, INPUT_CHUNK_SIZE);
	//	chunkSize = fs.gcount();
	//	gpuErrchk(cudaMemcpy(d_chunk, chunk, chunkSize * sizeof(char), cudaMemcpyHostToDevice));
	//	AddChunkToGraph << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, chunkSize, d_out_numAs);
	//	kernelErrchk();
	//} while (chunkSize == INPUT_CHUNK_SIZE);
	//gpuErrchk(cudaMemcpy(&noAs, d_out_numAs, sizeof(int), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaFree(d_chunk));
	//gpuErrchk(cudaFree(d_out_numAs));
	//free(chunk);
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

template <int TNoBlocks>
void precleanedStreamGPU(std::ifstream& fs)
{
	char* d_chunk;
	unsigned long long int* d_out_numAs;
	char* clearedChunk = (char*)malloc(sizeof(char)*HOST_CHUNK_SIZE);
	int clearedChunkSize = 0;
	unsigned long long int noAs = 0;
	gpuErrchk(cudaMalloc(&d_chunk, DEVICE_CHUNK_SIZE * sizeof(char)));
	gpuErrchk(cudaMalloc(&d_out_numAs, sizeof(unsigned long long int)));
	gpuErrchk(cudaMemset(d_out_numAs, 0, sizeof(unsigned long long int)));
	while(!fs.eof())
	{
		fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		int freeSpace = DEVICE_CHUNK_SIZE - clearedChunkSize;
		fs.getline(&(clearedChunk[clearedChunkSize]), freeSpace);
		if (fs.fail())	//if line didn't fit into chunk
		{
			if (fs.eof())
			{
				break;
			}
			//printf("Kernel launch\n");
			gpuErrchk(cudaMemcpy(d_chunk, clearedChunk, clearedChunkSize * sizeof(char), cudaMemcpyHostToDevice));
			AddPrecleanedChunkToGraph << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_out_numAs);
			kernelErrchk();
			int savedLen = fs.gcount() - 1;
			fs.clear();
			memcpy(clearedChunk, clearedChunk + clearedChunkSize, sizeof(char)*savedLen);
			clearedChunkSize = savedLen;
			fs.getline(&(clearedChunk[clearedChunkSize]), DEVICE_CHUNK_SIZE);
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
	//printf("ClearedChunkSize: %d/%d\n", clearedChunkSize, INPUT_CHUNK_SIZE);
	//printf("Kernel launch\n");
	gpuErrchk(cudaMemcpy(d_chunk, clearedChunk, clearedChunkSize * sizeof(char), cudaMemcpyHostToDevice));
	AddPrecleanedChunkToGraph << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_out_numAs);
	kernelErrchk();
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(&noAs, d_out_numAs, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_out_numAs));
	free(clearedChunk);
	if (noAs != DEBUG_A_COUNT)
	{
		throw std::runtime_error("invalid value calculated by tested function");
	}
}

template <int TNoBlocks>
void precleanedGPU(std::ifstream& fs)
{
	char* d_chunk;
	unsigned long long int* d_out_numAs;
	char* chunk = (char*)malloc(sizeof(char)*HOST_CHUNK_SIZE);
	char* clearedChunk = (char*)malloc(sizeof(char)*DEVICE_CHUNK_SIZE);
	int clearedChunkSize = 0;
	int chunkOffset = 0;
	unsigned long long int noAs = 0;
	int cutPhase = 2;
	gpuErrchk(cudaMalloc(&d_chunk, DEVICE_CHUNK_SIZE * sizeof(char)));
	gpuErrchk(cudaMalloc(&d_out_numAs, sizeof(unsigned long long int)));
	gpuErrchk(cudaMemset(d_out_numAs, 0, sizeof(unsigned long long int)));
	while (!fs.eof())
	{
		fs.read(chunk + chunkOffset, HOST_CHUNK_SIZE - chunkOffset);
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
					AddPrecleanedChunkToGraph << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_out_numAs);
					kernelErrchk();
					clearedChunkSize = 0;
				}
				if (i == HOST_CHUNK_SIZE)
				{
					memcpy(chunk, chunk + startOfLetters, sizeof(char)*(HOST_CHUNK_SIZE - startOfLetters));
					chunkOffset = HOST_CHUNK_SIZE - startOfLetters;
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
					if (i == HOST_CHUNK_SIZE)
					{
						cutPhase = j;
						break;
					}
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
				}
				if (i == HOST_CHUNK_SIZE)
				{
					cutPhase = 3;
					chunkOffset = 0;
					break;
				}
			}
		}
	}
	gpuErrchk(cudaMemcpy(d_chunk, clearedChunk, clearedChunkSize * sizeof(char), cudaMemcpyHostToDevice));
	AddPrecleanedChunkToGraph << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_out_numAs);
	kernelErrchk();
	gpuErrchk(cudaMemcpy(&noAs, d_out_numAs, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_out_numAs));
	free(chunk);
	free(clearedChunk);
	if (noAs != DEBUG_A_COUNT)
	{
		throw std::runtime_error("invalid value calculated by tested function");
	}
}

template <int TNoBlocks>
void precleanedJumpGPU(std::ifstream& fs)
{
	char* d_chunk;
	unsigned long long int* d_out_numAs;
	char* chunk = (char*)malloc(sizeof(char)*HOST_CHUNK_SIZE);
	char* clearedChunk = (char*)malloc(sizeof(char)*DEVICE_CHUNK_SIZE);
	int clearedChunkSize = 0;
	int chunkOffset = 0;
	unsigned long long int noAs = 0;
	int cutPhase = 2;
	gpuErrchk(cudaMalloc(&d_chunk, DEVICE_CHUNK_SIZE * sizeof(char)));
	gpuErrchk(cudaMalloc(&d_out_numAs, sizeof(unsigned long long int)));
	gpuErrchk(cudaMemset(d_out_numAs, 0, sizeof(unsigned long long int)));
	int lettersLength;
	while (!fs.eof())
	{
		fs.read(chunk + chunkOffset, HOST_CHUNK_SIZE - chunkOffset);
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
					AddPrecleanedChunkToGraph << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_out_numAs);
					kernelErrchk();
					clearedChunkSize = 0;
				}
				if (i == HOST_CHUNK_SIZE)
				{
					memcpy(chunk, chunk + startOfLetters, sizeof(char)*(HOST_CHUNK_SIZE - startOfLetters));
					chunkOffset = HOST_CHUNK_SIZE - startOfLetters;
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
				if (i + 2 >= HOST_CHUNK_SIZE)
				{
					cutPhase = 0;
					break;
				}
				if (i + 2 + lettersLength >= HOST_CHUNK_SIZE)
				{
					cutPhase = 1;
					break;
				}
				i += 2 + lettersLength;
				while (chunk[i] != '\n')
				{
					i++;
					if (i == HOST_CHUNK_SIZE)
					{
						cutPhase = 2;
						endOfChunk = true;
						break;
					}
				}
				i++;
				if (i == HOST_CHUNK_SIZE)
				{
					cutPhase = 3;
					chunkOffset = 0;
					break;
				}
			}
		}
	}
	gpuErrchk(cudaMemcpy(d_chunk, clearedChunk, clearedChunkSize * sizeof(char), cudaMemcpyHostToDevice));
	AddPrecleanedChunkToGraph << <TNoBlocks, BLOCK_SIZE >> > (TNoBlocks, d_chunk, clearedChunkSize, d_out_numAs);
	kernelErrchk();
	gpuErrchk(cudaMemcpy(&noAs, d_out_numAs, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_chunk));
	gpuErrchk(cudaFree(d_out_numAs));
	free(chunk);
	free(clearedChunk);
	if (noAs != DEBUG_A_COUNT)
	{
		throw std::runtime_error("invalid value calculated by tested function");
	}
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