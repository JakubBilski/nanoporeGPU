#include <cuda_runtime.h>

template <int TMerLength>
__global__
void AddPrecleanedChunkToGraph(int noBlocks, char* file, int length, int* tree, int* treeLength)
{
	int thid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	int k;	//index of letter being processed
	int currentNode;	//index of node correspoding to already processed letters' string, always multiple of 4
	bool isStringValid;
	int merLetters[TMerLength];
	//if (threadIdx.x == 0)
	//{
	//	printf("[block %d] elo jestem w srodku\n", blockIdx.x);
	//}
	while (thid < length - TMerLength)
	{
		k = 0;
		currentNode = 0;
		isStringValid = true;
		for (int i = 0; i < TMerLength; i++)
		{
			merLetters[i] = file[thid + i];
			switch (merLetters[i])
			{
			case 'A':
			{
				merLetters[i] = 0;
				break;
			}
			case 'T':
			{
				merLetters[i] = 1;
				break;
			}
			case 'C':
			{
				merLetters[i] = 2;
				break;
			}
			case 'G':
			{
				merLetters[i] = 3;
				break;
			}
			case '\n':
			{
				isStringValid = false;
				break;
			}
			default:
				printf("Letter[%d] was ilegal: %c!!!!\n", thid + i, (char)(merLetters[i]));
			}
			//check isStringValid here, or let it go through all of the letters?
		}
		if (isStringValid)
		{
			while (k < TMerLength)
			{
				int nextNode = tree[currentNode + merLetters[k]];
				//order of this ifs should probably be altered
				if (nextNode == 0 && atomicExch(&(tree[currentNode + merLetters[k]]), -1) == 0)
				{
					//if(node was not present in tree) && (this thread was ordered to allocate new node)
					int newNode = atomicAdd(treeLength, 4);
					tree[currentNode + merLetters[k]] = newNode;
					currentNode = newNode;
					k++;
				}
				else if (nextNode != -1 && nextNode != 0) //node is present
				{
					currentNode = nextNode;
					k++;
				}
				//else busy waiting
			}
			//k == merLength, currentNode == (node where we store edges' weights)
			switch (file[thid + TMerLength])
			{
			case 'A':
			{
				atomicAdd(&(tree[currentNode]), 1);
				break;
			}
			case 'T':
			{
				atomicAdd(&(tree[currentNode + 1]), 1);
				break;
			}
			case 'C':
			{
				atomicAdd(&(tree[currentNode + 2]), 1);
				break;
			}
			case 'G':
			{
				atomicAdd(&(tree[currentNode + 3]), 1);
				break;
			}
			//case '\n':
			//{
			//	string was the end of the sequence
			//}
			}
		}
		thid += BLOCK_SIZE * noBlocks;
	}
}