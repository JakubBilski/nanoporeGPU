#include <cuda_runtime.h>

template <int TMerLength>
__global__
void AddPrecleanedChunkToGraph(int noBlocks, char* file, int length, int* tree, int* treeLength)
{
	int thid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	int k;	//index of letter being processed
	int currentNode;	//index of node correspoding to already processed letters' string, always multiple of 4
	bool isStringValid;
	int merLetters[TMerLength];	//string, loaded to thread memory to speed up access
	while (thid < length - TMerLength - 1)
	{
		k = 0;
		currentNode = 0;
		isStringValid = true;
		for (int i = 0; i < TMerLength; i++) //loading letters from global memory
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
		if (isStringValid) //if loaded string is a valid part of sequence (it doesn't contain \n)
		{
			while (k < TMerLength)
			{
				int nextNode = tree[currentNode + merLetters[k]];	//get index of the node corresponding to the next letter

				//following 'if' explanation:
				//if this thread reads value 0 from the next node
				//it means that tree[currentNode + merLetters[k]] node has not yet been allocated
				//Then this thread exchanges this 0 with a -1 valuem which tells the other threads that the memory is being allocated.
				//The whole operation is atomic, so only one thread will read the 0 value and perform the allocation
				if (atomicCAS(&(tree[currentNode + merLetters[k]]), 0, -1) == 0)
				{
					int newNode = atomicAdd(treeLength, 4);
					tree[currentNode + merLetters[k]] = newNode;
					currentNode = newNode;
					k++;
				}
				else if (nextNode != -1 && nextNode != 0) //if node is present
				{
					currentNode = nextNode;
					k++;
				}
				//else busy waiting
			}
			//k == merLength, currentNode == (node where we store edges' weights)
			switch (file[thid + TMerLength]) //check what k-mer this string transforms into and increase proper edge weight
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
		thid += BLOCK_SIZE * noBlocks;	//go to the next k-mer
	}
}