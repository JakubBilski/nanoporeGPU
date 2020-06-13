#pragma once
#include <cuda_runtime.h>
#include "defines.cuh"

template <int TMerLength>
__global__
void DeleteWeakLeaves(int noBlocks, int* tree)
{
	//liczba lisci = 4^TMerLength

	int thid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	//dopoki sa jeszcze jakies liscie
	//potrzebne przejscie int -> ACCAGAT
	int noKMers = 1 << (2 * TMerLength);
	while (thid < noKMers)
	{
		int k = 0;
		int node = 0;
		int thidMask = thid;
		while (k < TMerLength)
		{
			int nextIndex = thidMask & 3;
			node = tree[node + nextIndex];
			k++;
			thidMask = thidMask >> 2;
		}
		//tutaj juz node = leaf
		if (tree[node] + tree[node + 1] + tree[node + 2] + tree[node + 3] < WEAK_TRESHOLD)
		{
			tree[node] = -1;
			tree[node+1] = -1;
			tree[node+2] = -1;
			tree[node+3] = -1;
		}
		else
		{
			for (int i = 0; i < 4; i++)
			{
				int searchingMask = thid >> 2;
				int searchingK = 0;
				int searchingNode = 0;
				while (searchingK < TMerLength - 1)	//one less
				{
					int nextIndex = searchingMask & 3;
					searchingNode = tree[searchingNode + nextIndex];
					searchingK++;
					searchingMask = searchingMask >> 2;
				}
				//i is the last nextIndex
				tree[node+i] = tree[searchingNode + i];
			}
		}
		thid += BLOCK_SIZE * noBlocks;	//go to the next leaf
	}
}