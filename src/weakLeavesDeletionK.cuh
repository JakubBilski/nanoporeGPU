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
		thid += BLOCK_SIZE * noBlocks;	//go to the next leaf
	}
}