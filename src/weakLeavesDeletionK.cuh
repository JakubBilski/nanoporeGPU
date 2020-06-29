#pragma once
#include <cuda_runtime.h>
#include "defines.cuh"

__device__
int NodeFromThid(int* tree, int thid, int length)
{
	//przejscie int -> ACCAGAT
	//jedna literka to 2 bity w thidzie

	int k = 0;
	int node = 0;
	int thidMask = thid;
	while (k < length)
	{
		int nextIndex = thidMask & 3;
		node = tree[node + nextIndex];
		if (node == 0)
		{
			return 0;
		}
		k++;
		//to sie po cichu wywali jak bedzie TMerLength*2 > liczba bitow w thidMask
		thidMask = thidMask >> 2;
	}
	return node;
}

template <int TMerLength>
__global__
void DeleteWeakLeaves(int noBlocks, int* tree, int* noDeleted_debug)
{
	//liczba lisci = 4^TMerLength

	unsigned int thid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	//dopoki sa jeszcze jakies liscie
	//potrzebne przejscie int -> ACCAGAT
	int noKMers = 1 << (2 * TMerLength);
	while (thid < noKMers)
	{
		int node = NodeFromThid(tree, thid, TMerLength);
		
		if (node!=0)
		{
			//tutaj juz node = leaf
			if (tree[node] + tree[node + 1] + tree[node + 2] + tree[node + 3] < WEAK_TRESHOLD)
			{
				atomicAdd(noDeleted_debug, 1);
				/*
				node ACCAGAT
				nie istnieje
				*/
				/*
				usuwamy wszystkie wskaüniki z tego noda (na wszelki wypadek
				*/
				tree[node] = 0;
				tree[node + 1] = 0;
				tree[node + 2] = 0;
				tree[node + 3] = 0;
				/*
				znajdujemy ostatni char noda (T)
				*/
				int last_char = (thid >> (TMerLength - 1) * 2) && 3;
				/*
				usuwamy wskaünik z noda ACCAGA (przechodzπc po drzewie o jeden poziom mniej)
				*/
				int upNode = NodeFromThid(tree, thid, TMerLength - 1);
				if(upNode>0)
					tree[upNode + last_char] = 0;
				/*
				usuwamy wskaüniki z 
				AACCAGA
				CACCAGA
				GACCAGA
				TACCAGA
				*/
				for (int i = 0; i < 4; i++)
				{
					int searchingNode = NodeFromThid(tree, (thid << 2)+i, TMerLength);
					if(searchingNode>0)
						tree[searchingNode + last_char] = 0;
				}
			}
			else
			{
				/*
				node ACCAGAT
				uzupe≥niamy wskaüniki z tego do
				CCAGATA
				CCAGATC
				CCAGATG
				CCAGATT
				
				Czyli znajdujemy node CCAGAT i przepisujemy wszystkie 4 wskaüniki.
				*/
				int searchingNode = NodeFromThid(tree, thid>>2, TMerLength-1);
				if (searchingNode>0)
				{
					for (int i = 0; i < 4; i++)
					{

						tree[node + i] = tree[searchingNode + i];
					}
				}
			}
		}
		thid += BLOCK_SIZE * noBlocks;	//go to the next leaf
	}
}

