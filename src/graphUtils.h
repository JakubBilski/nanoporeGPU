#pragma once

#include <vector>
#include <string>


int char_id(char c)
{
	switch (c)
	{
	case 'A':
		return 0;
	case 'T':
		return 1;
	case 'C':
		return 2;
	case 'G':
		return 3;
	}
	throw "Zla litera!!";
	return -1;
}

char reverse_char_id(int c)
{
	switch (c)
	{
	case 0:
		return 'A';
	case 1:
		return'T';
	case 2:
		return 'C';
	case 3:
		return 'G';
	}
	throw "Zla litera!!";
	return 'X';
}


int get_kmer_id(std::string kmer, std::vector<int>& DBG)
{
	int node = 0;
	for (int i=0;i<kmer.size();i++)
	{
		char c = kmer[i];
		node = DBG[node + char_id(c)];
		if (node == 0)
			return 0;
	}
	return node;
}

void AddKmer(std::string kmer, std::vector<int>& DBG)
{
	int node = 0;
	for (auto c : kmer)
	{
		int id = char_id(c);
		node = node + id;
		if (DBG[node] == 0)
		{
			DBG[node] = DBG.size();
			for (int i = 0; i < 4; i++)
				DBG.push_back(0);
		}
		node = DBG[node];
	}
}
void AddEdge(std::string kmer1, std::string kmer2, std::vector<int>& DBG)
{
	int id0 = get_kmer_id(kmer1, DBG);
	int id1 = get_kmer_id(kmer2, DBG);
	DBG[id0 + char_id(kmer2[kmer2.size() - 1])] = id1;
}
