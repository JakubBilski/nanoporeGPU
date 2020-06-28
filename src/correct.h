#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "correctInner.h"
#include "graphUtils.h"



void CorrectRead(std::string& read, std::vector<int>& DBG, int k)
{
	int read_len = read.length();
	bool check_head = false;
	int i = 0;
	for (; i < read_len - k; i++)
	{
		if (get_kmer_id(read.substr(i, k), DBG))
		{
			check_head = true;
			//CorrectHead(read.substr(0, i),...);
			break;
		}
	}
	if (!check_head)
	{
		std::cout << "Read discarded - no solid regions";
		return;
	}
	int weak_region = i;
	std::string begin_kmer = "";
	for (; i < read_len - k; i++)
	{
		std::string end_kmer = read.substr(i, k);
		if (get_kmer_id(end_kmer, DBG))
		{
			if (i > weak_region + 1)
				CorrectInner(begin_kmer, end_kmer, read.substr(weak_region+k, i - weak_region -k),
					100, DBG);
			weak_region = i;
			begin_kmer = end_kmer;
		}
	}
	//correct tail
}

void Correct(std::fstream& input_file, std::fstream& output_file, std::vector<int> DBG, int k)
{
	std::string read;
	while (input_file >> read)
	{
		CorrectRead(read, DBG, k);
		output_file << read;
	}
}


void TestCorrect()
{
	std::string s1 = "ACTAAAGE";
	std::string s2 = "ACACAGE";
	std::cout << EditDistance(s1, s2) << std::endl;
	/*std::vector<int> DBG;
	for (int i = 0; i < 4; i++)
		DBG.push_back(0);

	std::string DNA = "AATCATCTTAACTACATTTACTAA";
	int k = 3;
	for (int i = 0; i < DNA.size() - k+1; i++)
	{
		std::string kmer = DNA.substr(i, k);
		std::cout << kmer << std::endl;
		AddKmer(kmer, DBG);
	}
	std::string past_kmer = DNA.substr(0, k);
	for (int i = 1; i < DNA.size() - k+1; i++)
	{
		std::string kmer = DNA.substr(i, k);
		AddEdge(past_kmer, kmer, DBG);
		past_kmer = kmer;
	}

	std::cout << "DBG:" << std::endl;
	for (int i = 0; i < DBG.size(); i += 4)
	{
		std::cout << i << ":::";
		for (int j = 0; j < 4; j++)
			std::cout << DBG[i + j] << " ";
		std::cout << std::endl;
	}

	std::cout << std::endl << std::endl << "Check:" << std::endl;

	std::string read = "AATCATGGTAACTA";
	CorrectRead(read, DBG, k);*/
}