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
	int act_kmer_id = -1;
	int i = 0;
	for (; i < read_len - k; i++)
	{
		if (act_kmer_id = get_kmer_id(read.substr(i, k), DBG))
		{
			check_head = true;
			//CorrectHead(read.substr(0, i),...);
			break;
		}
	}
	if (i > 0)
	{
		std::cout << "Correct head" << std::endl;
	}
	if (!check_head)
	{
		std::cout << "Read discarded - no solid regions";
		return;
	}
	int weak_region = i;
	std::string begin_kmer = "";
	bool inside_graph = true;
	for (i++; i < read_len - k; i++)
	{
		if (inside_graph)
		{
			/*if (!get_kmer_id(read.substr(i-1, k), DBG))
				throw "AAAA";*/
			int next_kmer_id = DBG[act_kmer_id + char_id(read[i+k-1])];
			if (next_kmer_id)
			{
				/*if (!get_kmer_id(read.substr(i, k), DBG))
					throw "AAAA";*/
				act_kmer_id = next_kmer_id;
			}
			else
			{
				inside_graph = false;
				begin_kmer = read.substr(i-1, k);
				weak_region = i+k-1;  // begin of weak region is after end of last solid
			}
		}
		else
		{
			std::string end_kmer = read.substr(i, k);
			if (act_kmer_id = get_kmer_id(end_kmer, DBG))
			{
				int weak_len = i - weak_region;  // i - first letter of ending solid kmer
				if (weak_len <= MAX_WEAK_CORRECTED_REGION && weak_len >= 1)
				{
					std::string corrected_weak = CorrectInner(begin_kmer, end_kmer,
						read.substr(weak_region + k, weak_len),
						100, DBG);
					read = read.substr(0, weak_region) + corrected_weak + read.substr(i);
					i = i + corrected_weak.size() - weak_len;
					read_len = read_len + corrected_weak.size() - weak_len;
				}
				
				inside_graph = true;
			}
		}
	}
	//correct tail
}

void Correct(std::fstream& input_file, std::fstream& output_file, std::vector<int> DBG, int k)
{
	std::string read;
	while (input_file >> read)
	{
		//std::cout << ".";
		CorrectRead(read, DBG, k);
		output_file << read << "\n";
	}
}


void TestEditDist()
{
	std::string s1 = "ACTAAAGE";
	std::string s2 = "ACACAGE";
	std::cout << EditDistance(s1, s2) << ", expected 2" << std::endl;

	s1 = "BANANANA";
	s2 = "B";
	std::cout << EditDistanceCheckPrefix(s1, s2, 1) << ", expected 1" << std::endl;
}


void TestCorrect()
{
	std::cout << "TEST EDIT DIST" << std::endl;
	TestEditDist();
	std::vector<int> DBG;
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
	CorrectRead(read, DBG, k);
}

