#pragma once

#include <vector>
#include <string>

#include "graphUtils.h"
#include "defines.h"
#include "editDistance.h"


class GraphDFS
{
private:
	int end_node;
	std::vector<int>& DBG;
	std::vector<bool> flags;
	int max_edit_distance;
	int max_depth;
	std::string act_correction;
	std::string desired;
	std::string best_correction;
	int best_edit_dist;
	int k;
	int depth;
public:
	GraphDFS(int end_node, std::vector<int>& DBG, int max_edit_distance, int max_depth, std::string desired, int k) :
		end_node(end_node), DBG(DBG), max_edit_distance(max_edit_distance), max_depth(max_depth), desired(desired), k(k)
	{
		flags.resize(DBG.size() / 4);
		for (int i = 0; i < flags.size(); i++)
			flags[i] = 0;
		best_correction = desired;
		best_edit_dist = max_edit_distance+1;
	}
	void DFS(int node)
	{
		std::string substr = act_correction.substr(0, act_correction.size() - k);
		if (node == end_node)
		{
			int edit_distance = EditDistance(act_correction.substr(0, act_correction.size() - k), desired);
			std::cout << "end node, " << act_correction << std::endl;
			return;
		}
		bool in_range = EditDistanceCheckPrefix(substr, desired, best_edit_dist-1);
		if (!in_range)
			return;
		flags[node / 4] = true;
		for (int i = 0; i < 4; i++)
		{

			//std::cout << act_correction.size() << " entering " << reverse_char_id(i) << std::endl;
			int next_id = DBG[node + i];
			if (next_id && !flags[next_id / 4])
			{
				act_correction.push_back(reverse_char_id(i));
				depth++;
				this->DFS(next_id);
				depth--;
				act_correction.pop_back();
			}
			//std::cout << act_correction.size() << " leaving " << reverse_char_id(i) << std::endl;
		}
	}
};



std::string CorrectInner(std::string begin_kmer, std::string end_kmer, std::string& weak_region,
	int max_branching, std::vector<int>& DBG)
{
#ifdef  DEBUG
	std::cout << "Correct inner input: " << begin_kmer << ";" << weak_region << ";" << end_kmer << std::endl;
	//return "";
#endif //  DEBUG
	
	int max_edit_distance = 4;
	int max_edit_table_size = 100;
	int max_depth = 100 / weak_region.size();

	int id0 = get_kmer_id(begin_kmer, DBG);
	int id1 = get_kmer_id(end_kmer, DBG);
	if (!id0 || !id1)
	{
		throw "Cos nie tak";
		return "";
	}

	
	GraphDFS g(id1, DBG, max_edit_distance, max_depth, weak_region, begin_kmer.size());
	g.DFS(id0);

	return "";
}





