void DisplaySubtreeRec(int* tree, int root, std::string gathered, int k, int merLength)
{
	if (k == merLength)
	{
		std::cout << gathered << " -> " << gathered.substr(1) << "A " << tree[root];
		std::cout << ((tree[root] == -1) ? "\t weak kmer!\n" : (tree[root] == 0) ? "\t edge not present!\n" : "\n");
		std::cout << gathered << " -> " << gathered.substr(1) << "T " << tree[root + 1];
		std::cout << ((tree[root+1] == -1) ? "\t weak kmer!\n" : (tree[root+1] == 0) ? "\t edge not present!\n" : "\n");
		std::cout << gathered << " -> " << gathered.substr(1) << "C " << tree[root + 2];
		std::cout << ((tree[root+2] == -1) ? "\t weak kmer!\n" : (tree[root+2] == 0) ? "\t edge not present!\n" : "\n");
		std::cout << gathered << " -> " << gathered.substr(1) << "G " << tree[root + 3];
		std::cout << ((tree[root+3] == -1) ? "\t weak kmer!\n" : (tree[root+3] == 0) ? "\t edge not present!\n" : "\n");
	}
	else
	{
		if(tree[root] != 0)
			DisplaySubtreeRec(tree, tree[root], gathered + "A", k + 1, merLength);
		if (tree[root+1] != 0)
			DisplaySubtreeRec(tree, tree[root + 1], gathered + "T", k + 1, merLength);
		if (tree[root+2] != 0)
			DisplaySubtreeRec(tree, tree[root + 2], gathered + "C", k + 1, merLength);
		if (tree[root+3] != 0)
			DisplaySubtreeRec(tree, tree[root + 3], gathered + "G", k + 1, merLength);
	}
}

void DisplayTree(int* tree, int merLength)
{
	std::cout << "\nTree:\n";
	DisplaySubtreeRec(tree, 0, "", 0, merLength);
}

void DisplayTable(int* table, int length)
{
	std::cout << "\nTable\n";
	for (size_t i = 0; i < length; i++)
	{
		std::cout << table[i] << " ";
		if(i%10 == 9)
			std::cout << std::endl;
	}
	std::cout << std::endl;
}

void DisplaySizeInfo(int finalTreeLength, int merLength)
{
	int thisMerTreeMaxLength = 1;
	for (int i = 0; i < merLength; i++)
	{
		thisMerTreeMaxLength *= 4;
		thisMerTreeMaxLength += 1;
	}
	thisMerTreeMaxLength *= 4;
	printf("Tree size: %d (%d MB), %.2f%% coverage\n", finalTreeLength, (finalTreeLength * sizeof(int)) / (1024 * 1024), 100.0f * finalTreeLength / thisMerTreeMaxLength);
}