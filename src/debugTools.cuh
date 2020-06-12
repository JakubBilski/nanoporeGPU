void DisplaySubtreeRec(int* tree, int root, std::string gathered, int k)
{
	if (k == MER_LENGHT)
	{
		if(tree[root] == -1)
			std::cout << gathered << std::endl;
		//std::cout << gathered << " -> " << gathered.substr(1) << "A " << tree[root] << std::endl;
		//std::cout << gathered << " -> " << gathered.substr(1) << "T " << tree[root + 1] << std::endl;
		//std::cout << gathered << " -> " << gathered.substr(1) << "C " << tree[root + 2] << std::endl;
		//std::cout << gathered << " -> " << gathered.substr(1) << "G " << tree[root + 3] << std::endl;
	}
	else
	{
		if(tree[root] != 0)
			DisplaySubtreeRec(tree, tree[root], gathered + "A", k + 1);
		if (tree[root+1] != 0)
			DisplaySubtreeRec(tree, tree[root + 1], gathered + "T", k + 1);
		if (tree[root+2] != 0)
			DisplaySubtreeRec(tree, tree[root + 2], gathered + "C", k + 1);
		if (tree[root+3] != 0)
			DisplaySubtreeRec(tree, tree[root + 3], gathered + "G", k + 1);
	}
}

void DisplayTree(int* tree)
{
	std::cout << "\nTree:\n";
	DisplaySubtreeRec(tree, 0, "", 0);
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