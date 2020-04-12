void DisplaySubtreeRec(int* tree, int root, std::string gathered, int k)
{
	if (k == MER_LENGHT)
	{
		std::cout << gathered << " -> " << gathered.substr(1) << "A " << tree[root] << std::endl;
		std::cout << gathered << " -> " << gathered.substr(1) << "T " << tree[root + 1] << std::endl;
		std::cout << gathered << " -> " << gathered.substr(1) << "C " << tree[root + 2] << std::endl;
		std::cout << gathered << " -> " << gathered.substr(1) << "G " << tree[root + 3] << std::endl;
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
	DisplaySubtreeRec(tree, 0, "", 0);
}

void DisplayTable(int* table, int length)
{
	for (size_t i = 0; i < length; i++)
	{
		std::cout << table[i] << " ";
	}
	std::cout << std::endl;
}