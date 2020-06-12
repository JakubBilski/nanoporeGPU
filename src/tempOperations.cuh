#include <fstream>

void AddPrecleanedChunkToTemp(std::fstream& ts, char* clearedChunk, int clearedChunkSize)
{
	ts.write(clearedChunk, clearedChunkSize);
}