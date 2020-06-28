#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define kernelErrchk() { kernelAssert(__FILE__, __LINE__); }
inline void kernelAssert(const char *file, int line, bool abort = true)
{
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
	{
		fprintf(stderr, "kernelAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void assertOpenFile(const std::fstream & fs, char* fileName)
{
	if (!fs.is_open())
	{
		printf("Unable to open ");
		printf(fileName);
		printf("\nClosing\n");
		exit(0);
	}
}

