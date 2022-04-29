# nanoporeGPU
Examination of the possibilities of performing an Oxford Nanopore sequence alingment on a typical GPU memory, using de Bruijn Graphs and de novo assembly, with CUDA

Despite using some memory-saving techniques introduced in LORMa, memory requirements were too high to support k values higher than 15. We concluded that running sequence alignment on a typical GPU would require developing a new alignment method. Because it was out of scope of the project, we abandoned the idea.

See the report:
https://github.com/JakubBilski/nanoporeGPU/blob/master/BrojaczBilskiReport.pdf

![obraz](https://user-images.githubusercontent.com/47048420/165967215-6608f19a-af92-4aa5-9274-16d5ce26c705.png)
