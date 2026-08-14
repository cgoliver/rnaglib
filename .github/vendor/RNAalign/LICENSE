===============================================================================
   The RNA-align program identifies the best alignment of two RNA (or DNA)
   structures with the highest TM-score. Please report suggestions and issues
   to yangzhanglab@umich.edu

   Reference to cite:
   Sha Gong, Chengxin Zhang, Yang Zhang. RNA-align: quick and accurate
   alignment of RNA 3D structures based on size-independent TM-scoreRNA.
   (2019) Bioinformatics.

   DISCLAIMER:
     Permission to use, copy, modify, and distribute this program for any
     purpose, with or without fee, is hereby granted, provided that the
     notices on the head, the reference information, and this copyright
     notice appear in all copies or substantial portions of the Software.
     It is provided "as is" without express or implied warranty.

=========================
 How to install RNA-align
=========================
To compile the program in your Linux computer, simply enter

    make

or

    g++ -static -O3 -ffast-math -lm -o RNAalign RNAalign.cpp

The "-static" flag should be removed on Mac OS, which does not support
building static executables.

RNA-align natively supports PDB (.pdb) and PDBx/mmCIF (.cif) format input.
If "zcat" or "bzcat" command is installed on your system, RNA-align can
also read gzip compressed file (.pdb.gz or .cif.gz) or bzip2 compressed
file (.pdb.bz2 or .cif.bz2).

=====================
 How to use RNA-align
=====================
Briefly, to alignment two single-chain structures (Chain_1.pdb and
Chain_2.pdb), enter the following:

     ./RNAalign Chain_1.pdb Chain_2.pdb
  
You can run the program without arguments to obtain a brief instruction.
Full document for all available options can be obained by:

     ./RNAalign -h

02/01/2019
