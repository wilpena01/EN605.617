This is an explanation about the report of assignment 3.

There should be 9 files here:
	assignment.cu
	Makefile
	TwoAdditionNumberOfThreads.txt
	TwoAdditionBlockSizes.txt
	baseOutput.txt
	FirstHalfOfProblem.txt
	branchingvsNonBranching.xslx
	Stretchproblem.txt
	README.txt
	

README.txt:
The explanation of all the documentation (This file).


FirstHalfOfProblem.txt:	
This is the output of the first half of the assignment where you 
will be able to find the output of all the mathematical calculations
done in the kernel using a lot of threads with one block. 


baseOutput.txt:
This is the output used as reference for the second half of the assignment.
This will be used to see the diference when we increase just two number of threads
or two numbers of blocks.


TwoAdditionNumberOfThreads.txt:
This is the output when I used two additional numbers of threads.


TwoAdditionBlockSizes.txt:
This is the output when I used two additional block size.


Makefile:
This is the make file used to create the assignment.exe.


assignment.cu:
is the main cuda file.


branchingvsNonBranching.xslx:
This file shows the difference in time between two function doing the same time
while one of them is branching half the time and the other do not.


Stretchproblem.txt:
Say the some of the good and bad things the stretch problem have.