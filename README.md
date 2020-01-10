
### Parallel length computation of polyline created from coordinates on earth

## How to run on visual studio:
Import DVA336_project.cpp into a new project.

### Enable openMP:
1. Open the project's Property Pages dialog box.

2. Expand the Configuration Properties > C/C++ > Language property page.

3. Change the OpenMP Support property into Yes.

4. Expand the Configuration Properties > C/C++ > Command Line property page.
5. Add /Zc:twoPhase- to the Additional Options property and then choose OK.

### Note:
This program requires support for AVX2 (if the computer's processor isn't older than 2015 it should support it)