#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
	srand(time(NULL));
	int num_seeds = atoi(argv[1]);
	int bres[2]; // boundary resolution. Place some number of points along the boundary. 
	bres[0] = atoi(argv[2]);	
	bres[1] = atoi(argv[3]);

	int total_boundary = bres[0]*2 + bres[1]*2 - 4;
	int total_random = num_seeds - total_boundary;	

	float xmin = 0.0;
	float xmax = 2.0;
	float ymin = 0.0;
	float ymax = 1.0;

	float xspace = (xmax-xmin)/(bres[0]-1);
	float yspace = (ymax-ymin)/(bres[1]-1);

	std::ofstream seedfile;
	seedfile.open("seeds.txt");

	// Lower boundary. Fixed ymin
	for(int i = 0; i < bres[0]; i++)
	{
		seedfile << (xmin+(i*xspace)) << " " << ymin << "\n";
	}
	// Top boundary. Fixed ymax
	for(int i = 0; i < bres[0]; i++)
	{
		seedfile << (xmin+(i*xspace)) << " " << ymax << "\n";
	}
	// Left boundary. Fixed xmin. Start at i = 1, and end 1 early to avoid redundant seeds at corners.
	for(int i = 1; i < bres[1]-1; i++)
	{
		seedfile << xmin << " " << (ymin+(i*yspace)) << "\n";
	}
	// Right boundary. Fixed xmax. Start at i = 1, and end 1 early to avoid redundant seeds at corners.
	for(int i = 1; i < bres[1]-1; i++)
	{
		seedfile << xmax << " " << (ymin+(i*yspace)) << "\n";
	}

	for(int i = 0; i < total_random; i++)
	{
		float x, y;
		x = (rand()%20000)/10000.0;
		y = (rand()%10000)/10000.0;
	
		seedfile << x << " " << y << "\n";
	}
	seedfile.close();
}
