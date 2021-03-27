#include <vtkSmartPointer.h>
#include <vtkRectilinearGridReader.h>
#include <vtkRectilinearGrid.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkAbstractArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkDataSetWriter.h>
#include "DoubleGyrefield.cxx"
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace std;

#define FIELD DoubleGyrefield 

int main(int argc, char* argv[])
{

FIELD field;

int num_cycles = atoi(argv[1]);
int interval = atoi(argv[2]);
int cycle_prev = 0;
int cycle_next = cycle_prev + interval;

int dims[2];
dims[0] = atoi(argv[3]);
dims[1] = atoi(argv[4]);
int num_seeds = dims[0]*dims[1];

float bounds[4];
bounds[0] = atof(argv[5]);
bounds[1] = atof(argv[6]);
bounds[2] = atof(argv[7]);
bounds[3] = atof(argv[8]);

float threshold = atof(argv[9]);

vector<float> x_current, y_current;
vector<float> x_gt, y_gt;
vector<float> error;
vector<float> aedr;

for(int i = 0; i < num_seeds; i++)
	aedr.push_back(0.0);


/* Define seed starting locations */


float x_spacing, y_spacing;
x_spacing = (bounds[1]-bounds[0])/(dims[0]-1);
y_spacing = (bounds[1]-bounds[0])/(dims[0]-1);

for(int i = 0; i < dims[1]; i++)
{
	for(int j = 0; j < dims[0]; j++)
	{
		float x, y;
		x = bounds[0] + x_spacing*j;	
		y = bounds[2] + y_spacing*i;	
		
		x_current.push_back(x);
		y_current.push_back(y);
		x_gt.push_back(x);
		y_gt.push_back(y);
	}
}

float h = 0.01; // step size

for(int c = 0; c < num_cycles; c++)
{
	if(c == cycle_next)
	{
		cycle_prev = c;
		cycle_next = cycle_prev + interval;	
	}	

	#pragma omp parallel for
	for(int n = 0; n < num_seeds; n++)
	{
		float vel_gt[2], vel_prev[2], vel_next[2], vel_eul[2];
		float loc_gt[2], loc_eul[2];
		float t = c*h; // this is the current time.

		// Ground truth computation
		loc_gt[0] = x_gt[n];
		loc_gt[1] = y_gt[n];

		field.calculateVelocity(loc_gt, t, vel_gt);

		x_gt[n] = loc_gt[0] + h*vel_gt[0];
		y_gt[n] = loc_gt[1] + h*vel_gt[1];
	

		// Subsampled Eulerian computation
		
		float t_prev, t_next;
		t_prev = cycle_prev*h;
		t_next = cycle_next*h;
		
		loc_eul[0] = x_current[n];
		loc_eul[1] = y_current[n];

		field.calculateVelocity(loc_eul, t_prev, vel_prev);
		field.calculateVelocity(loc_eul, t_next, vel_next);

		float d = (t - t_prev)/(t_next - t_prev);

		vel_eul[0] = vel_prev[0] + d*(vel_next[0] - vel_prev[0]);
		vel_eul[1] = vel_prev[1] + d*(vel_next[1] - vel_prev[1]);

		x_current[n] = loc_eul[0] + h*vel_eul[0];
		y_current[n] = loc_eul[1] + h*vel_eul[1];

		
		float diff = sqrt(pow(x_gt[n]-x_current[n],2.0) + pow(y_gt[n]-y_current[n],2.0));
		
		if(diff < threshold)
		{
			float flag = diff/threshold;
			aedr[n] += flag;
		}
		else
		{
			aedr[n] += 1;
		}
		
	}
}

float min = 999999999.0;
float max = 0.0;
float sum = 0.0;
float avg;

vtkSmartPointer<vtkFloatArray> errorArray = vtkSmartPointer<vtkFloatArray>::New();
errorArray->SetName("endpt");
errorArray->SetNumberOfComponents(1);

vtkSmartPointer<vtkFloatArray> aedrArray = vtkSmartPointer<vtkFloatArray>::New();
aedrArray->SetName("aedr");
aedrArray->SetNumberOfComponents(1);

for(int n = 0; n < num_seeds; n++)
{
	float pt1[2], pt2[2];
	
	pt1[0] = x_gt[n];
	pt1[1] = x_gt[n];
	
	pt2[0] = x_current[n];
	pt2[1] = x_current[n];
	
	float dist = sqrt(pow(pt1[0] - pt2[0], 2.0) + pow(pt1[1]-pt2[1], 2.0));
	errorArray->InsertNextTuple1(dist);

	float sim = aedr[n]/num_cycles;	
	aedrArray->InsertNextTuple1(sim);
 
	if(dist < min)
		min = dist;

	if(dist > max)
		max = dist;

	sum += dist;
}

	avg = sum/num_seeds;

	cout << "Stats for interval = " << interval << ": " << endl;
	cout << "Average error: " << avg << endl;
	cout << "Min error: " << min << endl;
	cout << "Max error: " << max << endl;

  /* Write binary_image data as a scalar field to a vtk data set and output it. */
  vtkSmartPointer<vtkDoubleArray> xCoords =
    vtkSmartPointer<vtkDoubleArray>::New();
  vtkSmartPointer<vtkDoubleArray> yCoords =
    vtkSmartPointer<vtkDoubleArray>::New();
  vtkSmartPointer<vtkDoubleArray> zCoords =
    vtkSmartPointer<vtkDoubleArray>::New();

  for(int i = 0; i < dims[0]; i++)
  xCoords->InsertNextValue(bounds[0] + x_spacing*i);

  for(int i = 0; i < dims[1]; i++)
  yCoords->InsertNextValue(bounds[2] + y_spacing*i);

	zCoords->InsertNextValue(0.0);

  vtkSmartPointer<vtkDataSetWriter> writer =
    vtkSmartPointer<vtkDataSetWriter>::New();

  vtkSmartPointer<vtkRectilinearGrid> outputGrid =
      vtkSmartPointer<vtkRectilinearGrid>::New();

  outputGrid->SetDimensions(dims[0], dims[1], 1);
  outputGrid->SetXCoordinates(xCoords);
  outputGrid->SetYCoordinates(yCoords);

  outputGrid->GetPointData()->AddArray(errorArray);
  outputGrid->GetPointData()->AddArray(aedrArray);

	stringstream s;
	s << "ErrorPlot_" << interval << "_" << num_cycles << ".vtk";

  writer->SetFileName(s.str().c_str());
  writer->SetInputData(outputGrid);
  writer->SetFileTypeToASCII();
  writer->Write();

}

