#include <vtkm/io/VTKDataSetReader.h>
#include <vtkm/cont/ArrayHandleVirtualCoordinates.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkCleanPolyData.h>
#include <vtkDelaunay2D.h>
#include <vtkDelaunay3D.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCell.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkIdList.h>
#include <vtkProbeFilter.h>
#include <vtkAppendFilter.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  vtkm::cont::DataSet input_flowmap;
  vtkm::io::VTKDataSetReader fm_reader(argv[1]);
  input_flowmap = fm_reader.ReadDataSet(); 
  vtkm::cont::ArrayHandleVirtualCoordinates coordinatesystem = input_flowmap.GetCoordinateSystem().GetData();
  int num_basis_points = coordinatesystem.GetNumberOfValues(); 
  auto coords_portal = input_flowmap.GetCoordinateSystem().GetData();
  int num_basis_total = num_basis_points/2;
	int iterations = atoi(argv[2]);	
	int num_basis_map = num_basis_total/iterations;

	std::cout << "Number of basis in total: " << num_basis_total << std::endl;
	std::cout << "Number of basis per flow map: " << num_basis_total/iterations << std::endl;
	
	std::vector<float> bx0, by0, bx1, by1;
	for(int p = 0; p < num_basis_total; p++)
	{
		int index0 = p;
		int index1 = p + num_basis_total;
		auto pt0 = coords_portal.ReadPortal().Get(index0);
		auto pt1 = coords_portal.ReadPortal().Get(index1);
		bx0.push_back(pt0[0]);
		by0.push_back(pt0[1]);
		bx1.push_back(pt1[0]);
		by1.push_back(pt1[1]);
	}

	std::ifstream seed_stream(argv[3]);	
	int num_seeds = atoi(argv[4]);
	std::vector<float> sx0, sy0;
	float x, y;
	while(seed_stream >> x)
	{
		seed_stream >> y;
		sx0.push_back(x);
		sy0.push_back(y);
	}
	
	vtkm::cont::DataSet gt_pathlines;
  vtkm::io::VTKDataSetReader pathline_reader(argv[5]);
	gt_pathlines = pathline_reader.ReadDataSet();
  auto pathline_pts = gt_pathlines.GetCoordinateSystem().GetData();	

	float threshold = atof(argv[6]);
	float aedr[num_seeds] = {0};
	int step_count[num_seeds] = {0};
	int valid[num_seeds] = {1}; // Assume all particles are valid initially.

	// Using VTK Delaunay + Probe 
	for(int iter = 0; iter < iterations; iter++)
	{
		// Identify the end location of sample locations. 	
		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
		vtkSmartPointer<vtkFloatArray> xdisp = vtkSmartPointer<vtkFloatArray>::New();	
		vtkSmartPointer<vtkFloatArray> ydisp = vtkSmartPointer<vtkFloatArray>::New();	

		xdisp->SetNumberOfValues(num_basis_map);
		ydisp->SetNumberOfValues(num_basis_map);

		xdisp->SetName("xdisp");
		ydisp->SetName("ydisp");

		for(int b = 0; b < num_basis_map; b++)
		{
			int index = (iter*num_basis_map) + b;
			points->InsertNextPoint(bx0[index], by0[index], 0.0);
			xdisp->SetValue(b, bx1[index]);
			ydisp->SetValue(b, by1[index]);
		}
		
		vtkSmartPointer<vtkUnstructuredGrid> basisMesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
		basisMesh->SetPoints(points);
		basisMesh->GetPointData()->AddArray(xdisp);
		basisMesh->GetPointData()->AddArray(ydisp);
	
		vtkSmartPointer<vtkDelaunay2D> triangulation = vtkSmartPointer<vtkDelaunay2D>::New();
		triangulation->SetInputData(basisMesh);
		triangulation->Update();
	
		vtkSmartPointer<vtkUnstructuredGrid> triangulated_basis = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkAppendFilter> appendFilter = vtkSmartPointer<vtkAppendFilter>::New();
		appendFilter->AddInputData(triangulation->GetOutput());
		appendFilter->Update();
		triangulated_basis->ShallowCopy(appendFilter->GetOutput());

		vtkSmartPointer<vtkPoints> particles = vtkSmartPointer<vtkPoints>::New();
		for(int p = 0; p < num_seeds; p++)
		{
			particles->InsertNextPoint(sx0[p], sy0[p], 0.0);
		}

		vtkSmartPointer<vtkUnstructuredGrid> particleMesh = vtkSmartPointer<vtkUnstructuredGrid>::New();	
		particleMesh->SetPoints(particles);

		vtkSmartPointer<vtkProbeFilter> probe = vtkSmartPointer<vtkProbeFilter>::New();
		probe->SetSourceData(triangulated_basis);
		probe->SetInputData(particleMesh);
		probe->Update();

		vtkSmartPointer<vtkIntArray> validPts = vtkSmartPointer<vtkIntArray>::New();
		validPts->DeepCopy(probe->GetOutput()->GetPointData()->GetArray(probe->GetValidPointMaskArrayName()));
				
		vtkSmartPointer<vtkFloatArray> xlocation = vtkFloatArray::SafeDownCast(probe->GetOutput()->GetPointData()->GetArray("xdisp"));
		vtkSmartPointer<vtkFloatArray> ylocation = vtkFloatArray::SafeDownCast(probe->GetOutput()->GetPointData()->GetArray("ydisp"));

		int valid_count = 0;
		for(int p = 0; p < num_seeds; p++)
		{
			if(validPts->GetValue(p) == 1)
			{
				step_count[p] += 1;
				float r_pt[2];
				r_pt[0] = xlocation->GetValue(p);
				r_pt[1] = ylocation->GetValue(p);
			
				int index = num_seeds*(iter+1) + p;
				auto gt_pt = pathline_pts.ReadPortal().Get(index);				
					
				float distance = sqrt(pow(gt_pt[0]-r_pt[0],2.0) + pow(gt_pt[1]-r_pt[1],2.0));
				if(distance < threshold)
				{
					aedr[p] += (distance/threshold);
				}
				else
				{
					aedr[p] += 1.0;
				}
				valid_count++;
			
				sx0[p] = r_pt[0];
				sy0[p] = r_pt[1];
			}		
			else
			{
				valid[p] = 0;
			}
		}		
		std::cout << "Number of points valid after iteration: " << iter << " is " << valid_count << std::endl;
	}
}
