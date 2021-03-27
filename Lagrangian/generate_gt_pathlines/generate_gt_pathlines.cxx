#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <omp.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkDataArray.h>
#include <vtkSmartPointer.h>
#include <vtkRectilinearGridReader.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkGradientFilter.h>
#include <vtkDataSet.h>
#include <vtkDataSetWriter.h>
#include <vtkm/Matrix.h>
#include <vtkm/Types.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include <vtkm/io/VTKDataSetWriter.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/DynamicCellSet.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/cont/ArrayPortalToIterators.h>
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/ErrorFilterExecution.h>
#include <vtkm/worklet/ParticleAdvection.h>
#include <vtkm/worklet/particleadvection/GridEvaluators.h>
#include <vtkm/worklet/particleadvection/Integrators.h>
#include <vtkm/worklet/particleadvection/Particles.h>
#include <vtkm/cont/VariantArrayHandle.h>
#include <vtkm/cont/ArrayHandleVirtual.h>
#include <vtkm/worklet/lcs/GridMetaData.h>
#include <vtkm/worklet/lcs/LagrangianStructureHelpers.h>
#include <vtkm/worklet/LagrangianStructures.h>

int main(int argc, char* argv[])
{
/* Set up flow field i
 * Read VTK binary data set
 * Move data set into VTKM object
 * Add a z-velocity for particle advection.
 * */

  vtkSmartPointer<vtkRectilinearGridReader> reader =
    vtkSmartPointer<vtkRectilinearGridReader>::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  vtkSmartPointer<vtkRectilinearGrid> mesh =
    vtkSmartPointer<vtkRectilinearGrid>::New();
  mesh = reader->GetOutput();

	int dims[3];
	dims[0] = atoi(argv[2]);
	dims[1] = atoi(argv[3]);
	dims[2] = atoi(argv[4]);

	int num_pts = mesh->GetNumberOfPoints();
	int num_pts_slice = dims[0]*dims[1];

	std::cout << "The number of grid points: " << num_pts << std::endl;
	std::cout << "The number of slice points: " << num_pts_slice << std::endl;

  vtkAbstractArray* a1 = mesh->GetPointData()->GetArray("u");
  vtkAbstractArray* a2 = mesh->GetPointData()->GetArray("v");

  vtkFloatArray* att1 = vtkFloatArray::SafeDownCast(a1);
  vtkFloatArray* att2 = vtkFloatArray::SafeDownCast(a2);

	std::cout << "Number of u : " << att1->GetNumberOfValues() << std::endl;
	std::cout << "Number of v : " << att2->GetNumberOfValues() << std::endl;

  float *xc = (float*) mesh->GetXCoordinates()->GetVoidPointer(0);
  float *yc = (float*) mesh->GetYCoordinates()->GetVoidPointer(0);
  float *zc = (float*) mesh->GetZCoordinates()->GetVoidPointer(0);

  float x_spacing = (xc[dims[0]-1] - xc[0])/(dims[0]-1);
  float y_spacing = (yc[dims[1]-1] - yc[0])/(dims[1]-1);
  float z_spacing = (zc[dims[2]-1] - zc[0])/(dims[2]-1);

	float step_size = z_spacing;

  using Vec3f = vtkm::Vec<vtkm::FloatDefault, 3>;
  vtkm::Id3 datasetDims(dims[0], dims[1], dims[2]);
  Vec3f origin3d(static_cast<vtkm::FloatDefault>(xc[0]),
                 static_cast<vtkm::FloatDefault>(yc[0]),
                 static_cast<vtkm::FloatDefault>(zc[0]));
  Vec3f spacing3d(static_cast<vtkm::FloatDefault>(x_spacing),
                  static_cast<vtkm::FloatDefault>(y_spacing),
                  static_cast<vtkm::FloatDefault>(z_spacing));
  vtkm::cont::DataSet dataset;
  vtkm::cont::DataSetBuilderUniform uniformDatasetBuilder3d;
  dataset = uniformDatasetBuilder3d.Create(datasetDims, origin3d, spacing3d);

	std::cout << "The data set vtkm" << std::endl;

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> velocity_field;
	velocity_field.Allocate(num_pts);
  
	for(int i = 0; i < num_pts; i++)
  {
    velocity_field.WritePortal().Set(i, vtkm::Vec<vtkm::FloatDefault, 3>(att1->GetTuple1(i), att2->GetTuple1(i), 1.0));
  }

	std::cout << "Created velocity field. " << std::endl;

  using FieldHandle3d = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>>;
  const vtkm::cont::DynamicCellSet& cells3d = dataset.GetCellSet();
  const vtkm::cont::CoordinateSystem& coords3d = dataset.GetCoordinateSystem();
  using GridEvalType3d = vtkm::worklet::particleadvection::GridEvaluator<FieldHandle3d>;

  using RK4Type = vtkm::worklet::particleadvection::RK4Integrator<GridEvalType3d>;
  GridEvalType3d eval_pathlines(coords3d, cells3d, velocity_field);
  RK4Type rk4(eval_pathlines, static_cast<vtkm::Float32>(step_size));

  vtkm::worklet::ParticleAdvection particleadvection;
  vtkm::worklet::ParticleAdvectionResult res_rk4;
	
	std::cout << "Created particle advection objects. " << std::endl;

	std::vector <vtkm::Vec3f_32> pointCoordinates;		
	std::vector<vtkm::UInt8> shapes;
	std::vector<vtkm::IdComponent> numIndices;
	std::vector<vtkm::Id> connectivity;
	std::vector<vtkm::Int32> ids;
	
	int num_seeds = atoi(argv[5]);
	
	vtkm::cont::ArrayHandle<vtkm::Particle> seed_set;
	seed_set.Allocate(num_seeds);
	
	std::ifstream seed_stream(argv[6]);
	float x, y;
	int seed_counter = 0;

	while(seed_stream >> x)
	{
		seed_stream >> y;
		seed_set.WritePortal().Set(seed_counter, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(x), static_cast<vtkm::FloatDefault>(y), static_cast<vtkm::FloatDefault>(0.0)), seed_counter));
		seed_counter++;
		pointCoordinates.push_back(vtkm::Vec3f_32(x,y,0.0f));
		ids.push_back(seed_counter);
	}
	
	std::cout << "Created seed list. " << seed_counter << std::endl;
	
	/* Loop over seed points and update the current locations to form polylines */ 

	// Double Gyre has 1001 dimensions in time direction. We perform 100 steps 10 times, to track the location of the particle over time. At the end of each cycle store the location of the particle. Give this some thought. 

	int iterations = atoi(argv[8]);
	int num_steps_iter = atoi(argv[9]);

	for(int iter = 0; iter < iterations; iter++)  /* Number of locations to store is hardcoded */
	{
		res_rk4 = particleadvection.Run(rk4, seed_set, num_steps_iter); // Number of steps hardcoded.
		auto current_set = res_rk4.Particles;	
		
		for(int p = 0; p < num_seeds; p++)
		{
			auto pt = current_set.ReadPortal().Get(p).Pos;
			pointCoordinates.push_back(vtkm::Vec3f_32(pt[0], pt[1], pt[2]));	
			seed_set.WritePortal().Set(p, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(pt[0]), 
			static_cast<vtkm::FloatDefault>(pt[1]), static_cast<vtkm::FloatDefault>(pt[2])), p));
		}
	}

	for(int p = 0; p < num_seeds; p++)
	{
		shapes.push_back(vtkm::CELL_SHAPE_POLY_LINE);
		numIndices.push_back(1+iterations);  /* Number of points along the pathline is hardcoded */
		for(int c = 0; c < (1+iterations); c++)
		{
			connectivity.push_back((num_seeds*c)+p);
		}
	}
	/* Write out data */ 
    vtkm::cont::DataSetBuilderExplicit dataSetBuilder;

    vtkm::cont::DataSet output = dataSetBuilder.Create(pointCoordinates, shapes, numIndices, connectivity);
		output.AddCellField("ID", ids);
    std::stringstream s;
    vtkm::io::VTKDataSetWriter wrt(argv[7]);
    wrt.WriteDataSet(output);
}
