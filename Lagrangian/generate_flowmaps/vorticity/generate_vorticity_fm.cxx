#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <time.h>
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
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandleCast.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/worklet/Magnitude.h>


#define CMWC_CYCLE 4096   // as George Marsaglia recommends
#define CMWC_C_MAX 809430660  // as George Marsaglia recommends

struct cmwc_state {
  uint32_t Q[CMWC_CYCLE];
  uint32_t c; // must be limited with CMWC_C_MAX
  unsigned i;
};

uint32_t rand32(void)
{
  uint32_t result = rand();
  return result << 16 | rand();
}

void initCMWC(struct cmwc_state *state, unsigned int seed)
{
  srand(seed);
  for (int i = 0; i < CMWC_CYCLE; i++)
    state->Q[i] = rand32();
  do
    state->c = rand32();
  while (state->c >= CMWC_C_MAX);
  state->i = CMWC_CYCLE - 1;
}

uint32_t randCMWC(struct cmwc_state *state)  //EDITED parameter *state was missing
{
  uint64_t const a = 18782; // as Marsaglia recommends
  uint32_t const m = 0xfffffffe;  // as Marsaglia recommends
  uint64_t t;
  uint32_t x;

  state->i = (state->i + 1) & (CMWC_CYCLE - 1);
  t = a * state->Q[state->i] + state->c;
  /* Let c = t / 0xffffffff, x = t mod 0xffffffff */
  state->c = t >> 32;
  x = t + state->c;
  if (x < state->c) {
    x++;
    state->c++;
  }
  return state->Q[state->i] = m - x;
}

namespace worklets
{

  class RandomSeedInCell : public vtkm::worklet::WorkletMapField
  {
    public: 
      typedef void ControlSignature(FieldIn, FieldIn, FieldOut);
      typedef void ExecutionSignature(_1, _2, _3);
      
      //VTKM_CONT RandomSeedInCell(std::vector<vtkm::Float64> x, std::vector<vtkm::Float64> y, int xdim, int ydim, float stime)
      VTKM_CONT RandomSeedInCell(float* x, float* y, int xdim, int ydim, float stime)
      {
        xC = x;
        yC = y;
 				
				t = stime; 

        dims[0] = xdim;
        dims[1] = ydim;
      }
      
      template<typename CellId, typename RandomPortal, typename SeedParticle>
      VTKM_EXEC void operator()(const CellId &index, const RandomPortal &random, SeedParticle &seed) const
      {
        int idx[2];
        idx[0] = index % (dims[0]-1);
        idx[1] = index / (dims[0]-1);
        
        double bbox[4];
        bbox[0] = xC[idx[0]];
        bbox[1] = xC[idx[0]+1];
        bbox[2] = yC[idx[1]];
        bbox[3] = yC[idx[1]+1];
        
        double rangex = bbox[1] - bbox[0];
        double rangey = bbox[3] - bbox[2];

        int rx = rangex * 10000; 
        int ry = rangey * 10000;
        double x,y; 

        x = (random[0]%rx)/10000.0 + bbox[0];
        y = (random[1]%ry)/10000.0 + bbox[2];

        seed.Pos[0] = x; 
        seed.Pos[1] = y; 
				seed.Pos[2] = t;
      }

    private:
    //  std::vector<vtkm::Float64> xC, yC;
			float *xC, *yC;
			float t;
      int dims[2];
  };


  class PointFieldToCellAverageField : public vtkm::worklet::WorkletMapField
  {
    public: 
      typedef void ControlSignature(FieldOut, WholeArrayIn);
      typedef void ExecutionSignature(_1, _2, WorkIndex);
      
      VTKM_CONT PointFieldToCellAverageField(int xdim, int ydim)
      {
        dims[0] = xdim;
        dims[1] = ydim;
      }
      
      template<typename CellValue, typename FeaturePortal>
      VTKM_EXEC void operator()(CellValue &cell, const FeaturePortal &feature, const vtkm::Id &index) const
      {
        int idx[3];
        idx[0] = index % (dims[0]-1);
        idx[1] = index / (dims[0]-1);
        double sample_sum = 0;
        sample_sum += feature.Get(idx[1]*dims[0] + idx[0]);
        sample_sum += feature.Get(idx[1]*dims[0] + (idx[0]+1));
        sample_sum += feature.Get((idx[1]+1)*dims[0] + idx[0]);
        sample_sum += feature.Get((idx[1]+1)*dims[0] + (idx[0]+1));
        double average_sum = sample_sum/4.0;
        cell = average_sum;
      }

    private:
      int dims[2];
  };

  class ExponentWorklet : public vtkm::worklet::WorkletMapField
  {
    public: 
      typedef void ControlSignature(FieldInOut);
      typedef void ExecutionSignature(_1);
      
      VTKM_CONT ExponentWorklet(double e)
      {
        exponent = e;
      }
      
      template<typename FeaturePortal>
      VTKM_EXEC void operator()(FeaturePortal &value) const
      {
        value = pow(value, exponent); 
      }

    private:
      double exponent;
  };

  class AbsoluteValueWorklet : public vtkm::worklet::WorkletMapField
  {
    public: 
      typedef void ControlSignature(FieldInOut);
      typedef void ExecutionSignature(_1);
      
      VTKM_CONT AbsoluteValueWorklet()
      {}
      
      template<typename FeaturePortal>
      VTKM_EXEC void operator()(FeaturePortal &value) const
      {
        value = abs(value); 
      }
  };


  class MinimumValueWorklet : public vtkm::worklet::WorkletMapField
  {
    public: 
      typedef void ControlSignature(FieldInOut);
      typedef void ExecutionSignature(_1);
      
      VTKM_CONT MinimumValueWorklet(double v)
      {
        min = v;
      }
      
      template<typename FeaturePortal>
      VTKM_EXEC void operator()(FeaturePortal &value) const
      {
        if(value < min)
          value = min;
        else
          value = value; 
      }

    private:
      double min;
  };
}


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
	int num_cells_slice = (dims[0]-1)*(dims[1]-1);

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

	float step_size = 0.01;

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
	std::vector<vtkm::FloatDefault> start_time;

/* Initialize seeds */

	int sdims[2];
	sdims[0] = atoi(argv[5]);	
	sdims[1] = atoi(argv[6]);	
	int iterations = atoi(argv[8]);
	int num_steps_iter = atoi(argv[9]);
	float offset = atof(argv[10]);
	float exponent = atof(argv[11]);	
	float minimum = atof(argv[12]);	
	float inflate = atof(argv[13]);	
	int num_seeds_slice = sdims[0]*sdims[1];
	int num_seeds_total = sdims[0]*sdims[1]*iterations;
  
	vtkm::cont::ArrayHandle<vtkm::Particle> seed_set;
	seed_set.Allocate(num_seeds_total);
	
	float x, y, s_time;
	int seed_counter = 0;

  float* vec_x, *vec_y, *vec_z;
  vec_x = (float*)malloc(sizeof(float)*num_pts_slice);
  vec_y = (float*)malloc(sizeof(float)*num_pts_slice);
  vec_z = (float*)malloc(sizeof(float)*num_pts_slice);

	/* Seed placement steps.
 * Extract field at grid points
 * Sum the values and create a cell field.
 * Scan the values.
 * Generate a random number. 
 */
  
/*	vtkm::Id3 vortDims(dims[0], dims[1], 1);
  Vec3f origin3d(static_cast<vtkm::FloatDefault>(xc[0]),
                 static_cast<vtkm::FloatDefault>(yc[0]),
                 static_cast<vtkm::FloatDefault>(zc[0]));
  Vec3f spacing3d(static_cast<vtkm::FloatDefault>(x_spacing),
                  static_cast<vtkm::FloatDefault>(y_spacing),
                  static_cast<vtkm::FloatDefault>(z_spacing));
  vtkm::cont::DataSet vortOutput;
  vortOutput = uniformDatasetBuilder3d.Create(vortDims, origin3d, spacing3d);
*/

	for(int k = 0; k < iterations; k++)
	{
		// Get the 2D slice at a specific value of Z. 0, 100, 200, 300, .. num_steps_iter*i
  	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  	vtkSmartPointer<vtkDoubleArray> vel_array = vtkSmartPointer<vtkDoubleArray>::New();
  	vel_array->SetNumberOfComponents(3);
  	vel_array->SetNumberOfTuples(num_pts_slice);
  	vel_array->SetName("velocity");
		
		int zslice = k*num_steps_iter;
	  for(int j = 0; j < dims[1]; j++)
  	{
    	for(int i = 0; i < dims[0]; i++)
    	{
    	  int index1 = zslice*dims[1]*dims[0] + j*dims[0] + i;
  	    int index2 = j*dims[0] + i;
	      double vec[3];
      	vec[0] = att1->GetTuple1(index1);
    	  vec[1] = att2->GetTuple1(index1);
  	    vec[2] = 0.0;
	      points->InsertNextPoint(xc[i], yc[j], 0);
      	vel_array->InsertTuple(index2, vec);
    	  vec_x[index2] = vec[0];
  	    vec_y[index2] = vec[1];
	      vec_z[index2] = vec[2];
    	}
  	}

		vtkSmartPointer<vtkStructuredGrid> slice = vtkSmartPointer<vtkStructuredGrid>::New();
  	slice->SetDimensions(dims[0], dims[1], 1);
  	slice->SetPoints(points);
  	slice->GetPointData()->AddArray(vel_array);

  	vtkSmartPointer<vtkGradientFilter> gradient = vtkSmartPointer<vtkGradientFilter>::New();
  	gradient->SetInputData(slice);
  	gradient->SetInputScalars(vtkDataObject::FIELD_ASSOCIATION_POINTS, "velocity");
  	gradient->SetComputeVorticity(1);
  	gradient->SetComputeQCriterion(0);
  	gradient->SetComputeDivergence(0);
	  gradient->Update();
	
	  vtkDoubleArray *vorticity_field = vtkDoubleArray::SafeDownCast(gradient->GetOutput()->GetPointData()->GetArray("Vorticity"));			

    vtkm::cont::ArrayHandle<vtkm::FloatDefault> featurePointField;
		featurePointField.Allocate(num_pts_slice);
    vtkm::cont::ArrayHandle<vtkm::FloatDefault> featureCellField;
		featureCellField.Allocate(num_cells_slice);
		vtkm::cont::ArrayHandle<vtkm::Particle> SeedArray;
		SeedArray.Allocate(num_seeds_slice);

  	for(int i = 0; i < num_pts_slice; i++)
  	{	 
   		double pt_vort[3];
    	vorticity_field->GetTuple(i, pt_vort);  
    	float val = sqrt(pow(pt_vort[0],2.0) + pow(pt_vort[1],2.0) + pow(pt_vort[2], 2.0)) * inflate;
			featurePointField.WritePortal().Set(i, static_cast<vtkm::FloatDefault>(val));
  	}		
	
		vtkm::worklet::DispatcherMapField<worklets::PointFieldToCellAverageField>(worklets::PointFieldToCellAverageField(dims[0], dims[1])).Invoke(featureCellField, featurePointField);	

    vtkm::worklet::DispatcherMapField<worklets::ExponentWorklet>(worklets::ExponentWorklet(exponent)).Invoke(featureCellField);
//    vtkm::worklet::DispatcherMapField<worklets::AbsoluteValueWorklet>(worklets::AbsoluteValueWorklet()).Invoke(featureCellField);
    vtkm::worklet::DispatcherMapField<worklets::MinimumValueWorklet>(worklets::MinimumValueWorklet(minimum)).Invoke(featureCellField);	

    vtkm::cont::ArrayHandle<vtkm::FloatDefault> scan;
    scan.Allocate(num_cells_slice);
      
    vtkm::cont::Algorithm::ScanInclusive(featureCellField, scan);

		float max_val = scan.ReadPortal().Get(scan.GetNumberOfValues()-1);
		int range = max_val;

    struct cmwc_state cmwc1, cmwc2;
    unsigned int seed1 = time(NULL);
    initCMWC(&cmwc1, seed1);
    unsigned int seed2 = time(NULL);
    initCMWC(&cmwc2, seed2);

    vtkm::cont::ArrayHandle<vtkm::Float64> randomNumForCells;
    randomNumForCells.Allocate(num_seeds_slice);

    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32, 2>> randomNumForSeeds;
    randomNumForSeeds.Allocate(num_seeds_slice);

    for(int i = 0; i < num_seeds_slice; i++)
    {
      vtkm::Float64 r = (randCMWC(&cmwc1)%range)*1.0d;
      randomNumForCells.WritePortal().Set(i,r);

      vtkm::Int32 a = randCMWC(&cmwc2);
      vtkm::Int32 b = randCMWC(&cmwc2);
      randomNumForSeeds.WritePortal().Set(i, vtkm::Vec<vtkm::Int32, 2>(a, b));
    }

    vtkm::cont::ArrayHandle<vtkm::Id> cellIds;
    cellIds.Allocate(num_seeds_slice);

    vtkm::cont::Algorithm::UpperBounds(scan,randomNumForCells,cellIds);
		float seed_time = k*1.0f;
    vtkm::worklet::DispatcherMapField<worklets::RandomSeedInCell>(worklets::RandomSeedInCell(xc, yc, dims[0], dims[1], seed_time)).Invoke(cellIds, randomNumForSeeds, SeedArray);

		for(int i = 0; i < num_seeds_slice; i++)
		{
			float x, y;
			auto pt = SeedArray.ReadPortal().Get(i).Pos;
			x = pt[0];
			y = pt[1];
			seed_set.WritePortal().Set(seed_counter, vtkm::Particle(Vec3f(static_cast<vtkm::FloatDefault>(x), static_cast<vtkm::FloatDefault>(y), static_cast<vtkm::FloatDefault>(seed_time)), seed_counter));
			pointCoordinates.push_back(vtkm::Vec3f_32(x,y,seed_time));
			ids.push_back(seed_counter);
			start_time.push_back(seed_time);
			seed_counter++;
		}
	}

	std::cout << "Created seed list. " << seed_counter << std::endl;
	
	/* Loop over seed points and update the current locations to form polylines */ 

	res_rk4 = particleadvection.Run(rk4, seed_set, num_steps_iter); // Number of steps hardcoded.
	auto current_set = res_rk4.Particles;	
		
	for(int p = 0; p < num_seeds_total; p++)
	{
		auto pt = current_set.ReadPortal().Get(p).Pos;
		pointCoordinates.push_back(vtkm::Vec3f_32(pt[0], pt[1], pt[2]));	// This is a problem ** Some particles are not being advected. 
	}

	for(int p = 0; p < num_seeds_total; p++)
	{
		shapes.push_back(vtkm::CELL_SHAPE_LINE);
		numIndices.push_back(2);  // Number of points along the pathline is hardcoded //
		for(int c = 0; c < 2; c++)
		{
			connectivity.push_back((num_seeds_total*c)+p);
		}
	}
	// Write out data // 
    vtkm::cont::DataSetBuilderExplicit dataSetBuilder;

    vtkm::cont::DataSet output = dataSetBuilder.Create(pointCoordinates, shapes, numIndices, connectivity);
		output.AddCellField("ID", ids);
		output.AddCellField("START_TIME", start_time);
    std::stringstream s;
    vtkm::io::VTKDataSetWriter wrt(argv[7]);
    wrt.WriteDataSet(output);
}
