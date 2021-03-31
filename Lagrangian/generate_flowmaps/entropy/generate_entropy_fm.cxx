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

#define PI 3.14159265
#define DIM_THETA 60
#define DIM_PHI 1

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

float calculateEntropy(int *bins, int num_bins)
{
  float probability[num_bins];
  int total_sum = 0;
  for(int i = 0; i < num_bins; i++)
  {
    total_sum += bins[i];
  }

  for(int i = 0; i < num_bins; i++)
  {
    probability[i] = bins[i]/(total_sum*1.0);
  }

  float entropy = 0.0;

  for(int i = 0; i < num_bins; i++)
  {
    if(probability[i] > 0.0)
      entropy -= (probability[i]*log2(probability[i]));
  }

  return entropy;
}

void estimateDistribution(int *bins, int num_bins, float* x, float* y, float* z, int N)
{ 
  /* There are DIM_THETA * DIM_PHI bins */
  
  double theta_range = 360.0/DIM_THETA; // 12
  double phi_range = 180.0/DIM_PHI; // 12
  
  /* BINS are ordered by increasing PHI
 *  // 0 [0,0] to [theta_range + 0 , phi_range + 0] .. 1 [0, phi_range + 0] to [theta_range + 0, phi_range*2 + 0]
 *  // Theta_index = theta/theta_range; */
  
  int num_samples = N*N;
  for(int i = 0; i < num_samples; i++)
  { 
    /* All values of theta and phi are between:
 *  // 0 < Theta < 360
 *  // 0 < Phi < 180 */
    
    double radius, theta, phi;
    radius = sqrt((x[i]*x[i]) + (y[i]*y[i]) + (z[i]*z[i]));
    if(radius > 0)
    { 
      theta = (atan2(y[i],x[i]))*(180/PI);
      phi = (acos(z[i]/radius))*(180/PI);
      if(theta < 0) 
        theta = 360 + theta; /* theta value is negative, i.e., -90 is the same as 270 */
    }
    else
    { 
      theta = 0;
      phi = 0;
    }
    
    int t_index = theta/theta_range;
    int p_index = phi/phi_range;
    int bin_index = (t_index * DIM_PHI) + p_index;
    if(bin_index > DIM_THETA*DIM_PHI)
    { 
      cout << "Indexing error" << endl;
    }
    else
    { 
      bins[bin_index]++;
    }
  }
}

void sampleNeighborhood(float *sample_x, float *sample_y, float *sample_z, int N, int *dims, int i, int j, float *vec_x, float *vec_y, float *vec_z)
{ 
  int offset = (N-1)/2;
  int num_samples = N*N;
  int cnt = 0;
  
  if(dims[0] > N && dims[1] > N) // && dims[2] > N
  {   
    for(int q = j - offset; q < j + offset; q++)
    { 
      for(int p = i - offset; p < i + offset; p++) // x grows fastest
      {     
    /* Left out
 *  // Right out
 *  // Top out
 *  // Bottom out
 *  // Front out
 *  // Back out */
            int x_i, y_i, z_i;
            if(p < 0)
              x_i = -1*p;
            else if(p > (dims[0] -1))        
              x_i = (dims[0] - 1) - (p - (dims[0] - 1));
            else
              x_i = p;
            if(q < 0)
              y_i = -1*q;
            else if(q > (dims[1] -1))        
              y_i = (dims[1] - 1) - (q - (dims[1] - 1));
            else
              y_i = q;
              
						int index = y_i*dims[0] + x_i;
            sample_x[cnt] = vec_x[index];
            sample_y[cnt] = vec_y[index];
            sample_z[cnt] = vec_z[index];
            cnt++;
        }
      }
  }
  else
  { 
    /* Neighborhood too large */
  }
}

namespace worklets
{

  class VectorEntropy : public vtkm::worklet::WorkletMapField
  {
    public: 
      typedef void ControlSignature(FieldOut, WholeArrayIn);
      typedef void ExecutionSignature(_1, _2, WorkIndex);
      
      VTKM_CONT VectorEntropy(int t, int p, int x, int y, int n)
      {
        dim_theta = t;
        dim_phi = p;
        dims[0] = x;
        dims[1] = y;
				N = n;
//        dims[2] = z;
      }
      
      template<typename EntropyType, typename VelocityPortal>
      VTKM_EXEC void operator()(EntropyType &vectorEntropy, const VelocityPortal &velocityArray, const vtkm::Id &index) const
      {
        int num_bins = dim_theta * dim_phi;
        int bins[num_bins] = {0};
        int idx[2];
        idx[0] = index % (dims[0]);
        idx[1] = index / (dims[0]);
        int offset = (N-1)/2;            
//        int num_samples = ((offset*2)+1) * ((offset*2)+1) * ((offset*2)+1); 
        int num_samples = ((offset*2)+1) * ((offset*2)+1); 
   
        int count = 0;
            
        vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64, 3>> sample_vec;
        sample_vec.Allocate(num_samples);
        // Neighborhood sampling code //
        { 
        	if(dims[0] > N && dims[1] > N) 
        	{
          	for(int q = idx[1] - offset; q <= idx[1] + offset; q++)
          	{ 	  
            	for(int p = idx[0] - offset; p <= idx[0] + offset; p++) // x grows fastest
            	{   
	              int x_i, y_i;
  	            if(p < 0)
    	            x_i = -1*p;
      	        else if(p > (dims[0] -1))    
        	       x_i = (dims[0] - 1) - (p - (dims[0] - 1));
          	    else
            	    x_i = p;
              
	              if(q < 0)
  	              y_i = -1*q;
    	          else if(q > (dims[1] -1))    
      	          y_i = (dims[1] - 1) - (q - (dims[1] - 1));
        	      else
          	      y_i = q;
               
            		int pt_id = y_i*dims[0] + x_i;
        
       	    		if(pt_id < dims[0]*dims[1] && pt_id >= 0)
        	  		{
             			auto v = velocityArray.Get(pt_id);
              		sample_vec.WritePortal().Set(count, vtkm::Vec<vtkm::Float64,3>(v[0],v[1],v[2]));
              		count++;  
            		}
          		}
        		}		   
      		}	
				} // End of finding neighborhood samples //
      		{ // Perform binning of each sample //
		        double theta_range = 360.0/dim_theta;  
   		      double phi_range = 180.0/dim_phi;  
              
        		for(int n = 0; n < count; n++)
        		{
          		auto sample = sample_vec.ReadPortal().Get(n);
          		double radius, theta, phi;
          		radius = sqrt((sample[0]*sample[0]) + (sample[1]*sample[1]) + (sample[2]*sample[2]));
		          if(radius > 0)
    		      {
		            theta = (atan2(sample[1],sample[0]))*(180/PI);
    		        phi = (acos(sample[2]/radius))*(180/PI);
        		    if(theta < 0)
            		  theta = 360 + theta; // theta value is negative, i.e., -90 is the same as 270 
          		}
          		else
          		{
            		theta = 0;
            		phi = 0;
          		}
        
		          if(theta == 360)
    		        theta = 0;
          		if(phi == 180)
            		phi = 0;
                 
		          int t_index = theta/theta_range;
   		    	  int p_index = phi/phi_range;
      		    int bin_index = (t_index * dim_phi) + p_index;
		          if(bin_index < dim_theta*dim_phi && bin_index >= 0)
    		      {
        		    bins[bin_index]++;
          		}
        		}
      		} // End of binning process //
      		{ // Compute the entropy // 
        		double probability[num_bins];
        		int total_sum = 0;
        		for(int n = 0; n < num_bins; n++)
		        {
    		      total_sum += bins[n];
        		}
            
		        for(int n = 0; n < num_bins; n++)
    		    {
        		  probability[n] = bins[n]/(total_sum*1.0);
        		}
        
		        vtkm::FloatDefault entropy = 0.0;
            
		        for(int n = 0; n < num_bins; n++)
    		    {
        		  if(probability[n] > 0.0)
            	entropy -= (probability[n]*log2(probability[n]));
        		}
		        vectorEntropy = entropy;      
   			  } // End compute entropy //
    	  }

    private:
      int dim_theta, dim_phi;
      int dims[2];
			int N;
  };
  




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
	int neighborhood = atoi(argv[14]);
	int num_seeds_slice = sdims[0]*sdims[1];
	int num_seeds_total = sdims[0]*sdims[1]*iterations;
  
	vtkm::cont::ArrayHandle<vtkm::Particle> seed_set;
	seed_set.Allocate(num_seeds_total);
	
	float x, y, s_time;
	int seed_counter = 0;


/*
  float* vec_x, *vec_y, *vec_z;
  vec_x = (float*)malloc(sizeof(float)*num_pts_slice);
  vec_y = (float*)malloc(sizeof(float)*num_pts_slice);
  vec_z = (float*)malloc(sizeof(float)*num_pts_slice);
*/


	/* Seed placement steps.
 * Extract field at grid points
 * Sum the values and create a cell field.
 * Scan the values.
 * Generate a random number. 
 */
  
/*	vtkm::Id3 entDims(dims[0], dims[1], 1);
//  Vec3f origin3d(static_cast<vtkm::FloatDefault>(xc[0]),
//                 static_cast<vtkm::FloatDefault>(yc[0]),
//                 static_cast<vtkm::FloatDefault>(zc[0]));
//  Vec3f spacing3d(static_cast<vtkm::FloatDefault>(x_spacing),
//                  static_cast<vtkm::FloatDefault>(y_spacing),
//                  static_cast<vtkm::FloatDefault>(z_spacing));
  vtkm::cont::DataSet entOutput;
  entOutput = uniformDatasetBuilder3d.Create(entDims, origin3d, spacing3d);
*/

	for(int k = 0; k < iterations; k++)
	{
		// Get the 2D slice at a specific value of Z. 0, 100, 200, 300, .. num_steps_iter*i
		vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> velocityArray;
		velocityArray.Allocate(num_pts_slice);	
		
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
				velocityArray.WritePortal().Set(index2, vtkm::Vec<vtkm::FloatDefault,3>(vec[0], vec[1], vec[2]));

//        vec_x[index2] = vec[0];
//        vec_y[index2] = vec[1];
//        vec_z[index2] = vec[2];
    	}
  	}
    
		vtkm::cont::ArrayHandle<vtkm::FloatDefault> featurePointField;
		featurePointField.Allocate(num_pts_slice);
    vtkm::cont::ArrayHandle<vtkm::FloatDefault> featureCellField;
		featureCellField.Allocate(num_cells_slice);
		vtkm::cont::ArrayHandle<vtkm::Particle> SeedArray;
		SeedArray.Allocate(num_seeds_slice);

    vtkm::worklet::DispatcherMapField<worklets::VectorEntropy>(worklets::VectorEntropy(DIM_THETA, DIM_PHI, dims[0], dims[1], neighborhood)).Invoke(featurePointField, velocityArray);
/*
		int num_samples = neighborhood*neighborhood;

    for(int j = 0; j < dims[1]; j++)
    { 
      for(int i = 0; i < dims[0]; i++)
      { 
        int num_bins = DIM_THETA * DIM_PHI;
        int bins[num_bins] = {0};
				int index = j*dims[0] + i; 
        float sample_x[num_samples],sample_y[num_samples],sample_z[num_samples];
          
        sampleNeighborhood(sample_x, sample_y, sample_z, neighborhood, dims, i, j, vec_x, vec_y, vec_z);
        estimateDistribution(bins, num_bins, sample_x, sample_y, sample_z, neighborhood);
        float H = calculateEntropy(bins, num_bins);
				
				featurePointField.WritePortal().Set(index, static_cast<vtkm::FloatDefault>(H));
      }
    }		
*/		
		vtkm::worklet::DispatcherMapField<worklets::PointFieldToCellAverageField>(worklets::PointFieldToCellAverageField(dims[0], dims[1])).Invoke(featureCellField, featurePointField);	
    vtkm::worklet::DispatcherMapField<worklets::ExponentWorklet>(worklets::ExponentWorklet(exponent)).Invoke(featureCellField);
//    vtkm::worklet::DispatcherMapField<worklets::AbsoluteValueWorklet>(worklets::AbsoluteValueWorklet()).Invoke(featureCellField);
    vtkm::worklet::DispatcherMapField<worklets::MinimumValueWorklet>(worklets::MinimumValueWorklet(minimum)).Invoke(featureCellField);	


/*		std::stringstream s;
		s << "Ent" << k;
		std::stringstream s2;
		s2 << "EntP" << k;
		entOutput.AddPointField(s2.str().c_str(), featurePointField);
		entOutput.AddCellField(s.str().c_str(), featureCellField);
*/

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

//    vtkm::io::VTKDataSetWriter wrt2("entropy_fields.vtk");
//    wrt2.WriteDataSet(entOutput);
		
}
