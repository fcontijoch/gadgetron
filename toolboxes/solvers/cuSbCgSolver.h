#pragma once

#include "sbSolver.h"
#include "cuCgSolver.h"
#include "cuNDArray.h"
#include "real_utilities.h"
#include "vector_td_utilities.h"
#include "ndarray_vector_td_utilities.h"

template <class REAL, class T> class cuSbCgSolver 
  : public sbSolver< REAL, T, cuNDArray<REAL>, cuNDArray<T>, cuCgSolver<REAL,T> >
{
public:
  
  cuSbCgSolver() : sbSolver< REAL, T, cuNDArray<REAL>, cuNDArray<T>, cuCgSolver<REAL,T> >() { set_device(-1); }
  virtual ~cuSbCgSolver() {}
  
#include "cuSbSolver_macros.h"
  
protected:
  int device_;
  int old_device_;
};