#pragma once

#include "cgSolver.h"

#include "cgSolver.h"
#include "hoNDArray_math.h"
#include "hoCuNDArray_math.h"
#include <string>

using namespace std;
using namespace Gadgetron;

/** \class hoCuCgSolver
      \brief Instantiation of the conjugate gradient solver on the cpu.

      The class hoCuCgSolver is a convienience wrapper for the device independent cgSolver class.
      hoCuCgSolver instantiates the cgSolver for type hoNDArray<T>.
  */
template <class T> class hoCuCgSolver : public cgSolver< hoCuNDArray<T> >
{
public:
    hoCuCgSolver() : cgSolver<hoCuNDArray<T> >(), _it(0)
    {

    }
    virtual ~hoCuCgSolver() {};

    // TSS: This is too expensive to do in general. Move responsibility of dumping to the apps.
    virtual void solver_dump(hoCuNDArray<T>* x)
    {
        /*std::stringstream ss;
            ss << "iteration-" << _it << ".real";
            write_nd_array(x,ss.str().c_str());
            _it++;
          */
        if (dumpFreq_ < 10000)
        {

            if( (i % dumpFreq_) == 0 )
            {
                printf("Dumping frame\n");
                char filename[1000];
                sprintf(filename, "%s_%04i.real",dumpName_.c_str(),i);
                std::cout << "Dump Name " << filename << std::endl;
                write_nd_array<float>(x, filename);
            }
        }
    };




    void set_dump_frequency(unsigned int dumpFreq)
    {
        if( dumpFreq == 0 )
            this->dumpFreq_ = 9999999; // Not sure how modulus 0 behaves, so just make it a large number that is never reached...
        else
            this->dumpFreq_ = dumpFreq;
    };

    void set_dump_name(string dumpName)
    {
        this->dumpName_ = dumpName;
    };

private:
    int _it;

protected:
    unsigned int dumpFreq_;
    string dumpName_;
};

