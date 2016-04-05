#pragma once

#include "hoNDArray_math.h"
#include "hoCuNDArray_math.h"
#include "hoNDArray_fileio.h"
#include "complext.h"
#include "nlcgSolver.h"
#include "hoSolverUtils.h"
#include <iostream>
#include <string>

using namespace std;
using namespace Gadgetron;

//namespace Gadgetron
//{

template<class T> class hoCuNlcgSolver: public nlcgSolver<hoCuNDArray<T> >
{
    typedef typename realType<T>::Type REAL;
public:
    hoCuNlcgSolver():nlcgSolver<hoCuNDArray<T> >()
    {

    }

    virtual ~hoCuNlcgSolver(){};

    virtual void iteration_callback(hoCuNDArray<T>* x,int i,REAL data_res,REAL reg_res)
    {

        /*
      if (i == 0){
          std::ofstream textFile("residual.txt",std::ios::trunc);
          textFile << data_res << std::endl;
      } else{
          std::ofstream textFile("residual.txt",std::ios::app);
          textFile << data_res << std::endl;
      }
      std::stringstream ss;
      ss << "iteration-" << i << ".real";
      write_nd_array(x,ss.str().c_str());
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

protected:
    unsigned int dumpFreq_;
    string dumpName_;
};
//}
