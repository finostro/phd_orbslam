/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2013, Keith Leung, Felipe Inostroza
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Advanced Mining Technology Center (AMTC), the
 *       Universidad de Chile, nor the names of its contributors may be
 *       used to endorse or promote products derived from this software without
 *       specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AMTC, UNIVERSIDAD DE CHILE, OR THE COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 * THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef MEASUREMENTWITHDESCRIPTOR_HPP
#define MEASUREMENTWITHDESCRIPTOR_HPP

#include "measurement_models/MeasurementModel.hpp"
#include "Landmark.hpp"
#include "Pose.hpp"

namespace rfs{

  template <class MeasurementType, class DescriptorType>
  class MeasurementWithDescriptor: public MeasurementType {
  public:
    typedef DescriptorType TDescriptor;
    typedef MeasurementType TMeasurement;

    TDescriptor desc;


    double evalGaussianLikelihood(const MeasurementWithDescriptor &x_eval,
                                      double* mDist2 = NULL){
      return TMeasurement::evalGaussianLikelihood(x_eval , mDist2) * desc.likelihood(x_eval.desc);
    }
    double  evalGaussianLikelihood (const MeasurementWithDescriptor &x_eval, typename TMeasurement::Vec &n_error, double* mDist2 = NULL){
      return TMeasurement::evalGaussianLikelihood(x_eval , n_error , mDist2)*desc.likelihood(x_eval.desc);
    }
     void sample(){
       TMeasurement::sample();
       desc=desc.sample();
     }
     void sample( MeasurementWithDescriptor &s_sample ){
       s_sample = *this;
       s_sample.sample();
     }
  };



}

#endif
