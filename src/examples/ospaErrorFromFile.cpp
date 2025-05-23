/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2014, Keith Leung
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

#include <iostream>
#include <math.h>
#include <vector>
#include "OSPA.hpp"
#include <fstream>
class Pos2d{

public:
  
  Pos2d(double x, double y): x_(x), y_(y){}
  ~Pos2d(){}

  static double distance( const Pos2d x1 , const Pos2d x2 ){
    double dx = x1.x_ - x2.x_;
    double dy = x1.y_ - x2.y_;
    return sqrt(dx*dx + dy*dy);
  }
  
private:

  double x_;
  double y_;

};


int main(int argc, char *argv[]){

  const double cutoff = 1;
  const double order = 1;


  std::ifstream in1("file1");
  std::ifstream in2("file2");


  std::vector<Pos2d> set1;
  std::vector<Pos2d> set2;


  while(true){
    double x , y;
    in1 >> x;
    in1 >> y;
    if(!in1.good()) break;
    std:: cout << "x,y   "  << x << "  ,  " << y << "\n";

    set1.push_back(Pos2d(x,y));
  }
  std::cout << "\n";
  while(true){
      double x , y;
      in2 >> x;
      in2 >> y;
      if(!in2.good()) break;
      std:: cout << "x,y   "  << x << "  ,  " << y << "\n";
      set2.push_back(Pos2d(x,y));
    }

  /* Set 1
  set1.push_back(Pos2d(175, 190));
  set1.push_back(Pos2d(205, 363)); 
  set1.push_back(Pos2d(158, 623)); 
  set1.push_back(Pos2d(282, 501));
  set1.push_back(Pos2d(418, 431));
  set1.push_back(Pos2d(496, 646)); 
  set1.push_back(Pos2d(561, 787)); 
  set1.push_back(Pos2d(757, 789)); 
  set1.push_back(Pos2d(877, 759));

  set2.push_back(Pos2d(175, 192));
  set2.push_back(Pos2d(205, 361));
  set2.push_back(Pos2d(160, 623));
  set2.push_back(Pos2d(282, 500));
  set2.push_back(Pos2d(418, 433));
  set2.push_back(Pos2d(496, 648));
  set2.push_back(Pos2d(561, 789));
  set2.push_back(Pos2d(757, 787));
  set2.push_back(Pos2d(877, 757));
  set2.push_back(Pos2d(783, 132));
   */

  /* Set 2 
  set1.push_back(Pos2d(183, 222));
  set1.push_back(Pos2d(374, 310));
  set1.push_back(Pos2d(671, 178));
  set1.push_back(Pos2d(248, 578));
  set1.push_back(Pos2d(725, 541));
  set1.push_back(Pos2d(666, 739));

  set2.push_back(Pos2d(184, 222));
  set2.push_back(Pos2d(375, 310));
  set2.push_back(Pos2d(672, 178));
  set2.push_back(Pos2d(249, 578));
  set2.push_back(Pos2d(726, 541));
  set2.push_back(Pos2d(667, 739));
  set2.push_back(Pos2d(182, 221));
  set2.push_back(Pos2d(373, 309));
  set2.push_back(Pos2d(670, 177));
  set2.push_back(Pos2d(247, 577));
  set2.push_back(Pos2d(724, 540));
  set2.push_back(Pos2d(665, 738));
  set2.push_back(Pos2d(183, 223));
  set2.push_back(Pos2d(374, 311));
  set2.push_back(Pos2d(671, 179));
  set2.push_back(Pos2d(248, 579));
  set2.push_back(Pos2d(725, 542));
  set2.push_back(Pos2d(666, 740));
  set2.push_back(Pos2d(183, 221));
  set2.push_back(Pos2d(374, 309));
  set2.push_back(Pos2d(671, 177));
  set2.push_back(Pos2d(248, 577));
  set2.push_back(Pos2d(725, 540));
  set2.push_back(Pos2d(666, 738));
  set2.push_back(Pos2d(182, 222));
  set2.push_back(Pos2d(373, 310));
  set2.push_back(Pos2d(670, 178));
  set2.push_back(Pos2d(247, 578));
  set2.push_back(Pos2d(724, 541));
  set2.push_back(Pos2d(665, 739));
  */


  /* Set 3 
  set1.push_back(Pos2d(300, 500));

  set2.push_back(Pos2d(300, 499));
  set2.push_back(Pos2d(300, 501));
  */

  /* Set 4 
  set1.push_back(Pos2d(300, 500));
  set1.push_back(Pos2d(700, 500));

  set2.push_back(Pos2d(300, 499));
  set2.push_back(Pos2d(300, 501));
  set2.push_back(Pos2d(700, 499));
  */

  /* Set 5 
  set1.push_back(Pos2d(300, 500));
  set1.push_back(Pos2d(700, 500));

  set2.push_back(Pos2d(300, 499));
  set2.push_back(Pos2d(300, 501));
  set2.push_back(Pos2d(700, 499));
  set2.push_back(Pos2d(700, 501));
  */

  rfs::OSPA<Pos2d> ospa(set1, set2, cutoff, order, &Pos2d::distance);
  double e, e_d, e_c;
  e = ospa.calcError(&e_d, &e_c, true);
  ospa.reportSoln();
  std::cout << "OSPA error:        " << e << std::endl;
  std::cout << "distance error:    " << e_d << std::endl;
  std::cout << "cardinality error: " << e_c << std::endl;
  for(int i = 0; i < set1.size(); i++){
    double cost;
    int j = ospa.getOptAssignment(i, &cost);
    std::cout << i << " --- " << j << "  Cost: " << cost << std::endl;
  }
};
