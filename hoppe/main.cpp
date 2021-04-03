//
//  main.cpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#include <iostream>
#include <fstream>
#include "Hoppe.hpp"


int main(int argc, const char * argv[]) {
    Hoppe hoppe;
    hoppe.load_pointcloud("assets/res.xyz");
    hoppe.run();
    hoppe.export_mesh("result.obj");
    return 0;
}
