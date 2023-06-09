#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "vec.cuh"
#include "misc.h"
#include <iostream>
#include <limits>
#include "color.cuh"
#include <math.h>
#include <vector>

class Ray;
class Parse;

class Light : public Managed
{
public:
    std::string name;
    vec3 position;
    float _pi = pi;
    const Color* color = nullptr; // RGB color components
    double brightness = 0;

    Light(const Light& l);
    Light(const Parse* parse,std::istream& in);
    ~Light() {cudaFree((void*)color);}
    __host__ __device__
    vec3 Emitted_Light(const vec3& vector_to_light) const;

    static constexpr const char* parse_name = "point_light";

private:
    void _realloc(){
        if(color != 0){
            cudaFree((void*)color);
        }
        cudaMallocManaged(&color,sizeof(Color));
    }
};
#endif
