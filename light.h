#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "vec.cuh"
#include "misc.h"
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>
#include "managed.cuh"

class Ray;
class Parse;

class Light: public Managed
{
public:
    std::string name;
    vec3 position;
    
    Light() = default;
    virtual ~Light() = default;

    virtual vec3 Emitted_Light(const vec3& vector_to_light) const=0;
};
#endif
