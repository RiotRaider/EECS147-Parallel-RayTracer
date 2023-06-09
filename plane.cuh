#ifndef __PLANE_H__
#define __PLANE_H__

#include "object.cuh"

class Parse;

class Plane : public Object
{
public:
    vec3 x;
    vec3 normal;

    Plane(const Parse* parse,std::istream& in);
    Plane() :x(0,0,0), normal(0,0,1) {}

    ~Plane();

    __host__ __device__
    Hit Intersection(const Ray& ray, int part) const;
    
    __host__ __device__
    vec3 Normal(const Ray& ray, const Hit& hit) const;
    
    __host__ __device__
    std::pair<Box,bool> Bounding_Box(int part) const;

    static constexpr const char* parse_name = "plane";
};
#endif
