#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "object.cuh"

class Parse;

class Sphere : public Object
{
public:
    vec3 center;
    double radius;

    Sphere(const Parse* parse,std::istream& in);
    Sphere() 
        :center(0,0,0), radius(0.0)
    {}

    ~Sphere() = default;

    __host__ __device__
    Hit Intersection(const Ray& ray, int part) const ;
    
    __host__ __device__
    vec3 Normal(const Ray& ray, const Hit& hit) const;
    
    __host__ __device__
    std::pair<Box,bool> Bounding_Box(int part) const ;

    static constexpr const char* parse_name = "sphere";
};
#endif
