#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "object.cuh"

class Parse;

class Sphere : public Object, public Managed
{
    vec3 center;
    double radius;

public:
    Sphere(const Parse* parse,std::istream& in);

    ~Sphere();

    __host__ __device__
    virtual Hit Intersection(const Ray& ray, int part) const override;
    
    __host__ __device__
    virtual vec3 Normal(const Ray& ray, const Hit& hit) const override;
    
    __host__ __device__
    virtual std::pair<Box,bool> Bounding_Box(int part) const override;

    static constexpr const char* parse_name = "sphere";
};
#endif
