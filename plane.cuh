#ifndef __PLANE_H__
#define __PLANE_H__

#include "object.cuh"

class Parse;

class Plane : public Object, public Managed
{
public:
    vec3 x;
    vec3 normal;

    Plane(const Parse* parse,std::istream& in);
    virtual ~Plane() = default;

    __host__ __device__
    virtual Hit Intersection(const Ray& ray, int part) const override;
    
    __host__ __device__
    virtual vec3 Normal(const Ray& ray, const Hit& hit) const override;
    
    __host__ __device__
    virtual std::pair<Box,bool> Bounding_Box(int part) const override;

    static constexpr const char* parse_name = "plane";
};
#endif
