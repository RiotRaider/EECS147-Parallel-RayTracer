//Student Name: Justin Sanders
//Student ID: 862192429
#include "sphere.h"
#include "ray.h"

Sphere::Sphere(const Parse* parse, std::istream& in)
{
    in>>name>>center>>radius;
}

// Determine if the ray intersects with the sphere
Hit Sphere::Intersection(const Ray& ray, int part) const
{
    //TODO-HW2
    TODO;
    //END TODO-HW2
    return {};
}

vec3 Sphere::Normal(const Ray& ray, const Hit& hit) const
{
    vec3 normal;
    //TODO-HW2
    TODO; // compute the normal direction
    //END TODO-HW2 
    return normal;
}

std::pair<Box,bool> Sphere::Bounding_Box(int part) const
{
    return {{center-radius,center+radius},false};
}
