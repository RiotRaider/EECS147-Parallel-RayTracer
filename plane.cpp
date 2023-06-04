//Student Name: Justin Sanders
//Student ID: 862192429
#include "plane.h"
#include "hit.cuh"
#include "ray.cuh"
#include <cfloat>
#include <limits>

Plane::Plane(const Parse* parse,std::istream& in)
{
    in>>name>>x>>normal;
    normal=normal.normalized();
}

// Intersect with the plane.  The plane's normal points outside.
Hit Plane::Intersection(const Ray& ray, int part) const
{
    Hit hit;
    double un=0;//dotproduct of ray direction and plane normal
    double rn=0;//dotproduct of ray formed by plane point and endpoint and the plane normal
    un = dot(ray.direction,this->normal);
    rn = dot((this->x-ray.endpoint),this->normal);

    if(un!=0){
        hit.dist = rn/un;
    }
    return hit;
}

vec3 Plane::Normal(const Ray& ray, const Hit& hit) const
{
    return normal;
}

std::pair<Box,bool> Plane::Bounding_Box(int part) const
{
    Box b;
    b.Make_Full();
    return {b,true};
}
