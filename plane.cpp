//Student Name: Justin Sanders
//Student ID: 862192429
#include "plane.h"
#include "hit.h"
#include "ray.h"
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
    for(int i = 0;i<3;i++){
         un += ray.direction[i]*this->normal[i];
         rn += (this->x[i]-ray.endpoint[i])*this->normal[i];
    }
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
