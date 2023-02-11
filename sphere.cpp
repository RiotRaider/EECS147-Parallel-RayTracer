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
    Hit hit;
    vec3 w= (this->center-ray.endpoint);
    double wu=0;
    double t1,t2;
    wu = dot(w,ray.direction);
    t1 = wu - sqrt(pow(this->radius,2)-(dot(w,w)-pow(wu,2)));
    t2 = wu + sqrt(pow(this->radius,2)-(dot(w,w)-pow(wu,2)));
    if(t1<small_t){hit.dist=t2;}
    else {hit.dist=t1;}

    return hit;
}

vec3 Sphere::Normal(const Ray& ray, const Hit& hit) const
{
    vec3 normal;
    // compute the normal direction
    normal = (ray.Point(hit.dist)-this->center)/this->radius;
    normal=normal.normalized();

    return normal;
}

std::pair<Box,bool> Sphere::Bounding_Box(int part) const
{
    return {{center-radius,center+radius},false};
}
