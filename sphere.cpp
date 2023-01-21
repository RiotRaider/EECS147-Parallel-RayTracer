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
    vec3 w= ray.endpoint-this->center;
    double wu=0,uu=0,ww=00;
    double t1,t2;
    for(int i=0;i<3;i++){
        wu+=(w[i]*ray.direction[i]);
        uu+=(ray.direction[i]*ray.direction[i]);
        ww+=(w[i]*w[i]);
    }
    t1 = -wu-sqrt((wu*wu)-(ww-(this->radius*this->radius)));
    t2 = -wu+sqrt((wu*wu)-(ww-(this->radius*this->radius)));
    if(t1<0){hit.dist=t2;}
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
