#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "hit.cuh"
#include "ray.cuh"
#include "object.cuh"
#include "plane.cuh"
#include "sphere.cuh"

/*======================TEMPORARY==========================*/

//Hit
__global__ void kernel_by_pointer_hit(Hit *hit, Hit *hit2);
__global__ void kernel_by_ref_hit(Hit &hit, Hit &hit2);
__global__ void kernel_by_value_hit(Hit hit, Hit hit2);

void launch_by_pointer_hit(Hit *hit, Hit *hit2);
void launch_by_ref_hit(Hit &hit, Hit &hit2);
void launch_by_value_hit(Hit hit, Hit hit2);

//Ray
__global__ void kernel_by_pointer_ray(Ray *ray);
__global__ void kernel_by_ref_ray(Ray &ray);
__global__ void kernel_by_value_ray(Ray ray);

void launch_by_pointer_ray(Ray *ray);
void launch_by_ref_ray(Ray &ray);
void launch_by_value_ray(Ray ray);

//Plane
//__global__ void kernel_by_pointer_plane(Plane *p);
//__global__ void kernel_by_ref_plane(Plane &p);
//__global__ void kernel_by_value_plane(Plane p);

//void launch_by_pointer_plane(Plane *p);
//void launch_by_ref_plane(Plane &p);
//void launch_by_value_plane(Plane p);

/*======================TEMPORARY==========================*/

#endif
