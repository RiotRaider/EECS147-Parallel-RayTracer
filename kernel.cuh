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

//Object
__global__ void kernel_by_pointer_object(Plane *obj);
__global__ void kernel_by_ref_object(Plane &obj);
__global__ void kernel_by_value_object(Plane obj);

void launch_by_pointer_object(Plane *obj);
void launch_by_ref_object(Plane &obj);
void launch_by_value_object(Plane obj);

/*======================TEMPORARY==========================*/

#endif
