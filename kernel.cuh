#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "hit.cuh"
#include "ray.cuh"

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

/*======================TEMPORARY==========================*/

#endif
