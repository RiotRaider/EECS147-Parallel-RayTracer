#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "hit.cuh"
#include "ray.cuh"
#include "object.cuh"
#include "plane.cuh"
#include "sphere.cuh"
#include "camera.cuh"

/*======================TEMPORARY==========================*/

//Hit
__global__ void kernel_by_pointer_hit(Hit *hit, Hit *hit2);
__global__ void kernel_by_ref_hit(Hit &hit, Hit &hit2);
__global__ void kernel_by_value_hit(Hit hit, Hit hit2);

__global__ void Kernel_by_pointer(Camera *elem);

__global__ void Kernel_by_ref(Camera &elem);

__global__ void Kernel_by_value(Camera elem);

void launch_by_pointer(Camera *elem);
void launch_by_ref(Camera &elem);
void launch_by_value(Camera elem);

void launch_by_pointer_object(Plane *obj);
void launch_by_ref_object(Plane &obj);
void launch_by_value_object(Plane obj);

/*======================TEMPORARY==========================*/

#endif
