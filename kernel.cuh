#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "hit.cuh"
#include "ray.cuh"
#include "object.cuh"
#include "plane.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "render_world.cuh"

/*======================TEMPORARY==========================*/

__global__ void Kernel_by_pointer(Camera *elem);

__global__ void Kernel_by_ref(Camera &elem);

__global__ void Kernel_by_value(Camera elem);

void launch_by_pointer(Camera *elem);
void launch_by_ref(Camera &elem);
void launch_by_value(Camera elem);

/*======================TEMPORARY==========================*/
__global__ void Kernel_Render_Pixel(Render_World& r);
#endif
