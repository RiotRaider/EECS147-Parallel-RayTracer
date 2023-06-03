#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "camera.cuh"

/*======================TEMPORARY==========================*/
/*struct DataElement : public Managed
{
  vec3 color;
  int value;
};
*/


__global__ void Kernel_by_pointer(Camera *elem);

__global__ void Kernel_by_ref(Camera &elem);

__global__ void Kernel_by_value(Camera elem);

void launch_by_pointer(Camera *elem);
void launch_by_ref(Camera &elem);
void launch_by_value(Camera elem);


/*======================TEMPORARY==========================*/

#endif
