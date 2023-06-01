#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "hit.h"

/*======================TEMPORARY==========================*/
/*struct DataElement : public Managed
{
  vec3 color;
  int value;
};
*/


__global__ void Kernel_by_pointer(Hit *elem, Hit *elem2);

__global__ void Kernel_by_ref(Hit &elem, Hit &elem2);

__global__ void Kernel_by_value(Hit elem, Hit elem2);

void launch_by_pointer(Hit *elem, Hit *elem2);
void launch_by_ref(Hit &elem, Hit &elem2);
void launch_by_value(Hit elem, Hit elem2);


/*======================TEMPORARY==========================*/

#endif
