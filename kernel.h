#ifndef __KERNEL_H__
#define __KERNEL_H__

/*======================TEMPORARY==========================*/
struct DataElement : public Managed
{
  vec3 color;
  int value;
};

__global__ void Kernel_by_pointer(DataElement *elem, DataElement *elem2);

__global__ 
void Kernel_by_ref(DataElement &elem, DataElement &elem2);

__global__ 
void Kernel_by_value(DataElement elem, DataElement elem2);

void launch_by_pointer(DataElement *elem, DataElement *elem2);
void launch_by_ref(DataElement &elem, DataElement &elem2);
void launch_by_value(DataElement elem, DataElement elem2);

/*======================TEMPORARY==========================*/

#endif
