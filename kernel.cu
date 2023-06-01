#include "vec.cu"
#include "managed.h"
#include "kernel.h"

/*======================TEMPORARY==========================*/
__global__ 
void Kernel_by_pointer(DataElement *elem, DataElement *elem2) {
  printf("On device by pointer (before changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value);
  
  //elem->color[0] = 255;
  elem->value+=10;

  vec3 color2 = {100, 100, 100};
  elem->color += elem2->color;
  elem->color += color2;

  printf("On device by pointer (after changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value);
}

__global__ 
void Kernel_by_ref(DataElement &elem, DataElement &elem2) {
  printf("On device by ref (before changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);

  //elem.color[1] = 255;
  elem.value+=20;
  elem.color += elem2.color;

  printf("On device by ref (after changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
}

__global__ 
void Kernel_by_value(DataElement elem, DataElement elem2) {
  printf("On device by value (before changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);

  //elem.color[2] = 255;
  elem.value+=30;
  elem.color += elem2.color;

  printf("On device by value (after changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
}

void launch_by_pointer(DataElement *elem, DataElement *elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by pointer: name=(%d, %d, %d), value=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value);
  Kernel_by_pointer<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}

void launch_by_ref(DataElement &elem, DataElement &elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by ref: name=(%d, %d, %d), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
  Kernel_by_ref<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}

void launch_by_value(DataElement elem, DataElement elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by value: name=(%d, %d, %d), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
  Kernel_by_value<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}

/*======================TEMPORARY==========================*/