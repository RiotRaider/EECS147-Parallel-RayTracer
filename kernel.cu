#include "vec.cuh"
#include "managed.cuh"
#include "kernel.cuh"

/*======================TEMPORARY==========================*/
__global__ 
void Kernel_by_pointer(Hit *elem, Hit *elem2) {
  printf("On device by pointer (before changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", elem->uv[0], elem->uv[1], elem->dist, elem->triangle);
  
  elem->dist+=10;
  elem->triangle+=50;

  vec2 uv2 = {25, 25};
  elem->uv += elem2->uv;
  elem->uv += uv2;

  printf("On device by pointer (after changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", elem->uv[0], elem->uv[1], elem->dist, elem->triangle);
}

__global__ 
void Kernel_by_ref(Hit &elem, Hit &elem2) {
  printf("On device by ref (before changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", elem.uv[0], elem.uv[1], elem.dist, elem.triangle);

  elem.dist+=20;
  elem.triangle+=100;

  vec2 uv2 = {75, 75};
  elem.uv += elem2.uv;
  elem.uv += uv2;

  printf("On device by ref (after changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", elem.uv[0], elem.uv[1], elem.dist, elem.triangle);
}

__global__ 
void Kernel_by_value(Hit elem, Hit elem2) {
  printf("On device by value (before changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", elem.uv[0], elem.uv[1], elem.dist, elem.triangle);

  elem.dist+=30;
  elem.triangle+=150;

  vec2 uv2 = {125, 125};
  elem.uv += elem2.uv;
  elem.uv += uv2;

  printf("On device by value (after changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", elem.uv[0], elem.uv[1], elem.dist, elem.triangle);
}

void launch_by_pointer(Hit *elem, Hit *elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by pointer: name=(%d, %d, %d), value=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value);
  Kernel_by_pointer<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}

void launch_by_ref(Hit &elem, Hit &elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by ref: name=(%d, %d, %d), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
  Kernel_by_ref<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}

void launch_by_value(Hit elem, Hit elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by value: name=(%d, %d, %d), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
  Kernel_by_value<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}

/*======================TEMPORARY==========================*/
