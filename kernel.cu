#include "vec.cuh"
#include "managed.cuh"
#include "kernel.cuh"

/*======================TEMPORARY==========================*/
__global__ 
void Kernel_by_pointer(Camera *elem) {
  int x=elem->number_pixels[0];
  int y=elem->number_pixels[1];
  //elem->colors[320*x+280]=(0xf<<24);

  printf("On device by pointer dimensions = %i x %i\n",x,y);
  //printf("Setting (%i,%i) to %s\n",280,320,"(255,0,0)");
}

__global__ 
void Kernel_by_ref(Camera &elem) {
  
  int x=elem.number_pixels[0];
  int y=elem.number_pixels[1];
  //elem.colors[100*x+250]=(0xf<<16);

  printf("On device by pointer dimensions = %i x %i\n",x,y);
  //printf("Setting (%i,%i) to %s\n",250,100,"(0,255,0)");
}

__global__ 
void Kernel_by_value(Camera elem) {
  
  int x=elem.number_pixels[0];
  int y=elem.number_pixels[1];

  //elem.colors[80*x+75]=(0xf<<8);

  printf("On device by pointer dimensions = %i x %i\n", x,y);
  //printf("Setting (%i,%i) to %s\n",75,80,"(0,0,255)");
  }

void launch_by_pointer(Camera *elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  printf("On host launch by pointer\n");
  Kernel_by_pointer<<< dim_grid, dim_block >>>(elem);

  cudaDeviceSynchronize();
}

void launch_by_ref(Camera &elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  printf("On host launch by reference\n");
  Kernel_by_ref<<< dim_grid, dim_block >>>(elem);
  cudaDeviceSynchronize();
}

void launch_by_value(Camera elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  printf("On host launch by value\n");
  Kernel_by_value<<< dim_grid, dim_block >>>(elem);
  cudaDeviceSynchronize();
}

/*======================TEMPORARY==========================*/
