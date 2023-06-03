#include "vec.cuh"
#include "managed.cuh"
#include "kernel.cuh"

/*======================TEMPORARY==========================*/
__global__ 
void Kernel_by_pointer(Camera *elem) {
  int x=elem->number_pixels[0];
  int y=elem->number_pixels[1];
  __shared__ vec3 color;
  color[threadIdx.x] = 1;
  if(threadIdx.x==0){
    elem->colors[320*x+280]=Pixel_Color(color);
    printf("On device by pointer \n");
  }
  __syncthreads();
  
}

__global__ 
void Kernel_by_ref(Camera &elem) {
  
  int x=elem.number_pixels[0];
  int y=elem.number_pixels[1];
  __shared__ vec3 color;
  color[threadIdx.x] = 1;
  if(threadIdx.x==0){
    elem.colors[320*x+280]=Pixel_Color(color);
    printf("On device by reference \n");
  }
  __syncthreads();
}

__global__ 
void Kernel_by_value(Camera elem) {
  
  int x=elem.number_pixels[0];
  int y=elem.number_pixels[1];
  __shared__ vec3 color;
  color[threadIdx.x] = 1;
  if(threadIdx.x==0){
    elem.colors[320*x+280]=Pixel_Color(color);
    printf("On device by value \n");
  }
  __syncthreads();
}

void launch_by_pointer(Camera *elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(3, 1, 1);
  elem->colors[320*elem->number_pixels[0]+280]=0;
  vec3 c1 = From_Pixel(elem->colors[320*elem->number_pixels[0]+280]);
  printf("Pixel of interest(before):(%i,%i) : (%f, %f, %f)\n",280,320,c1[0],c1[1],c1[2]);
  
  printf("On host launch by pointer\n");
  Kernel_by_pointer<<< dim_grid, dim_block >>>(elem);
  cudaDeviceSynchronize();
  
  c1 = From_Pixel(elem->colors[320*elem->number_pixels[0]+280]);
  printf("Pixel of interest(after):(%i,%i) : (%f, %f, %f)\n\n",280,320,c1[0],c1[1],c1[2]);
}

void launch_by_ref(Camera &elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(3, 1, 1);
  elem.colors[320*elem.number_pixels[0]+280]=0;
  
  vec3 c1 = From_Pixel(elem.colors[320*elem.number_pixels[0]+280]);
  printf("Pixel of interest(before):(%i,%i) : (%f, %f, %f)\n",280,320,c1[0],c1[1],c1[2]);

  printf("On host launch by reference\n");
  Kernel_by_ref<<< dim_grid, dim_block >>>(elem);
  cudaDeviceSynchronize();
  
  c1 = From_Pixel(elem.colors[320*elem.number_pixels[0]+280]);
  printf("Pixel of interest(after):(%i,%i) : (%f, %f, %f)\n\n",280,320,c1[0],c1[1],c1[2]);
}

void launch_by_value(Camera elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(3, 1, 1);
  elem.colors[320*elem.number_pixels[0]+280]=0;
  vec3 c1 = From_Pixel(elem.colors[320*elem.number_pixels[0]+280]);
  printf("Pixel of interest(before):(%i,%i) : (%f, %f, %f)\n",280,320,c1[0],c1[1],c1[2]);
  
  printf("On host launch by value\n");
  Kernel_by_value<<< dim_grid, dim_block >>>(elem);
  cudaDeviceSynchronize();

  c1 = From_Pixel(elem.colors[320*elem.number_pixels[0]+280]);
  printf("Pixel of interest(after):(%i,%i) : (%f, %f, %f)\n\n",280,320,c1[0],c1[1],c1[2]);
}

/*======================TEMPORARY==========================*/
