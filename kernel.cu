#include "vec.cuh"
#include "managed.cuh"
#include "kernel.cuh"

/*======================TEMPORARY==========================*/

/* __global__ 
void Kernel_by_pointer(Camera *elem) {
  int x=elem->number_pixels[0];
  int y=elem->number_pixels[1];
  __shared__ vec3 color;
  ivec2 ind;
  ind[0]=280;
  ind[1]=320;
  color[threadIdx.x] = 1;
  if(threadIdx.x==0){
    elem->Set_Pixel(ind,Pixel_Color(color));
    printf("On device by pointer \n");
  }
  __syncthreads();
  
}

__global__ 
void Kernel_by_ref(Camera &elem) {
  
  int x=elem.number_pixels[0];
  int y=elem.number_pixels[1];
  __shared__ vec3 color;
  ivec2 ind;
  ind[0]=280;
  ind[1]=320;
  color[threadIdx.x] = 1;
  if(threadIdx.x==0){
    elem.Set_Pixel(ind,Pixel_Color(color));
    printf("On device by reference \n");
  }
  __syncthreads();
}

__global__ 
void Kernel_by_value(Camera elem) {
  
  int x=elem.number_pixels[0];
  int y=elem.number_pixels[1];
  __shared__ vec3 color;
  ivec2 ind;
  ind[0]=280;
  ind[1]=320;
  color[threadIdx.x] = 1;
  if(threadIdx.x==0){
    elem.Set_Pixel(ind,Pixel_Color(color));
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
 */
/*======================TEMPORARY==========================*/

__global__ 
void Kernel_Render_Pixel(Render_World* r){
    if(blockIdx.x == 0 &&blockIdx.y == 0 && threadIdx.x == 0 &&threadIdx.y == 0 ){
      printf("The image is %i x %i\n",r->camera->number_pixels[0],r->camera->number_pixels[1]);
      printf("Flat shaded spheres: %d, phong shaded spheres: %d, flat shaded planes: %d, phong shaded planes: %d\nFlat shaders: %d, phong saders: %d", r->num_flat_shaded_spheres, r-> num_phong_shaded_spheres, r->num_flat_shaded_planes, r->num_phong_shaded_planes, r->num_flat_shaders, r->num_phong_shaders); //TODO: finish polymorph//
    }
    __syncthreads();
    if((threadIdx.x+blockDim.x*blockIdx.x) < r->camera->number_pixels[0] && (threadIdx.y+blockDim.y*blockIdx.y) < r->camera->number_pixels[1])
    {
        r->Render_Pixel(ivec2((threadIdx.x+blockDim.x*blockIdx.x),(threadIdx.y+blockDim.y*blockIdx.y)));
    }
    __syncthreads();
}
