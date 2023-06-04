#include "vec.cuh"
#include "managed.cuh"
#include "kernel.cuh"

/*======================TEMPORARY==========================*/

//Hit
__global__ 
void kernel_by_pointer_hit(Hit *hit, Hit *hit2) {
  printf("On device by pointer (before changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", hit->uv[0], hit->uv[1], hit->dist, hit->triangle);
  
  hit->dist+=10;
  hit->triangle+=50;

  vec2 uv2 = {25, 25};
  hit->uv += hit2->uv;
  hit->uv += uv2;

  printf("On device by pointer (after changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", hit->uv[0], hit->uv[1], hit->dist, hit->triangle);
}

__global__ 
void kernel_by_ref_hit(Hit &hit, Hit &hit2) {
  printf("On device by ref (before changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", hit.uv[0], hit.uv[1], hit.dist, hit.triangle);

  hit.dist+=20;
  hit.triangle+=100;

  vec2 uv2 = {75, 75};
  hit.uv += hit2.uv;
  hit.uv += uv2;

  printf("On device by ref (after changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", hit.uv[0], hit.uv[1], hit.dist, hit.triangle);
}

__global__ 
void kernel_by_value_hit(Hit hit, Hit hit2) {
  printf("On device by value (before changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", hit.uv[0], hit.uv[1], hit.dist, hit.triangle);

  hit.dist+=30;
  hit.triangle+=150;

  vec2 uv2 = {125, 125};
  hit.uv += hit2.uv;
  hit.uv += uv2;

  printf("On device by value (after changes): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", hit.uv[0], hit.uv[1], hit.dist, hit.triangle);
}

void launch_by_pointer_hit(Hit *hit, Hit *hit2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by pointer: name=(%d, %d, %d), value=%d\n", hit->color[0], hit->color[1], hit->color[2], hit->value);
  kernel_by_pointer_hit<<< dim_grid, dim_block >>>(hit, hit2);
  cudaDeviceSynchronize();
}

void launch_by_ref_hit(Hit &hit, Hit &hit2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by ref: name=(%d, %d, %d), value=%d\n", hit.color[0], hit.color[1], hit.color[2], hit.value);
  kernel_by_ref_hit<<< dim_grid, dim_block >>>(hit, hit2);
  cudaDeviceSynchronize();
}

void launch_by_value_hit(Hit hit, Hit hit2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by value: name=(%d, %d, %d), value=%d\n", hit.color[0], hit.color[1], hit.color[2], hit.value);
  kernel_by_value_hit<<< dim_grid, dim_block >>>(hit, hit2);
  cudaDeviceSynchronize();
}

//Ray
__global__ 
void kernel_by_pointer_ray(Ray *ray) {
  printf("On device by pointer (before changes) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f)\n", 
            ray->endpoint[0], ray->endpoint[1], ray->endpoint[2], ray->direction[0], ray->direction[1], ray->direction[2]);
  
  ray->endpoint = {20, 30, 40};
  ray->direction = {1, 2, 3};

  vec3 ray_point = ray->Point(5);

  printf("On device by pointer (after changes) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f), point=(%.2f, %.2f, %.2f)\n", 
            ray->endpoint[0], ray->endpoint[1], ray->endpoint[2], ray->direction[0], ray->direction[1], ray->direction[2], ray_point[0], ray_point[1], ray_point[2]);
}

__global__ 
void kernel_by_ref_ray(Ray &ray) {
  printf("On device by ref (before changes) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f)\n", 
            ray.endpoint[0], ray.endpoint[1], ray.endpoint[2], ray.direction[0], ray.direction[1], ray.direction[2]);
  
  ray.endpoint = {30, 40, 50};
  ray.direction = {10, 20, 30};

  vec3 ray_point = ray.Point(5);

  printf("On device by ref (after changes) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f), point=(%.2f, %.2f, %.2f)\n", 
            ray.endpoint[0], ray.endpoint[1], ray.endpoint[2], ray.direction[0], ray.direction[1], ray.direction[2], ray_point[0], ray_point[1], ray_point[2]);
}

__global__ 
void kernel_by_value_ray(Ray ray) {
  printf("On device by val (before changes) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f)\n", 
            ray.endpoint[0], ray.endpoint[1], ray.endpoint[2], ray.direction[0], ray.direction[1], ray.direction[2]);
  
  ray.endpoint = {40, 50, 60};
  ray.direction = {5, 10, 15};

  vec3 ray_point = ray.Point(5);

  printf("On device by val (after changes) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f), point=(%.2f, %.2f, %.2f)\n", 
            ray.endpoint[0], ray.endpoint[1], ray.endpoint[2], ray.direction[0], ray.direction[1], ray.direction[2], ray_point[0], ray_point[1], ray_point[2]);
}

void launch_by_pointer_ray(Ray *ray) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by pointer: name=(%d, %d, %d), value=%d\n", hit->color[0], hit->color[1], hit->color[2], hit->value);
  kernel_by_pointer_ray<<< dim_grid, dim_block >>>(ray);
  cudaDeviceSynchronize();
}

void launch_by_ref_ray(Ray &ray) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by ref: name=(%d, %d, %d), value=%d\n", hit.color[0], hit.color[1], hit.color[2], hit.value);
  kernel_by_ref_ray<<< dim_grid, dim_block >>>(ray);
  cudaDeviceSynchronize();
}

void launch_by_value_ray(Ray ray) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by value: name=(%d, %d, %d), value=%d\n", hit.color[0], hit.color[1], hit.color[2], hit.value);
  kernel_by_value_ray<<< dim_grid, dim_block >>>(ray);
  cudaDeviceSynchronize();
}

/*======================TEMPORARY==========================*/
