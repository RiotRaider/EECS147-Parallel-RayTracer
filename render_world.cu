// Student Name: Justin Sanders
// Student ID: 862192429

#include <stdio.h>
#include <stdlib.h>

#include "render_world.cuh"
// #include "flat_shader.cuh"
// #include "object.cuh"
// #include "light.cuh"
// #include "ray.cuh"

#include "support.h"
#include "kernel.cuh"

#include "plane.cuh"
#include "sphere.cuh"

extern bool enable_acceleration;

#define BLOCK_SIZE 16

Render_World::~Render_World()
{
    /*
    for (auto a : all_objects)
        delete a;
    for (auto a : all_shaders)
        delete a;
    for (auto a : all_colors)
        delete a;
    for (auto a : lights)
        delete a;
    
        */
}

// Find and return the Hit structure for the closest intersection.  Be careful
// to ensure that hit.dist>=small_t.
__host__ __device__
std::pair<Flat_Shaded_Sphere, Hit> Render_World::Closest_Flat_Sphere_Intersection(const Ray &ray) const
{
    double min_t = std::numeric_limits<double>::max();
    Flat_Shaded_Sphere o;
    Hit h;
    std::pair<Flat_Shaded_Sphere, Hit> obj = {o, h};
    Hit hit_test;
    for (int i = 0; i<num_flat_shaded_spheres;i++)
    {
        hit_test = flat_shaded_spheres[i].sphere->Intersection(ray, -1);
        if (hit_test.dist >= small_t)
        {
            if (hit_test.dist < min_t)
            {
                min_t = hit_test.dist;
                obj.first = flat_shaded_spheres[i];
                obj.second = hit_test;
            }
        }
    }
    return obj;
}
__host__ __device__
std::pair<Phong_Shaded_Sphere, Hit> Render_World::Closest_Phong_Sphere_Intersection(const Ray &ray) const
{
    double min_t = std::numeric_limits<double>::max();
    Phong_Shaded_Sphere o;
    Hit h;
    std::pair<Phong_Shaded_Sphere, Hit> obj = {o, h};
    Hit hit_test;
    for (int i = 0; i<num_phong_shaded_spheres;i++)
    {
        hit_test = phong_shaded_spheres[i].sphere->Intersection(ray, -1);
        if (hit_test.dist >= small_t)
        {
            if (hit_test.dist < min_t)
            {
                min_t = hit_test.dist;
                obj.first = phong_shaded_spheres[i];
                obj.second = hit_test;
            }
        }
    }
    return obj;
}
__host__ __device__
std::pair<Flat_Shaded_Plane, Hit> Render_World::Closest_Flat_Plane_Intersection(const Ray &ray) const
{
    double min_t = std::numeric_limits<double>::max();
    Flat_Shaded_Plane o;
    Hit h;
    std::pair<Flat_Shaded_Plane, Hit> obj = {o, h};
    Hit hit_test;
    for (int i = 0; i<num_flat_shaded_planes;i++)
    {
        hit_test = flat_shaded_planes[i].plane->Intersection(ray, -1);
        if (hit_test.dist >= small_t)
        {
            if (hit_test.dist < min_t)
            {
                min_t = hit_test.dist;
                obj.first = flat_shaded_planes[i];
                obj.second = hit_test;
            }
        }
    }

    return obj;
    
}
__host__ __device__
std::pair<Phong_Shaded_Plane, Hit> Render_World::Closest_Phong_Plane_Intersection(const Ray &ray) const
{
    double min_t = std::numeric_limits<double>::max();
    Phong_Shaded_Plane o;
    Hit h;
    std::pair<Phong_Shaded_Plane, Hit> obj = {o, h};
    Hit hit_test;
    for (int i = 0; i<num_phong_shaded_planes;i++)
    {
        hit_test = phong_shaded_planes[i].plane->Intersection(ray, -1);
        if (hit_test.dist >= small_t)
        {
            if (hit_test.dist < min_t)
            {
                min_t = hit_test.dist;
                obj.first = phong_shaded_planes[i];
                obj.second = hit_test;
            }
        }
    }
    return obj;
}



// set up the initial view ray and call
__host__ __device__
void Render_World::Render_Pixel(const ivec2 &pixel_index)
{
    // set up the initial view ray here
    vec3 rayDir = (camera->World_Position(pixel_index) - camera->position).normalized();
    Ray ray(camera->position, rayDir);
    vec3 color = Cast_Ray(ray, 1);
    if(gpu_on){
        camera->Set_Pixel(pixel_index, Pixel_Color(color));
    }else{
        //printf("Color: (%f, %f, %f)\n",color[0],color[1],color[2]);
	    camera->Set_Pixel(pixel_index, Pixel_Color(color));

    }
}



void Render_World::Render()
{
    Timer timer;

    if (gpu_on) {
        //compute on gpu
        printf("Render image on gpu..."); fflush(stdout);
        //printf("Attempt Launch Kernel\n");
        startTime(&timer);

        // pls iterate through spheres, planes, shaders, lights, colors, flat_shaded_spheres, phong_shaded_spheres, flat_shaded_planes, phong_shaded_planes
        // check if the object types are loaded correctly and are properly in um
    

        //launch kernel        
        /*================================*/
        dim3 grid(ceil(camera->number_pixels[0]/(float)BLOCK_SIZE),ceil(camera->number_pixels[1]/(float)BLOCK_SIZE),1);
        dim3 block(BLOCK_SIZE,BLOCK_SIZE,1);
        
            Kernel_Render_Pixel<<<grid,block>>>(this);
            //Kernel_Render_Pixel<<<dim3(1,1,1),dim3(1,1,1)>>>(this);
            cudaDeviceSynchronize();
            
            
        /*================================*/

        stopTime(&timer); 
        //printf("Kernel Success\n");
        printf("%f s\n", elapsedTime(timer));

    }
    else {
        //compute on cpu
        printf("Render image on cpu..."); fflush(stdout);
        startTime(&timer);

        for (int j = 0; j < camera->number_pixels[1]; j++) {
            for (int i = 0; i < camera->number_pixels[0]; i++) {
                Render_Pixel(ivec2(i, j));
            }
        }

        //Render_Pixel(ivec2(320,240));

        stopTime(&timer); 
        printf("%f s\n", elapsedTime(timer));
    }
}

// cast ray and return the color of the closest intersected surface point,
// or the background color if there is no object intersection
__host__ __device__
vec3 Render_World::Cast_Ray(const Ray &ray, int recursion_depth) const
{
    // determine the color here (change if not at recursion limit)
    char intersect;
    vec3 color;
    vec3 q;
    vec3 n;
    if (recursion_depth > recursion_depth_limit)
    {
        color.make_zero();
        return color;
    }
    // Set color to background color as default
    Hit dummyHit;
    
    //printf("error check\n");
    std::pair<Flat_Shaded_Sphere, Hit> obj = Closest_Flat_Sphere_Intersection(ray);
    //printf("error check 2\n");
    std::pair<Phong_Shaded_Sphere, Hit> ps_obj = Closest_Phong_Sphere_Intersection(ray);
    //printf("error checl 3\n");
    std::pair<Flat_Shaded_Plane, Hit> fp_obj = Closest_Flat_Plane_Intersection(ray);
    std::pair<Phong_Shaded_Plane, Hit> pp_obj = Closest_Phong_Plane_Intersection(ray);
    //printf("Distances:\n\tFS:%f\n\tPS:%f\n\tFP:%f\n\tPP:%f\n\t",obj.second.dist,ps_obj.second.dist,fp_obj.second.dist,pp_obj.second.dist);
    if((obj.second.dist < ps_obj.second.dist || ps_obj.second.dist < small_t) && (obj.second.dist < fp_obj.second.dist || fp_obj.second.dist < small_t) && (obj.second.dist < pp_obj.second.dist || pp_obj.second.dist < small_t) && obj.second.dist >= small_t){
        intersect = 1;
    }else if((ps_obj.second.dist < fp_obj.second.dist|| fp_obj.second.dist < small_t) && (ps_obj.second.dist < pp_obj.second.dist|| pp_obj.second.dist < small_t) && ps_obj.second.dist >= small_t){
        intersect = 2;
    }else if((fp_obj.second.dist < pp_obj.second.dist|| pp_obj.second.dist < small_t) && fp_obj.second.dist >= small_t){
        intersect = 3;
    }else if(pp_obj.second.dist && pp_obj.second.dist >= small_t){
        intersect = 4;
    }else if(background_shader!=nullptr){
        intersect = 5;
    }else{
        intersect = 6;
    }
    switch(intersect){
        case 1:
            //printf("FS Hit\n");
		    q = ray.endpoint + (ray.direction * obj.second.dist);
		    n = (obj.first.sphere->Normal(ray, obj.second)).normalized();
		    color = obj.first.flat_shader->Shade_Flat_Sphere_Surface(*this, ray, obj.second, q, n, recursion_depth);
            break;
        case 2:
            //printf("PS Hit\n");
		    q = ray.endpoint + (ray.direction * ps_obj.second.dist);
		    n = (ps_obj.first.sphere->Normal(ray, ps_obj.second)).normalized();
		    color = ps_obj.first.phong_shader->Shade_Phong_Sphere_Surface(*this, ray, ps_obj.second, q, n, recursion_depth);
            break;
        case 3:
            //printf("FP Hit\n");
		    q = ray.endpoint + (ray.direction * fp_obj.second.dist);
		    n = (fp_obj.first.plane->Normal(ray, fp_obj.second)).normalized();
		    color = fp_obj.first.flat_shader->Shade_Flat_Plane_Surface(*this, ray, fp_obj.second, q, n, recursion_depth);
            break;
        case 4:
            //printf("PP Hit\n");
		    q = ray.endpoint + (ray.direction * pp_obj.second.dist);
		    n = (pp_obj.first.plane->Normal(ray, pp_obj.second)).normalized();
		    color = pp_obj.first.phong_shader->Shade_Phong_Plane_Surface(*this, ray, pp_obj.second, q, n, recursion_depth);
            break;
        case 5:
            //printf("Background Shader\n");
		    color = background_shader->Shade_Flat_Plane_Surface(*this, ray, dummyHit, ray.direction, ray.direction, 1);
            break;
        case 6:
            //printf("NULL Background\n");
		    color.make_zero();
            break;
    }   
    return color;
}
