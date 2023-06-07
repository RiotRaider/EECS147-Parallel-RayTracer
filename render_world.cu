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
std::pair<Phong_Shaded_Sphere, Hit> Render_World::Closest_Phong_Sphere_Intersection(const Ray &ray) const
{
    double min_t = std::numeric_limits<double>::max();
    Phong_Shaded_Sphere o;
    Hit h;
    std::pair<Phong_Shaded_Sphere, Hit> obj = {o, h};
    Hit hit_test;
    for (int i = 0; i<num_flat_shaded_spheres;i++)
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
    //vec3 color = Cast_Ray(ray, 1);
    //camera->Set_Pixel(pixel_index, Pixel_Color(color));
    if(gpu_on){
        camera->Set_Pixel(pixel_index, Pixel_Color(vec3(255,0,255)));
    }else{
        camera->Set_Pixel(pixel_index, Pixel_Color(vec3(0,255,255)));
    }
}



void Render_World::Render()
{
    Timer timer;

    if (gpu_on) {
        //compute on gpu
        printf("Render image on gpu...\n"); fflush(stdout);
        startTime(&timer);

        // pls iterate through spheres, planes, shaders, lights, colors, flat_shaded_spheres, phong_shaded_spheres, flat_shaded_planes, phong_shaded_planes
        // check if the object types are loaded correctly and are properly in um
    

        //launch kernel        
        /*================================*/
        dim3 grid(ceil(camera->number_pixels[0]/(float)16),ceil(camera->number_pixels[1]/(float)16),1);
        dim3 block(16,16,1);
        printf("Attempt Launch Kernel\n");
            Kernel_Render_Pixel<<<grid,block>>>(this);
            cudaDeviceSynchronize();
            printf("Kernel Success\n");
            
        /*================================*/

        stopTime(&timer); 
        printf("\n...%f s\n", elapsedTime(timer));
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

        stopTime(&timer); 
        printf("%f s\n", elapsedTime(timer));
    }
}

// cast ray and return the color of the closest intersected surface point,
// or the background color if there is no object intersection

vec3 Render_World::Cast_Ray(const Ray &ray, int recursion_depth) const
{
    // determine the color here (change if not at recursion limit)
    vec3 color;
    if (recursion_depth > recursion_depth_limit)
    {
        color.make_zero();
        return color;
    }
    // Set color to background color as default
    Hit dummyHit;
    
    
    std::pair<Flat_Shaded_Sphere, Hit> obj = Closest_Flat_Sphere_Intersection(ray);
    std::pair<Phong_Shaded_Sphere, Hit> ps_obj = Closest_Phong_Sphere_Intersection(ray);
    std::pair<Flat_Shaded_Plane, Hit> fp_obj = Closest_Flat_Plane_Intersection(ray);
    std::pair<Phong_Shaded_Plane, Hit> pp_obj = Closest_Phong_Plane_Intersection(ray);

    double closest_hit = std::min(std::min(fp_obj.second.dist, pp_obj.second.dist), std::min(obj.second.dist, ps_obj.second.dist));
    if (closest_hit == ps_obj.second.dist) {
	    if (ps_obj.first.sphere != nullptr){
		vec3 q = ray.endpoint + (ray.direction * ps_obj.second.dist);
		vec3 n = (ps_obj.first.sphere->Normal(ray, ps_obj.second)).normalized();
		color = ps_obj.first.phong_shader->Shade_Phong_Sphere_Surface(*this, ray, ps_obj.second, q, n, recursion_depth);
	    }else{
		if (background_shader == nullptr){
		    color.make_zero();
		} else {
		    color = background_shader->Shade_Flat_Sphere_Surface(*this, ray, dummyHit, ray.direction, ray.direction, 1);
		}
	    }
     }

	else if (closest_hit == fp_obj.second.dist) {
    	    if (fp_obj.first.plane!= nullptr){
		vec3 q = ray.endpoint + (ray.direction * fp_obj.second.dist);
		vec3 n = (fp_obj.first.plane->Normal(ray, fp_obj.second)).normalized();
		color = fp_obj.first.flat_shader->Shade_Flat_Plane_Surface(*this, ray, fp_obj.second, q, n, recursion_depth);
	    }else{
		if (background_shader == nullptr){
		    color.make_zero();
		} else {
		    color = background_shader->Shade_Flat_Plane_Surface(*this, ray, dummyHit, ray.direction, ray.direction, 1);
		}
	    }
	}
	else if (closest_hit == pp_obj.second.dist) {
		 if (pp_obj.first.plane!= nullptr){
		vec3 q = ray.endpoint + (ray.direction * pp_obj.second.dist);
		vec3 n = (pp_obj.first.plane->Normal(ray, pp_obj.second)).normalized();
		color = pp_obj.first.phong_shader->Shade_Phong_Plane_Surface(*this, ray, pp_obj.second, q, n, recursion_depth);
	    }else{
		if (background_shader == nullptr){
		    color.make_zero();
		} else {
		    color = background_shader->Shade_Flat_Plane_Surface(*this, ray, dummyHit, ray.direction, ray.direction, 1);
		}
	    }
	} 
	else {
	    if (obj.first.sphere != nullptr){
		vec3 q = ray.endpoint + (ray.direction * obj.second.dist);
		vec3 n = (obj.first.sphere->Normal(ray, obj.second)).normalized();
		color = obj.first.flat_shader->Shade_Flat_Sphere_Surface(*this, ray, obj.second, q, n, recursion_depth);
	    }else{
		if (background_shader == nullptr){
		    color.make_zero();
		} else {
		    color = background_shader->Shade_Flat_Sphere_Surface(*this, ray, dummyHit, ray.direction, ray.direction, 1);
		}
	    }
	}

	    

    //TODO able to handle objects other than flat_shaded_spheres;
   
    return color;
}
