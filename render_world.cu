// Student Name: Justin Sanders
// Student ID: 862192429

#include <stdio.h>
#include <stdlib.h>

#include "render_world.h"
#include "flat_shader.h"
#include "object.cuh"
#include "light.h"
#include "ray.cuh"

#include "support.h"
#include "kernel.cuh"

extern bool enable_acceleration;

Render_World::~Render_World()
{
    
    for (auto a : all_objects)
        delete a;
    for (auto a : all_shaders)
        delete a;
    for (auto a : all_colors)
        delete a;
    for (auto a : lights)
        delete a;
    
        
}

// Find and return the Hit structure for the closest intersection.  Be careful
// to ensure that hit.dist>=small_t.
std::pair<Shaded_Object, Hit> Render_World::Closest_Intersection(const Ray &ray) const
{
    double min_t = std::numeric_limits<double>::max();
    Shaded_Object o;
    Hit h;
    std::pair<Shaded_Object, Hit> obj = {o, h};
    Hit hit_test;
    for (auto a : this->objects)
    {
        hit_test = a.object->Intersection(ray, -1);
        if (hit_test.dist >= small_t)
        {
            if (hit_test.dist < min_t)
            {
                min_t = hit_test.dist;
                obj.first = a;
                obj.second = hit_test;
            }
        }
    }
    return obj;
}

// set up the initial view ray and call
void Render_World::Render_Pixel(const ivec2 &pixel_index)
{
    // set up the initial view ray here
    vec3 rayDir = (camera.World_Position(pixel_index) - camera.position).normalized();
    Ray ray(camera.position, rayDir);
    vec3 color = Cast_Ray(ray, 1);
    camera.Set_Pixel(pixel_index, Pixel_Color(color));
}

void Render_World::Render()
{
    Timer timer;

    if (gpu_on) {
        //compute on gpu
        printf("Render image on gpu..."); fflush(stdout);
        startTime(&timer);

        //launch kernel
        //temporary - test launch kernel with vec class

        /*================================*/

        //Hit
        
        Hit *e = new Hit;
        Hit *f = new Hit;
        
        for (int i = 0; i < 2; i++) {
            e->uv[i] = 10;
        }

        e->dist = 100;
        e->triangle = 5;

        for (int i = 0; i < 2; i++) {
            f->uv[i] = 20;
        }

        f->dist = 200;
        f->triangle = 10;

        printf("\nHit:\nOn host (print) e: uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);
        printf("On host (print) f: uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", f->uv[0], f->uv[1], f->dist, f->triangle);

        //add
        e->uv += f->uv;
        printf("On host (after e + f): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);
        
        launch_by_pointer_hit(e, f);
        printf("On host (after by-pointer): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);

        launch_by_ref_hit(*e, *f);
        printf("On host (after by-ref): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);

        launch_by_value_hit(*e, *f);
        printf("On host (after by-value): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);
        

        //Ray
        Ray *q = new Ray;
        //Ray *r = new Ray;
        
        q->endpoint = {10, 20, 30};
        q->direction = {0.10, 0.20, 0.30};
        
        vec3 q_ray_point = q->Point(5);

        printf("\nRay:\nOn host (print) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f), point=(%.2f, %.2f, %.2f)\n", 
            q->endpoint[0], q->endpoint[1], q->endpoint[2], q->direction[0], q->direction[1], q->direction[2], q_ray_point[0], q_ray_point[1], q_ray_point[2]);
        
        
        launch_by_pointer_ray(q);
        printf("On host (after by-pointer) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f), point=(%.2f, %.2f, %.2f)\n", 
            q->endpoint[0], q->endpoint[1], q->endpoint[2], q->direction[0], q->direction[1], q->direction[2], q_ray_point[0], q_ray_point[1], q_ray_point[2]);
        
        launch_by_ref_ray(*q);
        printf("On host (after by-ref) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f), point=(%.2f, %.2f, %.2f)\n", 
            q->endpoint[0], q->endpoint[1], q->endpoint[2], q->direction[0], q->direction[1], q->direction[2], q_ray_point[0], q_ray_point[1], q_ray_point[2]);

        launch_by_value_ray(*q);
        printf("On host (after by-val) q: endpoint=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f), point=(%.2f, %.2f, %.2f)\n", 
            q->endpoint[0], q->endpoint[1], q->endpoint[2], q->direction[0], q->direction[1], q->direction[2], q_ray_point[0], q_ray_point[1], q_ray_point[2]);
        
        //delete these pointers
        delete e;
        delete f;
        delete q;

        cudaDeviceReset();

        /*================================*/

        stopTime(&timer); 
        printf("\n...%f s\n", elapsedTime(timer));
    }
    else {
        //compute on cpu
        printf("Render image on cpu..."); fflush(stdout);
        startTime(&timer);

        for (int j = 0; j < camera.number_pixels[1]; j++) {
            for (int i = 0; i < camera.number_pixels[0]; i++) {
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
    vec3 color;
    if (recursion_depth > recursion_depth_limit)
    {
        color.make_zero();
        return color;
    }
    // Set color to background color as default
    Hit dummyHit;
    
    
    // determine the color here (change if not at recursion limit)
    std::pair<Shaded_Object, Hit> obj = Closest_Intersection(ray);
    if (obj.first.object != nullptr)
    {
        vec3 q = ray.endpoint + (ray.direction * obj.second.dist);
        vec3 n = (obj.first.object->Normal(ray, obj.second)).normalized();
        color = obj.first.shader->Shade_Surface(*this, ray, obj.second, q, n, recursion_depth);
    }else{
        if (background_shader == nullptr)
        {
            color.make_zero();
        }
        else
        {
            color = background_shader->Shade_Surface(*this, ray, dummyHit, ray.direction, ray.direction, 1);
        }
    }

    return color;
}
