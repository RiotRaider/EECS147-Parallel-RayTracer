// Student Name: Justin Sanders
// Student ID: 862192429

#include <stdio.h>
#include <stdlib.h>

#include "render_world.h"
#include "flat_shader.h"
#include "object.h"
#include "light.h"
#include "ray.h"

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
    // PIXEL TRACE
    Debug_Scope scope;
    // END PIXEL TRACE

    double min_t = std::numeric_limits<double>::max();
    Shaded_Object o;
    Hit h;
    std::pair<Shaded_Object, Hit> obj = {o, h};
    Hit hit_test;
    bool intersect = false;
    for (auto a : this->objects)
    {
        hit_test = a.object->Intersection(ray, -1);
        if (hit_test.dist >= small_t)
        {
            // PIXEL TRACE
            if (Debug_Scope::enable)
            {
                Pixel_Print("intersect test with ", a.object->name, "; hit: ", hit_test);
            }
            // END PIXEL TRACE
            if (hit_test.dist < min_t)
            {
                min_t = hit_test.dist;
                obj.first = a;
                obj.second = hit_test;
            }
            intersect = true;
        }
        else
        {
            // PIXEL TRACE
            if (Debug_Scope::enable)
            {
                Pixel_Print("no intersection with ", a.object->name);
            }
            // END PIXEL TRACE
        }
    }
    if (intersect)
    {
        // PIXEL TRACE
        if (Debug_Scope::enable)
        {
            Pixel_Print("closest intersection; obj: ", obj.first.object->name, "; hit: ", obj.second);
        } // END PIXEL TRACE
    }
    else
    {
        // PIXEL TRACE
        if (Debug_Scope::enable)
        {
            Pixel_Print("closest intersection; none");
        } // END PIXEL TRACE
    }
    return obj;
}

// set up the initial view ray and call
void Render_World::Render_Pixel(const ivec2 &pixel_index)
{
    // PIXEL TRACE
    Debug_Scope scope;
    if (Debug_Scope::enable)
    {
        Pixel_Print("debug pixel: -x ", pixel_index.x[0], " -y ", pixel_index.x[1]);
    }
    // END PIXEL TRACE

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
        printf("Render image on gpu...\n"); fflush(stdout);
        startTime(&timer);

        //launch kernel
        //temporary - test launch kernel with vec class

        /*=================TEMPORARY===============*/
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

        printf("On host (print) e: uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);
        printf("On host (print) f: uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", f->uv[0], f->uv[1], f->dist, f->triangle);

        //add
        e->uv += f->uv;
        printf("On host (after e + f): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);
        
        launch_by_pointer(e, f);

        printf("On host (after by-pointer): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);

        launch_by_ref(*e, *f);

        printf("On host (after by-ref): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);

        launch_by_value(*e, *f);

        printf("On host (after by-value): uv=(%.2f, %.2f), dist=%.2f, triangle=%d\n", e->uv[0], e->uv[1], e->dist, e->triangle);

        delete e;

        cudaDeviceReset();

        /*=================TEMPORARY===============*/

        stopTime(&timer); 
        printf("%f s\n", elapsedTime(timer));
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
    // PIXEL TRACE
    Debug_Scope scope;
    if (Debug_Scope::enable)
    {
        Pixel_Print("cast ray ", ray);
    }
    // END PIXEL TRACE
    vec3 color;
    if (recursion_depth > recursion_depth_limit)
    {
        // PIXEL TRACE
        if (Debug_Scope::enable)
        {
            Pixel_Print("ray too deep; return black");
        }
        // END PIXEL TRACE
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
        // PIXEL TRACE
        if (Debug_Scope::enable)
        {
            Pixel_Print("call Shade_Surface with location ", q, "; normal: ", n);
        }
        // END PIXEL TRACE
        color = obj.first.shader->Shade_Surface(*this, ray, obj.second, q, n, recursion_depth);
    }else{
        if (background_shader == nullptr)
        {
            // PIXEL TRACE
            if (Debug_Scope::enable)
            {
                Pixel_Print("no background; return black");
            }
            // END PIXEL TRACE
            color.make_zero();
        }
        else
        {
            
            color = background_shader->Shade_Surface(*this, ray, dummyHit, ray.direction, ray.direction, 1);
            // PIXEL TRACE
            if (Debug_Scope::enable)
            {
                Pixel_Print("background hit; return color: ", color);
            }
            // END PIXEL TRACE
        }
    }

    return color;
}
