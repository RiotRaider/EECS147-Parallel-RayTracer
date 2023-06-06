#ifndef __RENDER_WORLD_H__
#define __RENDER_WORLD_H__

#include <vector>
#include <utility>
#include "camera.cuh"
#include "object.cuh"
#include "flat_shader.cuh"
#include "light.cuh"
#include "ray.cuh"
#include "managed.cuh"
// #include "acceleration.h"

#define ARRAY_SIZE 32

class Light;
class Shader;
class Ray;
class Color;

struct Flat_Shaded_Sphere: public Managed
{
    const Sphere* sphere = nullptr;
    const Flat_Shader* flat_shader = nullptr;
};

struct Phong_Shaded_Sphere: public Managed
{
    const Sphere* sphere = nullptr;
    const Phong_Shader* phong_shader = nullptr;
};

struct Flat_Shaded_Plane: public Managed
{
    const Plane* plane = nullptr;
    const Flat_Shader* flat_shader = nullptr;
};

struct Phong_Shaded_Plane: public Managed
{
    const Plane* plane = nullptr;
    const Phong_Shader* phong_shader = nullptr;
};

class Render_World: public Managed
{

private:
    void _realloc() {
        if (camera != 0) {
            cudaFree(&camera);
        }
        cudaMallocManaged(&camera, sizeof(Camera));
        cudaDeviceSynchronize();

        if (ambient_color != 0) {
            cudaFree((void*)ambient_color);
        }
        cudaMallocManaged(&ambient_color, sizeof(Color));
        cudaDeviceSynchronize();

        if (background_shader != 0) {
            cudaFree((void*)background_shader)
        }
        cudaMallocManaged(&background_shader, sizeof(Flat_Shader));
        cudaDeviceSynchronize();

        //add lights, sphere, plane, flat shader, and phong shader to unified memory
        for(int i = 0; i<num_lights;i++){
            cudaMallocManaged(&lights[i],sizeof(Light));
            cudaDeviceSynchronize();
        }

        for(int i = 0; i<num_spheres;i++){
            cudaMallocManaged(&all_spheres[i],sizeof(Sphere));
            cudaDeviceSynchronize();
        }

        for(int i = 0; i<num_planes;i++){
            cudaMallocManaged(&all_planes[i],sizeof(Plane));
            cudaDeviceSynchronize();
        }

        for(int i = 0; i<num_flat_shaders;i++){
            cudaMallocManaged(&all_flat_shaders[i],sizeof(Flat_Shader));
            cudaDeviceSynchronize();
        }

        for(int i = 0; i<num_phong_shaders;i++){
            cudaMallocManaged(&all_phong_shaders[i],sizeof(Phong_Shader));
            cudaDeviceSynchronize();
        }

        for(int i = 0; i<num_colors;i++){
            cudaMallocManaged(&all_colors[i],sizeof(Color));
            cudaDeviceSynchronize();
        }

        for (int i = 0; i<num_flat_shaded_spheres; i++) {
            cudaMallocManaged(&flat_shaded_spheres[i].sphere, sizeof(Sphere));
            cudaMallocManaged(&flat_shaded_spheres[i].flat_shader, sizeof(Flat_Shader));
            cudaDeviceSynchronize();
        }

        for (int i = 0; i<num_phong_shaded_spheres; i++) {
            cudaMallocManaged(&phong_shaded_spheres[i].sphere, sizeof(Sphere));
            cudaMallocManaged(&phong_shaded_spheres[i].phong_shader, sizeof(Phong_Shader));
            cudaDeviceSynchronize();
        }

        for (int i = 0; i<num_flat_shaded_planes; i++) {
            cudaMallocManaged(&flat_shaded_planes[i].plane, sizeof(Plane));
            cudaMallocManaged(&flat_shaded_planes[i].flat_shader, sizeof(Flat_Shader));
            cudaDeviceSynchronize();
        }

        for (int i = 0; i<num_phong_shaded_planes; i++) {
            cudaMallocManaged(&phong_shaded_planes[i].plane, sizeof(Plane));
            cudaMallocManaged(&flat_shaded_planes[i].phong_shader, sizeof(Phong_Shader));
            cudaDeviceSynchronize();
        }

    }

public:
    Camera* camera = new Camera();

    bool gpu_on = false;

    // This is the background shader that you should use in case no other
    // objects are intersected.  If this pointer is null, then use black as the
    // color instead.
    const Shader* background_shader = nullptr;

    // Use these to get access to objects and lights in the scene.
    Flat_Shaded_Sphere flat_shaded_spheres[ARRAY_SIZE];
    Phong_Shaded_Sphere phong_shaded_spheres[ARRAY_SIZE];
    Flat_Shaded_Plane flat_shaded_planes[ARRAY_SIZE];
    Phong_Shaded_Plane phong_shaded_planes[ARRAY_SIZE];
    const Light* lights[ARRAY_SIZE];

    // Store pointers to these for deallocation.  You should not use these
    // directly.  Use the objects array above instead.
    Sphere* all_spheres[ARRAY_SIZE];
    Plane* all_planes[ARRAY_SIZE];
    Flat_Shader* all_flat_shaders[ARRAY_SIZE];
    Phong_Shader* all_phong_shaders[ARRAY_SIZE];
    Color* all_colors[ARRAY_SIZE];
    
    const Color* ambient_color = nullptr;
    double ambient_intensity = 0;

    bool enable_shadows = true;
    int recursion_depth_limit = 3;

    //counters for arrays
    int num_flat_shaded_spheres = 0; //num of flat shaded spheres
    int num_phong_shaded_spheres = 0; //num of phong shaded spheres
    int num_flat_shaded_planes = 0; //num of flat shaded planes
    int num_phong_shaded_planes = 0; //num of phong shaded planes
    int num_lights = 0; //num of lights (point light) 
    int num_spheres = 0; //num of spheres
    int num_planes = 0; //num of planes
    int num_flat_shaders = 0; //num of flatshaders
    int num_phong_shaders = 0; //num of phong shaders
    int num_colors = 0; //num all colors

    Render_World();
    Render_World(const Render_World& r)
        :gpu_on(r.gpu_on),ambient_intensity(r.ambient_intensity), num_flat_shaded_sphere(r.num_flat_shaded_sphere),
        num_phong_shaded_sphere(r.num_phong_shaded_sphere), num_flat_shaded_plane(r.num_flat_shaded_plane), num_phong_shaded_plane(r.num_phong_shaded_plane),
        num_lights(r.num_lights), num_spheres(r.num_spheres), num_planes(r.num_planes), num_flat_shaders(r.num_flat_shaders), 
        num_phong_shaders(r.num_phong_shaders), num_colors(r.num_colors), enable_shadows(r.enable_shadows),recursion_depth_limit(r.recursion_depth_limit)
    {

        _realloc();

        //copy data to class data
        ambient_color = r.ambient_color;
        background_shader = r.background_shader;
        memcpy(camera, r.camera, sizeof(Camera));

        for(int i = 0; i<num_shaded; i++){
            objects[i] = r.objects[i];
        }
        
        for(int i = 0; i < num_lights; i++){
            memcpy((void*)lights[i],r.lights[i],sizeof(Light));
        }

        for(int i = 0; i < num_objects; i++){
            memcpy(all_objects[i],r.all_objects[i],sizeof(Object));
        }

        for(int i = 0; i<num_shaders;i++){
            memcpy(all_shaders[i],r.all_shaders[i],sizeof(Shader));
        }

        for(int i = 0; i<num_colors;i++){
            memcpy(all_colors[i],r.all_colors[i],sizeof(Color));
        }

        /*
        ambient_color = r.ambient_color;
        background_shader = r.background_shader;
        for(int i = 0; i<num_shaded; i++){
            objects[i] = r.objects[i];
        }
        
        for(int i = 0; i<num_lights;i++){
            cudaMallocManaged(&lights[i],sizeof(Light));
            cudaDeviceSynchronize();
            memcpy((void*)lights[i],r.lights[i],sizeof(Light));
        }
        for(int i = 0; i<num_objects;i++){
            cudaMallocManaged(&all_objects[i],sizeof(Object));
            cudaDeviceSynchronize();
            memcpy(all_objects[i],r.all_objects[i],sizeof(Object));
        }
        for(int i = 0; i<num_shaders;i++){
            cudaMallocManaged(&all_shaders[i],sizeof(Shader));
            cudaDeviceSynchronize();
            memcpy(all_shaders[i],r.all_shaders[i],sizeof(Shader));
        }

        for(int i = 0; i<num_colors;i++){
            cudaMallocManaged(&all_colors[i],sizeof(Color));
            cudaDeviceSynchronize();
            memcpy(all_colors[i],r.all_colors[i],sizeof(Color));
        }

        cudaMallocManaged(&camera, sizeof(Camera));
        cudaDeviceSynchronize();
        memcpy(camera, r.camera, sizeof(Camera));
        */
    }

    ~Render_World();

    __host__ __device__
    void Render_Pixel(const ivec2& pixel_index);
    
    void Render();

    __host__ __device__
    vec3 Cast_Ray(const Ray& ray,int recursion_depth) const;

    __host__ __device__
    std::pair<Shaded_Object,Hit> Closest_Intersection(const Ray& ray) const;
};
#endif
