#ifndef __RENDER_WORLD_H__
#define __RENDER_WORLD_H__

#include <vector>
#include <utility>
#include "camera.cuh"
#include "object.cuh"
#include "managed.cuh"
// #include "acceleration.h"

#define ARRAY_SIZE 32

class Light;
class Shader;
class Ray;
class Color;

struct Shaded_Object
{
    const Object* object = nullptr;
    const Shader* shader = nullptr;
};

class Render_World: public Managed
{
public:
    Camera camera;

    bool gpu_on = false;

    // This is the background shader that you should use in case no other
    // objects are intersected.  If this pointer is null, then use black as the
    // color instead.
    const Shader* background_shader = nullptr;

    // Use these to get access to objects and lights in the scene.
    Shaded_Object objects[ARRAY_SIZE];
    const Light* lights[ARRAY_SIZE];

    // Store pointers to these for deallocation.  You should not use these
    // directly.  Use the objects array above instead.
    Object* all_objects[ARRAY_SIZE];
    Shader* all_shaders[ARRAY_SIZE];
    Color* all_colors[ARRAY_SIZE];
    
    const Color* ambient_color = nullptr;
    double ambient_intensity = 0;

    bool enable_shadows = true;
    int recursion_depth_limit = 3;

    //counters for arrays
    int num_shaded = 0; //Shaded Objects
    int num_lights = 0; //Lights
    int num_objects = 0; //all objects
    int num_shaders = 0; //all shaders
    int num_colors = 0; //all colors

//     Acceleration acceleration;

    Render_World() = default;
    Render_World(const Render_World& r)
        :camera(r.camera), gpu_on(r.gpu_on),ambient_intensity(r.ambient_intensity),num_shaded(r.num_shaded),num_lights(r.num_lights),num_objects(r.num_objects),
        num_shaders(r.num_shaders),num_colors(r.num_colors),enable_shadows(r.enable_shadows),recursion_depth_limit(r.recursion_depth_limit)
    {
        ambient_color=r.ambient_color;
        background_shader=r.background_shader;
        for(int i = 0; i<num_shaded;i++){
            objects[i] = r.objects[i];
        }
        for(int i = 0; i<num_lights;i++){
            cudaMallocManaged(&lights[i],sizeof(Light)*1)
        }

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
