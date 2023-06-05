#ifndef __RENDER_WORLD_H__
#define __RENDER_WORLD_H__

#include <vector>
#include <utility>
#include "camera.cuh"
#include "object.cuh"
#include "managed.cuh"
// #include "acceleration.h"

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
    std::vector<Shaded_Object> objects;
    std::vector<const Light*> lights;

    // Store pointers to these for deallocation.  You should not use these
    // directly.  Use the objects array above instead.
    std::vector<Object*> all_objects;
    std::vector<Shader*> all_shaders;
    std::vector<Color*> all_colors;
    
    const Color* ambient_color = nullptr;
    double ambient_intensity = 0;

    bool enable_shadows = true;
    int recursion_depth_limit = 3;

//     Acceleration acceleration;

    Render_World() = default;
    Render_World(const Render_World& r){
        ambient_color = r.ambient_color;
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
