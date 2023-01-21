//Student Name: Justin Sanders
//Student ID: 862192429
#include "render_world.h"
#include "flat_shader.h"
#include "object.h"
#include "light.h"
#include "ray.h"

extern bool enable_acceleration;

Render_World::~Render_World()
{
    for(auto a:all_objects) delete a;
    for(auto a:all_shaders) delete a;
    for(auto a:all_colors) delete a;
    for(auto a:lights) delete a;
}

// Find and return the Hit structure for the closest intersection.  Be careful
// to ensure that hit.dist>=small_t.
std::pair<Shaded_Object,Hit> Render_World::Closest_Intersection(const Ray& ray) const
{
    
    double min_t = std::numeric_limits<double>::max();
    
    Hit hit;
    Shaded_Object o;
    std::pair<Shaded_Object,Hit> obj = {o,hit};
    Hit hit_test;
    for(auto a:this->objects){
        hit_test=a.object->Intersection(ray,-1);
        if(hit_test.dist>=small_t && hit_test.dist<min_t){
            min_t = hit_test.dist;
            obj.first = a;
            obj.second = hit_test;
        }
    }
    return obj;
}

// set up the initial view ray and call
void Render_World::Render_Pixel(const ivec2& pixel_index)
{
    // set up the initial view ray here
    vec3 rayDir = (camera.World_Position(pixel_index) - camera.position).normalized();
    Ray ray(camera.position,rayDir);
    vec3 color=Cast_Ray(ray,1);
    camera.Set_Pixel(pixel_index,Pixel_Color(color));
}

void Render_World::Render()
{
    for(int j=0;j<camera.number_pixels[1];j++)
        for(int i=0;i<camera.number_pixels[0];i++)
            Render_Pixel(ivec2(i,j));
}

// cast ray and return the color of the closest intersected surface point,
// or the background color if there is no object intersection
vec3 Render_World::Cast_Ray(const Ray& ray,int recursion_depth) const
{
    vec3 color;
    // determine the color here
    std::pair<Shaded_Object,Hit> obj = Closest_Intersection(ray);
    if(obj.first.object == nullptr){
            if(background_shader==nullptr){
                color[0]=0;
                color[1]=0;
                color[2]=0;
            }
            else{
                color=background_shader->Shade_Surface(*this,ray,obj.second,ray.direction,ray.direction,1);
            }
    }else{
        vec3 q = ray.endpoint+(ray.direction*obj.second.dist);
        vec3 n = obj.first.object->Normal(ray,obj.second);
        color = obj.first.shader->Shade_Surface(*this,ray,obj.second,q,n,1);
    }
    return color;
}
