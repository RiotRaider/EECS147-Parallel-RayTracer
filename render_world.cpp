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
    //PIXEL TRACE
    Debug_Scope scope;
    //END PIXEL TRACE
    
    double min_t = std::numeric_limits<double>::max();
    Shaded_Object o;
    Hit h;
    std::pair<Shaded_Object,Hit> obj={o,h};
    Hit hit_test;
    bool intersect = false;
    for(auto a:this->objects){
        hit_test=a.object->Intersection(ray,-1);
        if(hit_test.dist>=small_t && hit_test.dist<min_t){
            //PIXEL TRACE
            if(Debug_Scope::enable){
                Pixel_Print("intersect test with ",a.object->name,"; hit: ", hit_test);
            }
            //END PIXEL TRACE
            min_t = hit_test.dist;
            obj.first = a;
            obj.second = hit_test;
            intersect = true;
        }else{
            //PIXEL TRACE
            if(Debug_Scope::enable){
                Pixel_Print("no intersection with ",a.object->name);
            }
            //END PIXEL TRACE
        }
    }   
    if(intersect){
        //PIXEL TRACE
        if(Debug_Scope::enable){
            Pixel_Print("closest intersection; obj: ", obj.first.object->name,"; hit: ", obj.second);
        }//END PIXEL TRACE
    }else{
        //PIXEL TRACE
        if(Debug_Scope::enable){
            Pixel_Print("closest intersection; none");
        }//END PIXEL TRACE
    }
    return obj;
}

// set up the initial view ray and call
void Render_World::Render_Pixel(const ivec2& pixel_index)
{
    //PIXEL TRACE
    Debug_Scope scope;
        if(Debug_Scope::enable){
            Pixel_Print("debug pixel: -x ", pixel_index.x[0]," -y ",pixel_index.x[1]);
        }
    //END PIXEL TRACE
    
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
    //PIXEL TRACE
    Debug_Scope scope;
        if(Debug_Scope::enable){
            Pixel_Print("cast ray ", ray);
        }
    //END PIXEL TRACE
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
        vec3 n = (obj.first.object->Normal(ray,obj.second)).normalized();
        //PIXEL TRACE
            if(Debug_Scope::enable){
                Pixel_Print("call Shade_Surface with location ", q ,"; normal: ", n);
            }
        //END PIXEL TRACE
        color = obj.first.shader->Shade_Surface(*this,ray,obj.second,q,n,1);
    }
    return color;
}
