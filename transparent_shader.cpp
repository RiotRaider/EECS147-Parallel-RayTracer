#include "transparent_shader.h"
#include "parse.h"
#include "ray.h"
#include "render_world.h"

Transparent_Shader::
    Transparent_Shader(const Parse *parse, std::istream &in)
{
    in >> name >> index_of_refraction >> opacity;
    shader = parse->Get_Shader(in);
    assert(index_of_refraction >= 1.0);
}

// Use opacity to determine the contribution of this->shader and the Schlick
// approximation to compute the reflectivity.  This routine shades transparent
// objects such as glass.  Note that the incoming and outgoing indices of
// refraction depend on whether the ray is entering the object or leaving it.
// You may assume that the object is surrounded by air with index of refraction
// 1.
vec3 Transparent_Shader::
    Shade_Surface(const Render_World &render_world, const Ray &ray, const Hit &hit,
                  const vec3 &intersection_point, const vec3 &normal, int recursion_depth) const
{
    // Object Color, Reflected Color, Transmitted Color, Total Color
    vec3 c0, cR, cT, color;
    //Rays for reflection and refraction anchored at intersection point
    Ray reflect, refract;
    reflect.endpoint = intersection_point;
    refract.endpoint = intersection_point;
    //Find angle of view vector with surface normal
    double cos1 = dot(-ray.direction, normal);

    //Calculate reflected ray and normalize it
    reflect.direction = ((2 * cos1 * normal) + ray.direction).normalized();

    // Schlick's Approximation
    
        //Recursively cast reflected ray and collect returned color
        cR = render_world.Cast_Ray(reflect, recursion_depth+1);
        //PIXEL TRACE
        Debug_Scope scope;
        if(Debug_Scope::enable)
            Pixel_Print("reflected ray: ", reflect,"; reflected color: ", cR);
        //END PIXEL TRACE

        //Find and cast Transmitted Ray
        
        color = /*(R * */cR/*)+((1-R)*cT)*/;
    /*/ PIXEL TRACE
    if (Debug_Scope::enable)
    {
        Pixel_Print("transmitted ray: ", refract, "; transmitted color: ", cT,"; Schlick Reflectivity: ", R, "; combined color: ", color);
    }
    */// END PIXEL TRACE
    //  Gather inherent color of object
    c0 = shader->Shade_Surface(render_world, ray, hit, intersection_point, normal, recursion_depth);

    color = (opacity * c0) + (1 - opacity) * color;
    // PIXEL TRACE
    if (Debug_Scope::enable)
    {
        Pixel_Print("object color: ", c0, "; final color: ", color);
    }
    // END PIXEL TRACE
    return color;
}
