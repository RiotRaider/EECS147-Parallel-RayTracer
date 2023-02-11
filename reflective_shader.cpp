#include "reflective_shader.h"
#include "parse.h"
#include "ray.h"
#include "render_world.h"

Reflective_Shader::Reflective_Shader(const Parse *parse, std::istream &in)
{
    in >> name;
    shader = parse->Get_Shader(in);
    in >> reflectivity;
    reflectivity = std::max(0.0, std::min(1.0, reflectivity));
}

vec3 Reflective_Shader::
    Shade_Surface(const Render_World &render_world, const Ray &ray, const Hit &hit,
                  const vec3 &intersection_point, const vec3 &normal, int recursion_depth) const
{
    // determine the color
    vec3 c0,cR,color;
    Ray reflection;
    reflection.endpoint=intersection_point;
    reflection.direction = ((2 * dot(-ray.direction, normal) * normal) + ray.direction).normalized();
    c0 = shader->Shade_Surface(render_world,ray,hit,intersection_point,normal,1);
    cR = render_world.Cast_Ray(reflection,recursion_depth+1);
    color = c0*(1-reflectivity) + cR*(reflectivity);
    //PIXEL TRACE
        Debug_Scope scope;
        if(Debug_Scope::enable)
            Pixel_Print("reflected ray: ", ray,"; reflected color: ", color);
    //END PIXEL TRACE
    return color;
}
