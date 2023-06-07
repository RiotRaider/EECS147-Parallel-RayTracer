#include "flat_shader.cuh"
#include "parse.h"

Flat_Shader::Flat_Shader(const Parse* parse,std::istream& in)
{
    in>>name;

    _realloc();
    color=parse->Get_Color(in);
}

Flat_Shader::Flat_Shader(const Flat_Shader& s){
    _realloc();
    color = s.color;
}

__host__ __device__
vec3 Flat_Shader::
Shade_Flat_Sphere_Surface(const Render_World& render_world,const Ray& ray,const Hit& hit,
    const vec3& intersection_point,const vec3& normal,int recursion_depth) const
{
    return color->Get_Color(hit.uv);
}
__host__ __device__
vec3 Flat_Shader::
Shade_Flat_Plane_Surface(const Render_World& render_world,const Ray& ray,const Hit& hit,
    const vec3& intersection_point,const vec3& normal,int recursion_depth) const
{
    return color->Get_Color(hit.uv);
}
