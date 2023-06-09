#include "light.cuh"
#include "parse.h"
#include "color.cuh"

Light::Light(const Light& l)
{
    position = l.position;
    cudaMallocManaged(&color, sizeof(Color));
    memcpy((void*)color,l.color,sizeof(Color));
}
Light::Light(const Parse* parse,std::istream& in)
{
    cudaMallocManaged(&color, sizeof(Color));
    in>>name>>position;
    memcpy((void*)color,parse->Get_Color(in),sizeof(Color));
    //color=parse->Get_Color(in);
    in>>brightness;
}

__host__ __device__
vec3 Light::Emitted_Light(const vec3& vector_to_light) const
{
    vec2 empty;
    vec3 _color = color->Get_Color(empty)*brightness/(4*pi*vector_to_light.magnitude_squared());
    //printf("Light Color: (%f, %f, %f)\n",_color[0],_color[1],_color[2]);
    return _color;
}
