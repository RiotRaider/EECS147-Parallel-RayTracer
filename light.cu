#include "light.cuh"
#include "parse.h"
#include "color.cuh"

Light::Light(const Light& l)
{
    position = l.position;
    _realloc();
    memcpy((void*)color,l.color,sizeof(Color));
    vec3 _color = color->Get_Color(vec2(0,0));
    vec3 old_color = l.color->Get_Color(vec2(0,0));
    printf("Light Copy Constructor Light to be copied Color: (%f, %f, %f)\n",old_color[0],old_color[1],old_color[2]);
    printf("Light Copy Constructor Light Color: (%f, %f, %f)\n",_color[0],_color[1],_color[2]);
}
Light::Light(const Parse* parse,std::istream& in)
{
    _realloc();
    in>>name>>position;
    memcpy((void*)color,parse->Get_Color(in),sizeof(Color));
    //color=parse->Get_Color(in);
    in>>brightness;
    vec3 _color = color->Get_Color(vec2(0,0));
    printf("Light Parse Constructor Light Color: (%f, %f, %f)\n",_color[0],_color[1],_color[2]);
}

__host__ __device__
vec3 Light::Emitted_Light(const vec3& vector_to_light) const
{
    //vec2 empty;
    vec3 _color = color->Get_Color(vec2(0,0))*brightness/(4*pi*vector_to_light.magnitude_squared());
    printf("Light Color: (%f, %f, %f)\n",_color[0],_color[1],_color[2]);
    return _color;
}
