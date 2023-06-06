#include "light.cuh"
#include "parse.h"
#include "color.cuh"

Light::Light(const Light& l)
{
    position = l.position;
    cudaMallocManaged(&color, sizeof(Color));
    color = l.color;
}
Light::Light(const Parse* parse,std::istream& in)
{
    in>>name>>position;
    color=parse->Get_Color(in);
    in>>brightness;
}

vec3 Light::Emitted_Light(const vec3& vector_to_light) const
{
    return color->Get_Color({})*brightness/(4*pi*vector_to_light.magnitude_squared());
}