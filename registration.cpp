#include "flat_shader.cuh"
//#include "mesh.h"
#include "parse.h"
#include "phong_shader.cuh"
#include "plane.cuh"
#include "point_light.cuh"
//#include "reflective_shader.h"
#include "sphere.cuh"
//#include "texture.h"
//#include "transparent_shader.h"

void Setup_Parsing(Parse& parse)
{
    parse.template Register_Sphere<Sphere>();
    parse.template Register_Plane<Plane>();
    //parse.template Register_Object<Mesh>();

    parse.template Register_Light<Light>();

    parse.template Register_Flat_Shader<Flat_Shader>();
    parse.template Register_Phong_Shader<Phong_Shader>();
    //parse.template Register_Shader<Reflective_Shader>();
    //parse.template Register_Shader<Transparent_Shader>();

    parse.template Register_Color<Color>();
    //parse.template Register_Color<Texture>();
}
