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
    // Reflected Ray Direction
    vec3 r;
    // Refracted Ray Direction
    vec3 t;
    // Reflected and Refracted Rays, endpoints at the intersection point with object
    Ray reflect, refract;
    reflect.endpoint = intersection_point;
    refract.endpoint = intersection_point;
    // Snell's Law Refraction indexes
    double n1, n2; // index of refraction (in and out)
    if (dot(ray.direction, normal) > 0)
    {
        n1 = index_of_refraction;
        n2 = 1.0;
    }
    else{
        n1 = 1.0;
        n2 = index_of_refraction;
    }
    // Schlick's Approximation variables
    double R_0, R;
    R_0 = pow(((n1 - n2) / (n1 + n2)), 2);
    R = R_0 + (1 - R_0) * pow((1 - dot(-ray.direction, normal)), 5);

    // Determine Refraction direction T
    double cos2 = 1 - pow((n1 / n2), 2) * (1 - pow(dot(normal, -ray.direction), 2)); //cosine of Theta 2 squared
    //If conditions for total internal reflection are met refracted ray contributes nothing
    if (cos2 < 0)
    {
        R = 1.0;
        cT.make_zero();
    }
    else
    {
        vec3 b = ((-ray.direction - dot(-ray.direction,normal)*normal)/sqrt(1-pow(dot(-ray.direction,normal),2))).normalized();
        cos2= sqrt(cos2); //cosine theta 2
        // determine refraction ray and refracted color
        refract.direction = (-cos2*normal-sqrt(1-pow(cos2,2))*b).normalized();
        cT = render_world.Cast_Ray(refract, ++recursion_depth);
    }
    // determine reflection ray and reflected color
    reflect.direction = (-(2 * dot(ray.direction, normal) * normal - ray.direction)).normalized();
    cR = render_world.Cast_Ray(reflect, ++recursion_depth);
    color = R*cR + (1-R)*cT;
    //PIXEL TRACE
        Debug_Scope scope;
        if(Debug_Scope::enable){
            Pixel_Print("transmitted ray: ", refract, "; transmitted color: ", cT);
            Pixel_Print("Schlick Reflectivity: ", R, "; combined color: ",color);
        }
    //END PIXEL TRACE
    // Gather inherent color of object
    c0 = shader->Shade_Surface(render_world, ray, hit, intersection_point, normal, recursion_depth);

    color = (opacity * c0) + (1 - opacity) * color;
    
    return color;
}
