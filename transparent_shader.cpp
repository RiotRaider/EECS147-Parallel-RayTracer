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
    // Rays for reflection and refraction anchored at intersection point
    Ray reflect, refract;
    reflect.endpoint = intersection_point;
    refract.endpoint = intersection_point;
    // Indexes of Refraction
    double n1, n2;
    // Find angle of view vector with surface normal
    double cos1 = dot(-ray.direction, normal);
    bool enter = (cos1 > 0) ? true : false; // Determine Enter or Exit
    if (enter)
    {
        n1 = 1.0;
        n2 = index_of_refraction;
    }
    else
    {
        n1 = index_of_refraction;
        n2 = 1.0;
    }

    // Calculate reflected ray and normalize it
    reflect.direction = ((2 * cos1 * normal) + ray.direction).normalized();

    // Schlick's Approximation
    double r0 = pow(((n1 - n2) / (n1 + n2)), 2);
    double R = r0;

    // Recursively cast reflected ray and collect returned color
    cR = render_world.Cast_Ray(reflect, recursion_depth + 1);
    // PIXEL TRACE
    Debug_Scope scope;
    if (Debug_Scope::enable)
        Pixel_Print("reflected ray: ", reflect, "; reflected color: ", cR);
    // END PIXEL TRACE

    // Find and cast Transmitted Ray
    double x = n1 / n2;
    double sini2 = 1 - pow(cos1, 2);
    if (!enter)
    {
        cos1 = dot(ray.direction, normal);
        sini2 = 1 - pow(cos1, 2);
    }
    if ((x * x * sini2) >= 1)
    { // complete internal reflection

        // PIXEL TRACE
        if (Debug_Scope::enable)
        {
            Pixel_Print("complete internal reflection;");
        }
        // END PIXEL TRACE
        R = 1;
    }
    else
    {
        if (enter)
        {
            refract.direction = x * ray.direction + normal * (x * cos1 - sqrt(1 - (x * x * sini2)));
        }
        else
        {
            refract.direction = x * ray.direction - normal * (x * cos1 - sqrt(1 - (x * x * sini2)));
        }

        refract.direction = refract.direction.normalized(); // normalize Transmitted Ray direction
        cT = render_world.Cast_Ray(refract, recursion_depth + 1);
        // Choose the cos(Theta) to use for Schlick's Approx
        if(!enter){
            cos1=dot(refract.direction,normal);
        }
        R += (1 - r0) * pow((1 - cos1), 5);
        // Calculate the reflective/refractive portion of the color
        color = ((1 - R) * cT);
        // PIXEL TRACE
        if (Debug_Scope::enable)
        {
            Pixel_Print("transmitted ray: ", refract, "; transmitted color: ", cT, "; Schlick Reflectivity: ", R, "; combined color: ", color);
        }
        // END PIXEL TRACE
    }
    color +=R*cR;
    //  Gather inherent color of object
    c0 = shader->Shade_Surface(render_world, ray, hit, intersection_point, normal, recursion_depth + 1);

    color = (opacity * c0) + (1 - opacity) * color;
    // PIXEL TRACE
    if (Debug_Scope::enable)
    {
        Pixel_Print("object color: ", c0, "; final color: ", color);
    }
    // END PIXEL TRACE
    return color;
}
