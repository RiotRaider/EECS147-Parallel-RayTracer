#include "light.h"
#include "parse.h"
#include "object.h"
#include "phong_shader.h"
#include "ray.h"
#include "render_world.h"

Phong_Shader::Phong_Shader(const Parse *parse, std::istream &in)
{
    in >> name;
    color_ambient = parse->Get_Color(in);
    color_diffuse = parse->Get_Color(in);
    color_specular = parse->Get_Color(in);
    in >> specular_power;
}

vec3 Phong_Shader::
    Shade_Surface(const Render_World &render_world, const Ray &ray, const Hit &hit,
                  const vec3 &intersection_point, const vec3 &normal, int recursion_depth) const
{
    // PIXEL TRACE
    Debug_Scope scope;
    // END PIXEL TRACE
    vec3 color;                                                              // variable for final color
    vec3 ambient, diffuse, specular;                                         // variables for total light components
    double intensity, phi;                                                   // variables for the intensity of a light component
    ambient = color_ambient->Get_Color(hit.uv) * render_world.ambient_intensity; // Capture ambient light
    if (render_world.ambient_color)
    {
        ambient *= render_world.ambient_color->Get_Color(hit.uv);
    }
    color = ambient;
    // PIXEL TRACE
    if (Debug_Scope::enable)
    {
        Pixel_Print("ambient: ", ambient);
    }
    // END PIXEL TRACE
    std::pair<Shaded_Object, Hit> obj; // object along ray from intersection back to light source
    vec3 shadeRay;                     // Shade ray for diffuse calculations.
    vec3 reflect;                      // Reflection Ray for specular
    bool lit = false;                  // flag for whether a light source is illuminating the point under inspection
    for (auto l : render_world.lights) // check all lights in world
    {
        shadeRay = l->position - intersection_point; // vector from intersection point back to light source

        if (render_world.enable_shadows) //Only need to check for light blocking if shadows are enabled
        {
            obj = render_world.Closest_Intersection(Ray(intersection_point, shadeRay.normalized())); //Find intersection along Shade Ray
            if (shadeRay.magnitude() > obj.second.dist && obj.second.dist != -1) //If the light is further than any object, it is blocked
            {
                // PIXEL TRACE
                if (Debug_Scope::enable)
                {
                    Pixel_Print("light ", l->name, " not visible; obscured by object ", obj.first.object->name, " at location ", intersection_point + (shadeRay.normalized() * obj.second.dist));
                }
                // END PIXEL TRACE
            }
            else //Any other case signal it is lit by that source
            {
                lit = true;
                // PIXEL TRACE
                if (Debug_Scope::enable)
                {
                    if (obj.second.dist == -1)
                    {
                        Pixel_Print("light ", l->name, " visible; closest object on ray too far away (light dist: ", shadeRay.magnitude(), "; object dist: inf)");
                    }
                }
                else
                {
                    Pixel_Print("light ", l->name, " visible; closest object on ray too far away (light dist: ", shadeRay.magnitude(), "; object dist: ", obj.second.dist, ")");
                }
                // END PIXEL TRACE
            }
        }
        else //If shadows are disabled, always calculate source contribution
        {
            lit = true;
        }
        if (lit)
        {
            intensity = dot(normal, shadeRay.normalized());
            if (intensity < 0)
            {
                intensity = 0;
            }
            diffuse = (l->Emitted_Light(shadeRay) * intensity) * color_diffuse->Get_Color(hit.uv);
            reflect = (2 * dot(normal, shadeRay.normalized()) * normal - (shadeRay.normalized()));
            phi = dot(-ray.direction, reflect);
            if (phi < 0)
            {
                phi = 0;
            }
            specular = (l->Emitted_Light(shadeRay) * pow(phi, specular_power)) * color_specular->Get_Color(hit.uv);
            // PIXEL TRACE
            if (Debug_Scope::enable)
            {
                Pixel_Print("shading for light ", l->name, ": diffuse: ", diffuse, "; specular: ", specular);
            }
            // END PIXEL TRACE
            lit = false;
            color += diffuse + specular;
        }
    }

    // PIXEL TRACE
    if (Debug_Scope::enable)
    {
        Pixel_Print("final color ", color);
    }
    // END PIXEL TRACE

    return color;
}
