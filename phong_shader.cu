#include "light.cuh"
#include "parse.h"
#include "object.cuh"
#include "phong_shader.cuh"
#include "ray.cuh"
#include "render_world.cuh"

Phong_Shader::Phong_Shader(const Parse *parse, std::istream &in)
{
    _realloc();
    in >> name;
    color_ambient = parse->Get_Color(in);
    color_diffuse = parse->Get_Color(in);
    color_specular = parse->Get_Color(in);
    in >> specular_power;
}

//copy constructor
Phong_Shader::Phong_Shader(const Phong_Shader& s){
   _realloc();
   specular_power = s.specular_power;
   color_ambient = s.color_ambient;
   color_diffuse = s.color_diffuse;
   color_specular = s.color_specular;
}



/*Shade Surface fo Phong Shaded Spheres*/
__host__ __device__
vec3 Phong_Shader::
    Shade_Phong_Sphere_Surface(const Render_World &render_world, const Ray &ray, const Hit &hit,
                  const vec3 &intersection_point, const vec3 &normal, int recursion_depth) const
{
    vec3 color;                                                              // variable for final color
    vec3 ambient, diffuse, specular;                                         // variables for total light components
    double intensity, phi;                                                   // variables for the intensity of a light component
    ambient = color_ambient->Get_Color(hit.uv) * render_world.ambient_intensity; // Capture ambient light
    if (render_world.ambient_color)
    {
        ambient *= render_world.ambient_color->Get_Color(hit.uv);
    }

    //printf("Ambient: (%f, %f, %f)\n",color[0],color[1],color[2]);
    color = ambient;
    std::pair<Phong_Shaded_Sphere, Hit> obj; // object along ray from intersection back to light source
    vec3 shadeRay;                     // Shade ray for diffuse calculations.
    vec3 reflect;                      // Reflection Ray for specular
    bool lit = false;                  // flag for whether a light source is illuminating the point under inspection
    for (int i = 0; i<render_world.num_lights;i++) // check all lights in world
    {
        shadeRay = render_world.lights[i]->position - intersection_point; // vector from intersection point back to light source

        if (render_world.enable_shadows) //Only need to check for light blocking if shadows are enabled
        {
            obj = render_world.Closest_Phong_Sphere_Intersection(Ray(intersection_point, shadeRay.normalized())); //Find intersection along Shade Ray
            if (shadeRay.magnitude() > obj.second.dist && obj.second.dist != -1) //If the light is further than any object, it is blocked
            {
                    //Removed pixel trace message, but did not change logic as it is functional
            }
            else //Any other case signal it is lit by that source
            {
                lit = true;
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
            diffuse = (render_world.lights[i]->Emitted_Light(shadeRay) * intensity) * color_diffuse->Get_Color(hit.uv);
            reflect = (2 * dot(normal, shadeRay.normalized()) * normal - (shadeRay.normalized()));
            phi = dot(-ray.direction, reflect);
            if (phi < 0)
            {
                phi = 0;
            }
            specular = (render_world.lights[i]->Emitted_Light(shadeRay) * pow(phi, specular_power)) * color_specular->Get_Color(hit.uv);
            lit = false;
            //printf("diffuse: (%f, %f, %f)\n",diffuse[0],diffuse[1],diffuse[2]);
            //printf("specular: (%f, %f, %f)\n",specular[0],specular[1],specular[2]);
            color += diffuse + specular;
        }
    }
    return color;
}

/*Shade Surface for Phong Shaded Planes*/
__host__ __device__
vec3 Phong_Shader::
    Shade_Phong_Plane_Surface(const Render_World &render_world, const Ray &ray, const Hit &hit,
                  const vec3 &intersection_point, const vec3 &normal, int recursion_depth) const
{
    vec3 color;                                                              // variable for final color
    vec3 ambient, diffuse, specular;                                         // variables for total light components
    double intensity, phi;                                                   // variables for the intensity of a light component
    ambient = color_ambient->Get_Color(hit.uv) * render_world.ambient_intensity; // Capture ambient light
    if (render_world.ambient_color)
    {
        ambient *= render_world.ambient_color->Get_Color(hit.uv);
    }
    color = ambient;
    std::pair<Phong_Shaded_Plane, Hit> obj; // object along ray from intersection back to light source
    vec3 shadeRay;                     // Shade ray for diffuse calculations.
    vec3 reflect;                      // Reflection Ray for specular
    bool lit = false;                  // flag for whether a light source is illuminating the point under inspection
    for (int i = 0; i<render_world.num_lights;i++) // check all lights in world
    {
        shadeRay = render_world.lights[i]->position - intersection_point; // vector from intersection point back to light source

        if (render_world.enable_shadows) //Only need to check for light blocking if shadows are enabled
        {
            obj = render_world.Closest_Phong_Plane_Intersection(Ray(intersection_point, shadeRay.normalized())); //Find intersection along Shade Ray
            if (shadeRay.magnitude() > obj.second.dist && obj.second.dist != -1) //If the light is further than any object, it is blocked
            {
                    //Removed pixel trace message, but did not change logic as it is functional
            }
            else //Any other case signal it is lit by that source
            {
                lit = true;
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
            diffuse = (render_world.lights[i]->Emitted_Light(shadeRay) * intensity) * color_diffuse->Get_Color(hit.uv);
            reflect = (2 * dot(normal, shadeRay.normalized()) * normal - (shadeRay.normalized()));
            phi = dot(-ray.direction, reflect);
            if (phi < 0)
            {
                phi = 0;
            }
            specular = (render_world.lights[i]->Emitted_Light(shadeRay) * pow(phi, specular_power)) * color_specular->Get_Color(hit.uv);
            lit = false;
            color += diffuse + specular;
        }
    }
    return color;
}
