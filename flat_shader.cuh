#ifndef __FLAT_SHADER_H__
#define __FLAT_SHADER_H__

#include "shader.cuh"
#include "color.cuh"

class Flat_Shader : public Shader
{
private:
    void _realloc(){
        if (color != 0) { 
            cudaFree((void*)color); 
        }
	
        cudaMallocManaged(&color, sizeof(Color));
    }

public:
    const Color* color = nullptr;

    Flat_Shader(const Parse* parse,std::istream& in);
    Flat_Shader(const Flat_Shader& s);
    
    ~Flat_Shader() { cudaFree((void*)color); } ;
    
    __host__ __device__
    vec3 Shade_Flat_Sphere_Surface(const Render_World& render_world,const Ray& ray,
        const Hit& hit,const vec3& intersection_point,const vec3& normal,
        int recursion_depth) const;

     __host__ __device__
    vec3 Shade_Flat_Plane_Surface(const Render_World& render_world,const Ray& ray,
        const Hit& hit,const vec3& intersection_point,const vec3& normal, 
	int recursion_depth) const;
    
    static constexpr const char* parse_name = "flat_shader";
};
#endif
