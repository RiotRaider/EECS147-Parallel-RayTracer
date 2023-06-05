#ifndef __PHONG_SHADER_H__
#define __PHONG_SHADER_H__

#include "shader.cuh"

class Phong_Shader : public Shader
{
private:
    void _realloc(){

	if (color_ambient != 0) cudaFree((void*)color_ambient);
    	cudaMallocManaged(&color_ambient, sizeof(Color));
//	if (cuda_ret != cudaSuccess) FATAL("could not allocate for color_amb");

	if (color_diffuse != 0) cudaFree((void*)color_diffuse);
	cudaMallocManaged(&color_diffuse, sizeof(Color));
//	if (cuda_ret != cudaSuccess) FATAL("Could not allocate for color_diff");

	if (color_specular != 0) cudaFree((void*)color_specular);
	cudaMallocManaged(&color_specular, sizeof(Color));
//	if (cuda_ret != cudaSuccess) FATAL("Could not allocate for color_spec");

    }
public:
    const Color* color_ambient = nullptr;
    const Color* color_diffuse = nullptr;
    const Color* color_specular = nullptr;
    double specular_power = 0;

    Phong_Shader(const Parse* parse,std::istream& in);
    Phong_Shader(const Phong_Shader& s);
    virtual ~Phong_Shader() {
	cudaFree((void*)color_ambient);
	cudaFree((void*)color_diffuse);
	cudaFree((void*)color_specular);
	};

    __host__ __device__
    virtual vec3 Shade_Surface(const Render_World& render_world,const Ray& ray,
        const Hit& hit,const vec3& intersection_point,const vec3& normal,
        int recursion_depth) const override;

    static constexpr const char* parse_name = "phong_shader";
};
#endif
