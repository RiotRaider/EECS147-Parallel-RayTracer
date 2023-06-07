#include "parse.h"
#include "render_world.cuh"
#include <map>
#include <sstream>
#include <iostream>
#include <string>

//modify parsing so it parses and adds the objects to the correct array in render
void Parse::Parse_Input(Render_World& render_world, std::istream& in)
{
    std::string token,s0,s1,line;
    vec3 u,v,w;
    double f0;

    while(getline(in,line))
    {
        std::stringstream ss(line);
        ss>>token;
	//if a comment, do nothign 
        if(token[0]=='#')
        {
        }
	/*if an object --> sphere and plane*/
        else if(auto it=parse_spheres.find(token); it!=parse_spheres.end()) 
        {
            auto o=it->second(this,ss);
            spheres[o->name]=o;
            render_world.all_spheres[render_world.num_spheres++]=o;
        }
	else if(auto it=parse_planes.find(token); it!=parse_planes.end()) 
	{
	    auto o=it->second(this,ss);
            planes[o->name]=o;
            render_world.all_planes[render_world.num_planes++]=o;

	}
	/*if a shader --> flat and phong*/
        else if(auto it=parse_flat_shaders.find(token); it!=parse_flat_shaders.end())
        {
            auto s=it->second(this,ss);
            flat_shaders[s->name]=s;
            render_world.all_flat_shaders[render_world.num_flat_shaders++]=s;
        }
	else if(auto it=parse_phong_shaders.find(token); it!=parse_phong_shaders.end())
        {
            auto s=it->second(this,ss);
            phong_shaders[s->name]=s;
            render_world.all_phong_shaders[render_world.num_phong_shaders++]=s;
	} 
	/*if a light*/
        else if(auto it=parse_lights.find(token); it!=parse_lights.end())
        {
            render_world.lights[render_world.num_lights++]=it->second(this,ss);
        }
	/*if a color*/
        else if(auto it=parse_colors.find(token); it!=parse_colors.end())
        {
            auto c=it->second(this,ss);
            colors[c->name]=c;
            render_world.all_colors[render_world.num_colors++]=c;
        }
	/*if a shaded object*/
	else if (token == "flat_shaded_sphere"){
	    auto o=Get_Sphere(ss);
	    auto s = Get_Flat_Shader(ss);
	    Flat_Shaded_Sphere fs;
	    fs.sphere = o;
	    fs.flat_shader = s;
            render_world.flat_shaded_spheres[render_world.num_flat_shaded_spheres++]=fs;
	    //printf("Flat Shaded Spheres: %d", render_world.num_flat_shaded_spheres);
	}
	else if (token == "phong_shaded_sphere"){
	    auto o = Get_Sphere(ss);
	    auto s = Get_Phong_Shader(ss);
	    Phong_Shaded_Sphere ps;
	    ps.sphere = o;
	    ps.phong_shader = s;
            render_world.phong_shaded_spheres[render_world.num_phong_shaded_spheres++]=ps;
	}
	else if (token == "flat_shaded_plane"){
	    auto o=Get_Plane(ss);
	    auto s = Get_Flat_Shader(ss);
	    Flat_Shaded_Plane fp;
	    fp.plane = o;
	    fp.flat_shader = s;
            render_world.flat_shaded_planes[render_world.num_flat_shaded_planes++]=fp;
	}
	else if (token == "phong_shaded_plane"){
	    auto o=Get_Plane(ss);
	    auto s = Get_Phong_Shader(ss);
	    Phong_Shaded_Plane pp;
	    pp.plane = o;
	    pp.phong_shader = s;
            render_world.phong_shaded_planes[render_world.num_phong_shaded_planes++]=pp;
	}
	/*if background shader*/
        else if(token=="background_shader")
        {
            render_world.background_shader=Get_Flat_Shader(ss);
        }
        else if(token=="ambient_light")
        {
            render_world.ambient_color=Get_Color(ss);
            ss>>render_world.ambient_intensity;
        }
        else if(token=="size")
        {
            ss>>width>>height;
        }
        else if(token=="camera")
        {
            ss>>u>>v>>w>>f0;
            render_world.camera->Position_And_Aim_Camera(u,v,w);
            render_world.camera->Focus_Camera(1,(double)width/height,f0*(pi/180));
        }
        else if(token=="enable_shadows")
        {
            ss>>render_world.enable_shadows;
        }
        else if(token=="recursion_depth_limit")
        {
            ss>>render_world.recursion_depth_limit;
        }
        else if(token=="gpu") {
            ss>>render_world.gpu_on;
        }
        else
        {
            std::cout<<"Failed to parse at: "<<token<<std::endl;
            exit(EXIT_FAILURE);
        }
        assert(ss);
    }
    render_world.camera->Set_Resolution(ivec2(width,height));
}

const Flat_Shader* Parse::Get_Flat_Shader(std::istream& in) const
{
    std::string token;
    in>>token;
        
    auto it=flat_shaders.find(token);
    assert(it!=flat_shaders.end());
    return it->second;
}

const Phong_Shader* Parse::Get_Phong_Shader(std::istream& in) const
{
    std::string token;
    in>>token;
        
    auto it=phong_shaders.find(token);
    assert(it!=phong_shaders.end());
    return it->second;
}

const Sphere* Parse::Get_Sphere(std::istream& in) const
{
    std::string token;
    in>>token;

    auto it=spheres.find(token);
    assert(it!=spheres.end());
    return it->second;
}

const Plane* Parse::Get_Plane(std::istream& in) const
{
    std::string token;
    in>>token;

    auto it=planes.find(token);
    assert(it!=planes.end());
    return it->second;
}

const Color* Parse::Get_Color(std::istream& in) const
{
    std::string token;
    in>>token;

    auto it=colors.find(token);
    assert(it!=colors.end());
    return it->second;
}
