#ifndef __PARSE_H__
#define __PARSE_H__

#include <map>

#include "light.cuh"
#include "object.cuh"
#include "shader.cuh"
#include "color.cuh"

class Render_World;

class Parse
{
    // Lookup shaders/objects/colors by name.
    std::map<std::string,const Sphere*> spheres;
    std::map<std::string,const Plane*> planes;
    std::map<std::string,const Flat_Shader*> flat_shaders;
    std::map<std::string,const Phong_Shader*> phong_shaders;
    std::map<std::string,const Color*> colors;

    // These are factories.  Given the class's parse name, construct an object
    // of the correct type.  The object's constructor will parse from the input
    // stream to initialize itself.  Note that the key is a string and the data
    // is a function pointer.  This function is called to construct the object.
    // These are populated by the registration routines below.
    std::map<std::string,Sphere*(*)(const Parse* parse,std::istream& in)> parse_spheres;
    std::map<std::string,Plane*(*)(const Parse* parse,std::istream& in)> parse_planes;
    std::map<std::string,Flat_Shader*(*)(const Parse* parse,std::istream& in)> parse_flat_shaders;
    std::map<std::string,Phong_Shader*(*)(const Parse* parse,std::istream& in)> parse_phong_shaders;
    std::map<std::string,Light*(*)(const Parse* parse,std::istream& in)> parse_lights;
    std::map<std::string,Color*(*)(const Parse* parse,std::istream& in)> parse_colors;

    // image dimensions
    int width=-1;
    int height=-1;
public:
    void Parse_Input(Render_World& render_world, std::istream& in);

    // Public access to the stored objects.
    const Color* Get_Color(std::istream& in) const;
    const Shader* Get_Sphere(std::istream& in) const;
    const Object* Get_Plane(std::istream& in) const;
    const Shader* Get_Flat_Shader(std::istream& in) const;
    const Object* Get_Phong_Shader(std::istream& in) const;

    // These are the routines that populate the factories above.  They are
    // called in registration.cpp.  You will need to modify that file as you get
    // additional source files in later homework assignments.  There is a bit of
    // "magic" happening in the routines below.  Interesting things to note:

    // 1. The routine is called with a template argument that indicates the type
    //    being registered (such as Register_Object<Sphere>() or
    //    Register_Shader<Flat_Shader>()).

    // 2. The routine then generates a function (a lambda function) that
    //    allocate an object of the appropriate type, passes its arguments to
    //    the constructor (which actually does the parsing), and then returns
    //    the pointer.  Caller owns the allocated memory and must free it.

    // 3. The generated function is then inserted into the lookup table.  The
    //    parse key is Type::parse_name.  This is a fixed string defined inside
    //    each derived class that can be parsed.  In particular, classes like
    //    Sphere or Flat_Shader know how to parse themselves.  Although this
    //    makes parsing more complicated, it lets us add new files, types, and
    //    the ability to parse them.

    // 4. One might ask whether it is possible for registration of Sphere to
    //    occur in sphere.cpp so that the file could be added without modifying
    //    any existing files.  In C and C++, this is not reliably possible.
    //    Without some other file referring to something (anything at all!) in
    //    sphere.cpp, the linker is free to discard it in its entirety.  The
    //    file registration.cpp (and only that file!) refers to something in
    //    those files (their constructors!).

    Sphere void Register_Sphere()
    {
        parse_sphere[Type::parse_name]=
            [](const Parse* parse,std::istream& in) -> Sphere*
            {return new Object(parse,in);};
    }
    
    void Register_Plane()
    {
        parse_plane[Type::parse_name]=
            [](const Parse* parse,std::istream& in) -> Plane*
            {return new Plane(parse,in);};
    }

    void Register_Flat_Shader()
    {
        parse_flat_shaders[Type::parse_name]=
            [](const Parse* parse,std::istream& in) -> Flat_Shader*
            {return new Flat_Shader(parse,in);};
    }

    void Register_Flat_Shader()
    {
        parse_phong_shaders[Type::parse_name]=
            [](const Parse* parse,std::istream& in) -> Phong_Shader*
            {return new Phong_Shader(parse,in);};
    }

    void Register_Light()
    {
        parse_lights[Type::parse_name]=
            [](const Parse* parse,std::istream& in) -> Light*
            {return new Light(parse,in);};
    }

    void Register_Color()
    {
        parse_colors[Type::parse_name]=
            [](const Parse* parse,std::istream& in) -> Color*
            {return new Color(parse,in);};
    }
};

#endif
