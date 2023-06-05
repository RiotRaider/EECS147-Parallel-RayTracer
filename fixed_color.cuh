#ifndef __FIXED_COLOR_H__
#define __FIXED_COLOR_H__

#include "color.cuh"

class Fixed_Color : public Color
{
/*    //vec3 color;

public:
    virtual ~Fixed_Color()=default;

    Fixed_Color(const Parse* parse,std::istream& in)
    {
        in>>name>>color;
    }
    
    __host__ __device__
    virtual vec3 Get_Color(const vec2& uv) const override
    {
        return color;
    }

    static constexpr const char* parse_name = "color";*/
};


#endif
