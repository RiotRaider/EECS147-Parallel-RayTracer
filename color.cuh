#ifndef __COLOR_H__
#define __COLOR_H__

struct Hit;
class Parse; //compiler error fix???? seen in sphere.cuh

class Color : public Managed
{
private:
    vec3 color;

public:
    std::string name;

    Color(const Parse* parse, std::istream& in)
    {
	in>>name>>color;
    }
    virtual ~Color()=default;
   
    __host__ __device__
    vec3 Get_Color(const vec2& uv) const {
	return color;
    }
    
    static constexpr const char* parse_name = "color";

};



#endif
