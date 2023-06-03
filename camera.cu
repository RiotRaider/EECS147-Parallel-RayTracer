#include "camera.cuh"

Camera::Camera()
{
    cudaMallocManaged(&colors,sizeof(Pixel)*(number_pixels[0]*number_pixels[1]));
}
Camera::Camera(const Camera& c)
    :position(c.position), film_position(c.film_position), look_vector(c.look_vector), vertical_vector(c.vertical_vector),
     horizontal_vector(c.horizontal_vector), min(c.min), max(c.max), image_size(c.image_size), pixel_size(c.pixel_size),
     number_pixels(c.number_pixels)
{
    int size = number_pixels[0]*number_pixels[1];
    cudaMallocManaged(&colors,sizeof(Pixel)*(number_pixels[0]*number_pixels[1]));
    //printf("Attempting memcopy\n");
    memcpy(colors, c.colors, sizeof(Pixel)*size);
    //printf("finished memcopy\n");
}
Camera::~Camera()
{
    //printf("Attempting delete\n");
    cudaFree(colors);
    //printf("finished delete\n");
}

void Camera::Position_And_Aim_Camera(const vec3& position_input,
    const vec3& look_at_point,const vec3& pseudo_up_vector)
{
    position=position_input;
    look_vector=(look_at_point-position).normalized();
    horizontal_vector=cross(look_vector,pseudo_up_vector).normalized();
    vertical_vector=cross(horizontal_vector,look_vector).normalized();
}

void Camera::Focus_Camera(double focal_distance,double aspect_ratio,
    double field_of_view)
{
    film_position=position+look_vector*focal_distance;
    double width=2.0*focal_distance*tan(.5*field_of_view);
    double height=width/aspect_ratio;
    image_size=vec2(width,height);
}
__host__ __device__
void Camera::Set_Resolution(const ivec2& number_pixels_input)
{
    number_pixels=number_pixels_input;
    cudaFree(colors);
    cudaMallocManaged(&colors,sizeof(Pixel)*(number_pixels[0]*number_pixels[1]));
    min=-0.5*image_size;
    max=0.5*image_size;
    pixel_size = image_size/vec2(number_pixels);
}

// Find the world position of the input pixel
vec3 Camera::World_Position(const ivec2& pixel_index)
{
    vec3 result;
    vec2 center = this->Cell_Center(pixel_index);
    result = this->film_position +(center[0]*this->horizontal_vector)+(center[1]*this->vertical_vector);
 
    return result;
}
