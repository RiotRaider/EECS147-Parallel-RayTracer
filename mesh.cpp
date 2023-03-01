#include "mesh.h"
#include <fstream>
#include <limits>
#include <string>
#include <algorithm>

// Consider a triangle to intersect a ray if the ray intersects the plane of the
// triangle with barycentric weights in [-weight_tolerance, 1+weight_tolerance]
static const double weight_tolerance = 1e-4;

Mesh::Mesh(const Parse *parse, std::istream &in)
{
    std::string file;
    in >> name >> file;
    Read_Obj(file.c_str());
}

// Read in a mesh from an obj file.  Populates the bounding box and registers
// one part per triangle (by setting number_parts).
void Mesh::Read_Obj(const char *file)
{
    std::ifstream fin(file);
    if (!fin)
    {
        exit(EXIT_FAILURE);
    }
    std::string line;
    ivec3 e, t;
    vec3 v;
    vec2 u;
    while (fin)
    {
        getline(fin, line);

        if (sscanf(line.c_str(), "v %lg %lg %lg", &v[0], &v[1], &v[2]) == 3)
        {
            vertices.push_back(v);
        }

        if (sscanf(line.c_str(), "f %d %d %d", &e[0], &e[1], &e[2]) == 3)
        {
            for (int i = 0; i < 3; i++)
                e[i]--;
            triangles.push_back(e);
        }

        if (sscanf(line.c_str(), "vt %lg %lg", &u[0], &u[1]) == 2)
        {
            uvs.push_back(u);
        }

        if (sscanf(line.c_str(), "f %d/%d %d/%d %d/%d", &e[0], &t[0], &e[1], &t[1], &e[2], &t[2]) == 6)
        {
            for (int i = 0; i < 3; i++)
                e[i]--;
            triangles.push_back(e);
            for (int i = 0; i < 3; i++)
                t[i]--;
            triangle_texture_index.push_back(t);
        }
    }
    num_parts = triangles.size();
}

// Check for an intersection against the ray.  See the base class for details.
Hit Mesh::Intersection(const Ray &ray, int part) const
{
    double min_t = std::numeric_limits<double>::max();
    Hit hit, test;
    if (part < 0)
    {
        for (size_t i = 0; i < triangles.size(); i++)
        {
            test = Intersect_Triangle(ray, i);
            if (test.dist >= small_t && test.dist < min_t)
            {
                min_t = test.dist;
                hit = test;
            }
        }
    }
    else
    {
        hit = Intersect_Triangle(ray, part);
    }

    return hit;
}

// Compute the normal direction for the triangle with index part.
vec3 Mesh::Normal(const Ray &ray, const Hit &hit) const
{
    assert(hit.triangle >= 0);
    // Find Vertices of triangle
    vec3 A, B, C, n;
    A = vertices[triangles[hit.triangle][0]];
    B = vertices[triangles[hit.triangle][1]];
    C = vertices[triangles[hit.triangle][2]];
    n = cross((B - A), (C - A)).normalized();

    return n;
}

// This is a helper routine whose purpose is to simplify the implementation
// of the Intersection routine.  It should test for an intersection between
// the ray and the triangle with index tri.  If an intersection exists,
// record the distance and return true.  Otherwise, return false.
// This intersection should be computed by determining the intersection of
// the ray and the plane of the triangle.  From this, determine (1) where
// along the ray the intersection point occurs (dist) and (2) the barycentric
// coordinates within the triangle where the intersection occurs.  The
// triangle intersects the ray if dist>small_t and the barycentric weights are
// larger than -weight_tolerance.  The use of small_t avoid the self-shadowing
// bug, and the use of weight_tolerance prevents rays from passing in between
// two triangles.
Hit Mesh::Intersect_Triangle(const Ray &ray, int tri) const
{
    Hit hit;
    hit.triangle = tri;
    ivec3 vert = triangles[tri];
    vec3 AB = vertices[vert[1]] - vertices[vert[0]]; // side AB
    vec3 AC = vertices[vert[2]] - vertices[vert[0]]; // side AC
    vec3 eA = ray.endpoint - vertices[vert[0]];      // Vector from Camera to Vertex A
    vec3 n = Normal(ray, hit);

    if (dot(ray.direction, n) == 0)
    { // Check Ray is not parallel to triangle
        return hit;
    }

    double t = -dot(eA, n) / dot(ray.direction, n);
    if (t < small_t)
    { // Check that distance is sufficiently far to count
        return hit;
    }

    // Find Barycentric Weights
    vec3 P = ray.endpoint + (t * ray.direction);
    // Area of Triangle ABC
    double ABC = 0.5 * cross(AB, AC).magnitude();
    // alpha
    double a = 0.5 * dot(cross(vertices[vert[1]] - P, vertices[vert[2]] - P), n) / ABC;
    // beta
    double b = 0.5 * dot(cross(P - vertices[vert[0]], AC), n) / ABC;
    // gamma
    double g = 1 - a - b;
    if (a > -weight_tolerance && b > -weight_tolerance && g > -weight_tolerance)
    {
        // PIXEL TRACE
        Debug_Scope scope;
        Pixel_Print("mesh ", this->name, " triangle ", tri, " intersected; weights: (", a, b, g, "); dist ", t);
        // END PIXEL TRACE
        hit.dist = t;
        if(!triangle_texture_index.empty()){
        ivec3 uv_vert = triangle_texture_index[tri];
        vec2 uvA=uvs[uv_vert[0]];
        vec2 uvB=uvs[uv_vert[1]]; 
        vec2 uvC=uvs[uv_vert[2]];
        hit.uv =a*uvA+b*uvB+g*uvC;}
    }
    return hit;
}

std::pair<Box, bool> Mesh::Bounding_Box(int part) const
{
    if (part < 0)
    {
        Box box;
        box.Make_Empty();
        for (const auto &v : vertices)
            box.Include_Point(v);
        return {box, false};
    }

    ivec3 e = triangles[part];
    vec3 A = vertices[e[0]];
    Box b = {A, A};
    b.Include_Point(vertices[e[1]]);
    b.Include_Point(vertices[e[2]]);
    return {b, false};
}
