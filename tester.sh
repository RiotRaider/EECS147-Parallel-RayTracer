#!/bin/bash

echo -e "\nTest Flat Shader-1"
./ray_tracer -i tests-cpu/f-00.txt -s test-solutions/f-00.png
./ray_tracer -i tests-gpu/f-00g.txt -s test-solutions/f-00.png

echo -e "\nTest Flat Shader-2"
./ray_tracer -i tests-cpu/f-01.txt -s test-solutions/f-01.png
./ray_tracer -i tests-gpu/f-01g.txt -s test-solutions/f-01.png

echo -e "\nTest Flat Shader-3"
./ray_tracer -i tests-cpu/f-02.txt -s test-solutions/f-02.png
./ray_tracer -i tests-gpu/f-02g.txt -s test-solutions/f-02.png

echo -e "\nTest Flat Shader-4"
./ray_tracer -i tests-cpu/f-03.txt -s test-solutions/f-03.png
./ray_tracer -i tests-gpu/f-03g.txt -s test-solutions/f-03.png

echo -e "\nTest Flat Shader-5"
./ray_tracer -i tests-cpu/f-04.txt -s test-solutions/f-04.png
./ray_tracer -i tests-gpu/f-04g.txt -s test-solutions/f-04.png

echo -e "\nTest Phong Shader-1"
./ray_tracer -i tests-cpu/p-00.txt -s test-solutions/p-00.png
./ray_tracer -i tests-gpu/p-00g.txt -s test-solutions/p-00.png

echo -e "\nTest Phong Shader-2"
./ray_tracer -i tests-cpu/p-01.txt -s test-solutions/p-01.png
./ray_tracer -i tests-gpu/p-01g.txt -s test-solutions/p-01.png

echo -e "\nTest Phong Shader-3"
./ray_tracer -i tests-cpu/p-02.txt -s test-solutions/p-02.png
./ray_tracer -i tests-gpu/p-02g.txt -s test-solutions/p-02.png

echo -e "\nTest Phong Shader-4"
./ray_tracer -i tests-cpu/p-05.txt -s test-solutions/p-05.png
./ray_tracer -i tests-gpu/p-05g.txt -s test-solutions/p-05.png

echo -e "\nTest Phong Shader-5"
./ray_tracer -i tests-cpu/p-06.txt -s test-solutions/p-06.png
./ray_tracer -i tests-gpu/p-06g.txt -s test-solutions/p-06.png

echo -e "\nTest Phong Shader-6"
./ray_tracer -i tests-cpu/p-07.txt -s test-solutions/p-07.png
./ray_tracer -i tests-gpu/p-07g.txt -s test-solutions/p-07.png

echo -e "\nTest Phong Shader-7"
./ray_tracer -i tests-cpu/p-08.txt -s test-solutions/p-08.png
./ray_tracer -i tests-gpu/p-08g.txt -s test-solutions/p-08.png

echo -e "\nTest Phong Shader-8"
./ray_tracer -i tests-cpu/p-09.txt -s test-solutions/p-09.png
./ray_tracer -i tests-gpu/p-09g.txt -s test-solutions/p-09.png

echo -e "\nTest Phong Shader-9"
./ray_tracer -i tests-cpu/p-10.txt -s test-solutions/p-10.png
./ray_tracer -i tests-gpu/p-10g.txt -s test-solutions/p-10.png

echo -e "\nTest Phong Shader-10"
./ray_tracer -i tests-cpu/p-12.txt -s test-solutions/p-12.png
./ray_tracer -i tests-gpu/p-12g.txt -s test-solutions/p-12.png

echo -e "\nTest Phong Shader-11"
./ray_tracer -i tests-cpu/p-13.txt -s test-solutions/p-13.png
./ray_tracer -i tests-gpu/p-13g.txt -s test-solutions/p-13.png

echo -e "\nTest Phong Shader-12"
./ray_tracer -i tests-cpu/p-14.txt -s test-solutions/p-14.png
./ray_tracer -i tests-gpu/p-14g.txt -s test-solutions/p-14.png

echo -e "\nTest Phong Shader-13"
./ray_tracer -i tests-cpu/p-15.txt -s test-solutions/p-15.png
./ray_tracer -i tests-gpu/p-15g.txt -s test-solutions/p-15.png

echo -e "\nTest Phong Shader-14"
./ray_tracer -i tests-cpu/p-16.txt -s test-solutions/p-16.png
./ray_tracer -i tests-gpu/p-16g.txt -s test-solutions/p-16.png

echo -e "\nTest Phong Shader-15"
./ray_tracer -i tests-cpu/p-17.txt -s test-solutions/p-17.png
./ray_tracer -i tests-gpu/p-17g.txt -s test-solutions/p-17.png

echo -e "\nTest Phong Shader-16"
./ray_tracer -i tests-cpu/p-18.txt -s test-solutions/p-18.png
./ray_tracer -i tests-gpu/p-18g.txt -s test-solutions/p-18.png

echo -e "\nTest Phong Shader-17"
./ray_tracer -i tests-cpu/p-19.txt -s test-solutions/p-19.png
./ray_tracer -i tests-gpu/p-19g.txt -s test-solutions/p-19.png

echo -e "\nTest Phong Shader-18"
./ray_tracer -i tests-cpu/p-20.txt -s test-solutions/p-20.png
./ray_tracer -i tests-gpu/p-20g.txt -s test-solutions/p-20.png

