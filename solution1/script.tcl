############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project FindContours
add_files FindContours/top_level.cpp
add_files FindContours/top_level.hpp
add_files -tb FindContours/foto.jpg
add_files -tb FindContours/tb.cpp
open_solution "solution1"
set_part {xc7vx485t-ffg1157-1}
create_clock -period 10 -name default
#source "./FindContours/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
