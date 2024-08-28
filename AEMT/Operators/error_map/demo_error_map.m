clc
clear
close all

load("test_data.mat");
error_map(source_pc, gt_tform, est_tform_cell);