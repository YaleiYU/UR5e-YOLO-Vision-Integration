        

%%
clc;
clear 
close all;

%% pose 1
pos.x = 0.4816; 
pos.y = -0.1774;
pos.z = 0.5; 

rotation_vector(1,1) = 2.816;
rotation_vector(1,2) = -1.420;
rotation_vector(1,3) = -0.092;

%% pose 2
% pos.x = 0.3792; 
% pos.y = -0.0416;
% pos.z = 0.5; 
% 
% rotation_vector(1,1) = 1.507;
% rotation_vector(1,2) = 2.540;
% rotation_vector(1,3) = 0.005;


%% send commands 
command = sprintf('ros2 topic pub --once /urscript_interface/script_command std_msgs/msg/String ''{data: "def my_prog():\\nset_digital_out(1, True)\\nmovej(p[%f, %f, %f, %f, %f, %f], a=1.2, v=0.25, r=0)\\ntextmsg(\\\"motion finished\\\")\\nend"}''', pos.x, pos.y, pos.z, rotation_vector(1,1), rotation_vector(1,2), rotation_vector(1,3));
[status, cmdout] = system(command);