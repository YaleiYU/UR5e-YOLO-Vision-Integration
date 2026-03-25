clc;
clear;
close all;


%% Define position list
% positions.x = [0.2; -0.2; 0.2];
% positions.y = [-0.3; -0.3; -0.3];
% positions.z = [0.5; 0.5; 0.5];
positions.x = [0.2; -0.2; 0.3];
positions.y = [-0.3; -0.4; 0.4];
positions.z = [0.6; 0.6; 0.6];

offsetY = -0.3;
numPoints = length(positions.x);

figure(1);
hold on;
grid on;
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
view(3);
title('UR5e Pose Visualization');


%% Iterate through each position
for i = 1:numPoints
    %% Extract current position
    currentPos = struct( ...
        'x', positions.x(i), ...
        'y', positions.y(i), ...
        'z', positions.z(i) ...
        );

    % Assign position to publisher message (if needed in ROS context)
    posiPubmsg.position = currentPos;


    %% Define target point to look at
    target = struct('x', 0, 'y', offsetY, 'z', 0);


    %%
    [R, quat, rotation_vector] = lookAtNoRoll(currentPos, target);


    %% Send motion command via ROS2 (URScript)
    command = sprintf('ros2 topic pub --once /urscript_interface/script_command std_msgs/msg/String ''{data: "def my_prog():\\nset_digital_out(1, True)\\nmovej(p[%f, %f, %f, %f, %f, %f], a=1.2, v=0.25, r=0)\\ntextmsg(\\\"motion finished\\\")\\nend"}''', currentPos.x, currentPos.y, currentPos.z, rotation_vector(1,1), rotation_vector(1,2), rotation_vector(1,3));

    [status, cmdout] = system(command);


    %% Visualize pose
    plotTransforms([currentPos.x, currentPos.y, currentPos.z], quat, 'FrameSize', 0.2);
    plot3(target.x, target.y, target.z, 'rx', 'MarkerSize', 12, 'LineWidth', 2);


    %% Pause for user input before continuing
    input('Press Enter to move to the next position...', 's');


end
