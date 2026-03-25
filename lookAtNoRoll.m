
function [R, quat, rotation_vector] = lookAtNoRoll(currentPos, target)


%% Compute the direction vector (Z-axis of the tool orientation)
dir = [target.x - currentPos.x, ...
       target.y - currentPos.y, ...
       target.z - currentPos.z];
z_axis = dir / norm(dir);  % Normalize

% Choose a fixed world up vector
world_up = [0, 0, 1];

% Project world_up onto the plane perpendicular to z_axis to remove roll
up_proj = world_up - dot(world_up, z_axis) * z_axis;

% If the projection is too small (i.e., z_axis is aligned with world_up), pick another vector
if norm(up_proj) < 1e-3
    world_up = [0, 1, 0];
    up_proj = world_up - dot(world_up, z_axis) * z_axis;
end

x_axis = up_proj / norm(up_proj);
y_axis = cross(z_axis, x_axis);

% Build rotation matrix with zero roll
R = [x_axis', y_axis', z_axis'];


%% Rotation matrix and quaternion conversion
% R = [x_axis', y_axis', z_axis'];
quat = rotm2quat(R);  % Format: [w, x, y, z]


%% Convert quaternion to axis-angle, then to rotation vector
axang = rotm2axang(quat2rotm(quat));
rotation_vector = axang(1:3) * axang(4);


end