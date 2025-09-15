%% Inputs
% Building the rainbow trap
J = 6; % total no. of layers

x_id = [1, 1, 1, 1, 1, 1]; % xid of the start of each layer
y_id = [1, 1, 1, 1, 1, 1]; % yid of the start of each layer

% Building the homogenous layers
omega = 1;    % frequency of the incident plane wave
th    = pi/6; % angle of incidence
rh    = 1;    % density of the homogenous layers
Gh    = 1;    % shear modulus of the homogeonus layers
ch    = sqrt(Gh / rh); % shear wave velocity of the homogenous layers
ky   = (omega / ch) * sin(th); % wavenumber along direction 

%% Solution
[ul, xi1_l, urt, xi1_rt, ur, xi1_r, xi2] = BVP1_solver(J, Gh, ch, omega, ky, x_id, y_id);

%% Plots

xi2_id = 51; % y-coordinate id to plot displacement field

u   = [ul(xi2_id, :), urt(xi2_id, :), ur(xi2_id, :)]; % Analytical u
xi1 = [xi1_l, xi1_rt, xi1_r]; % u(xi1)
plot(xi1   , real(u)   , 'k', 'LineWidth', 2)
hold on
plot(xi1   , imag(u)   , 'b', 'LineWidth', 2)