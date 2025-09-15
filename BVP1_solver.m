function [ul, xi1_l, urt, xi1_rt, ur, xi1_r, xi2] = BVP1_solver(J, Gh, ch, omega, ky, x_id, y_id)
%%% solution of the rainbow trap BVP.

% input:
% J = no. of layers
% Gh = shear modulus of the homogenous medium
% ch = shear wave speed of the homogenous medium
% omega = excitation frequency
% ky = wavenumber along the interface
% x_id = index of the x-shift of each layer
% y_id = index of the y-shift of each layer

% output:
% ul  = displacement field in the left homogenous medium
% urt = displacement field within the rainbow trap
% ur  = displacement field in the right homogenous medium
% xi1_l, xi1_rt, xi1_r = x coordinates of the corresponding fields
% xi2 = y coordinate of the correposponding fields

xj = [0, zeros(1, J)]; % cumulative lengths of each layers
% load the python file and store the variables
for j = 1 : J
    file_m = strcat("Matlab_data/uc_", string(j-1), ".mat");
    load(file_m, "kappa", "phi", "psi", "G", "x", "y")
    temp1(:, j)       = transpose(kappa);
    temp              = [phi(:, x_id(j) : end, :) , phi(:, 2 : x_id(j), : )];
    temp2(:, :, :, j) = [temp(y_id(j) : end, :, :); temp(2 : y_id(j), :, :)];
    temp              = [psi(:, x_id(j) : end, :) , psi(:, 2 : x_id(j), : )];
    temp3(:, :, :, j) = [temp(y_id(j) : end, :, :); temp(2 : y_id(j), :, :)];
    temp              = [G(:, x_id(j) : end) , psi(:, 2 : x_id(j))];
    temp4(:, :, j)    = [temp(y_id(j) : end, :); temp(2 : y_id(j), :)];
    xj(j+1) = xj(j) + x(end);
end
[inter_x, inter_y] = size(x);
temp2 = temp2(1:inter_x, 1:inter_y, :, :);
temp3 = temp3(1:inter_x, 1:inter_y, :, :);
temp4 = temp4(1:inter_x, 1:inter_y, :);
kappa = temp1; % eigenvalues of each layers
phi   = temp2; % eigefunctions of each layers
psi   = temp3; % gradient of the eigenfunctions
G     = temp4; % shear modulus

h = y(end); % height of the unit-cells (needs to be same)

N = length(kappa)/2; % 2*N   is the total no. of Bloch modes
M = (N-1)/2;         % 2*M+1 is the total no. of Fourier coefficients

% Diffraction grating
temp1 = zeros(1, 2*M+1);
temp2 = zeros(1, 2*M+1);
kkh = omega / ch;
kix = sqrt(kkh^2 - ky^2);
for m = -M : M
    temp1(m+M+1) = ky + (2*pi / h) * m;
    temp2(m+M+1) = -sqrt(kkh^2 - temp1(m+M+1)^2);
end
kry = temp1;
krx = temp2;

% Building Lambda and nE (see paper https://doi.org/10.1016/j.jmps.2024.105746 for details on notations)
Lam_jl   = zeros(4*M+2, N, J);
Lam_jp1l = zeros(4*M+2, N, J);
Lam_jr   = zeros(4*M+2, N, J);
Lam_jp1r = zeros(4*M+2, N, J);

nE_jl   = zeros(N, N, J);
nE_jp1l = zeros(N, N, J);
nE_jr   = zeros(N, N, J);
nE_jp1r = zeros(N, N, J);

for j = 1 : J
    S = Impedance(Gh, krx);
    Lam_l = [eye(2*M+1);  S];
    Lam_r = [eye(2*M+1); -S];
    
    Lam_j   = Lambda(1, phi(:, :, :, j), psi(:, :, :, j), y(:, 1), h, M, N);
    Lam_jp1 = Lambda(1, phi(:, :, :, j), psi(:, :, :, j), y(:, 1), h, M, N);
    Lam_jl  (:, :, j) = Lam_j(:, 1 : N);
    Lam_jp1l(:, :, j) = Lam_jp1(:, 1 : N);
    Lam_jr  (:, :, j) = Lam_j(:, N+1 : 2*N);
    Lam_jp1r(:, :, j) = Lam_jp1(:, N+1 : 2*N);
    
    nE_j    = Epsilon(xj(j)  , kappa(:, j), xj(j), xj(j+1), N);
    nE_jp1  = Epsilon(xj(j+1), kappa(:, j), xj(j), xj(j+1), N);
    nE_jl  (:, :, j) = nE_j(1 : N, 1 : N);
    nE_jp1l(:, :, j) = nE_jp1(1 : N, 1 : N);
    nE_jr  (:, :, j) = nE_j(N+1 : 2*N, N+1 : 2*N);
    nE_jp1r(:, :, j) = nE_jp1(N+1 : 2*N, N+1 : 2*N);
end

% Building transfer matrices
T_l = cell(1, J+1);
R_r = cell(1, J+1);
R_l = cell(1, J+1);
T_r = cell(1, J+1);
mat = [-Lam_l Lam_jr(:,:,1)];
temp = pinv(mat) * [-Lam_jl(:,:,1) Lam_r] ...
    * [nE_jl(:,:,1), zeros(N, 2*M+1); zeros(2*M+1, N), eye(2*M+1)];
T_l{1} = temp(1:2*M+1      , 1:N        );
R_r{1} = temp(1:2*M+1      , N+1:2*M+1+N);
R_l{1} = temp(2*M+2:2*M+1+N, 1:N        );
T_r{1} = temp(2*M+2:2*M+1+N, N+1:2*M+1+N);

for j = 2 : J
    mat = [-Lam_jp1l(:, :, j-1) Lam_jr(:, :, j)];
    temp = pinv(mat) * [-Lam_jl(:, :, j) Lam_jp1r(:, :, j-1)] ...
        * [nE_jl(:, :, j), zeros(N, N); zeros(N, N), nE_jp1r(:,:,j-1)];
    T_l{j} = temp(1:N    , 1:N    );
    R_r{j} = temp(1:N    , N+1:2*N);
    R_l{j} = temp(N+1:2*N, 1:N    );
    T_r{j} = temp(N+1:2*N, N+1:2*N);
end

mat = [-Lam_jp1l(:, :, J) Lam_r];
temp = pinv(mat) * [-Lam_l Lam_jp1r(:,:,J)] ...
    * [eye(2*M+1), zeros(2*M+1, N); zeros(N, 2*M+1), nE_jp1r(:,:,J)];
T_l{J+1} = temp(1:N        , 1:2*M+1      );
R_r{J+1} = temp(1:N        , 2*M+2:2*M+1+N);
R_l{J+1} = temp(N+1:2*M+1+N, 1:2*M+1      );
T_r{J+1} = temp(N+1:2*M+1+N, 2*M+2:2*M+1+N);

% ii vector
ii = zeros(2*M+1, 1);
ii(M+1) = 1;

% Solution
TT_r = {};
RR_r{J+1} = R_r{J+1};
for j = J:-1:1
    TT_r{j} = inv(eye(N) - R_l{j} * RR_r{j+1}) * T_r{j};
    RR_r{j} = R_r{j} + T_l{j} * RR_r{j+1} * TT_r{j};
end

beta_r = zeros(N, J);
for j = 1 : J
    T = TT_r{1};
    for i = 2 : j
        T = TT_r{i} * T;
    end
    beta_r(:, j) = T * ii;
end

beta_l = zeros(N, J);
for j = 1 : J
    beta_l(:, j) = RR_r{j+1} * beta_r(:, j);
end
beta = [beta_l; beta_r];

rr = T_l{1  } * beta_l(:, 1) + R_r{1} * ii;
tt = T_r{J+1} * beta_r(:, J);

% Displacement field
[inter_x, inter_y] = size(x);
xi1_l = linspace(-2, 0, 100);
xi1_rt = zeros(1, J*inter_x);
for j = 1 : J
    temp = linspace(xj(j), xj(j+1), inter_x);
    xi1_rt(1, 1+(j-1)*inter_x : j*inter_x) = temp;
end
xi1_r = linspace(xj(J+1), xj(J+1)+2, 100);
xi2 = y(:, 1);

S = Impedance(Gh, krx);
Lam_l = [eye(2*M+1);  S];
Lam_r = [eye(2*M+1); -S];
uth0 = zeros(2*M+1, length(xi1_l));
for i = 1 : length(xi1_l)
    E0 = [diag(exp(1i * krx * xi1_l(i))), zeros(2*M+1, 2*M+1); zeros(2*M+1, 2*M+1), diag(exp(-1i * krx * xi1_l(i)))];
    temp = [Lam_l Lam_r] * E0 * [rr; ii];
    uth0(:, i) = temp(1 : 2*M+1);
end

uthJp1 = zeros(2*M+1, length(xi1_r));
for i = 1 : length(xi1_r)
    zeta = xi1_r(i) - xj(J+1);
    E0 = [diag(exp(1i * krx * zeta)), zeros(2*M+1, 2*M+1); zeros(2*M+1, 2*M+1), diag(exp(-1i * krx * zeta))];
    temp = [Lam_l Lam_r] * E0 * [zeros(2*M+1, 1); tt];
    uthJp1(:, i) = temp(1 : 2*M+1);
end

uthj = zeros(2*M+1, length(xi1_rt));
for j = 1 : J
    for i = 1 : length(xi1_rt) / J
        Lam = Lambda(i, phi(:, :, :, j), psi(:, :, :, j), xi2, h, M, N);
        Eps = Epsilon(xi1_rt(i + (length(xi1_rt)/J)*(j-1)), kappa(:, j), xj(j), xj(j+1), N);
        temp = Lam * Eps * beta(:, j);
        uthj(:, i + (length(xi1_rt)/J)*(j-1)) = temp(1 : 2*M+1);
    end

end

ul  = zeros(length(xi2), length(xi1_l));
urt = zeros(length(xi2), length(xi1_rt));
ur  = zeros(length(xi2), length(xi1_r));

parfor j = 1 : length(xi2)
    temp1 = zeros(1, length(xi1_l));
    for i = 1 : length(xi1_l)
        temp1(i) = sum(uth0(:, i) .* transpose(exp(1i * (2*pi/h) * (-M:M) * xi2(j)))) * exp(1i * ky * xi2(j));
    end
    temp2 = zeros(1, length(xi1_rt));
    for i = 1 : length(xi1_rt)
        temp2(i) = sum(uthj(:, i) .* transpose(exp(1i * (2*pi/h) * (-M:M) * xi2(j)))) * exp(1i * ky * xi2(j));
    end
    temp3 = zeros(1, length(xi1_l));
    for i = 1 : length(xi1_r)
        temp3(i) = sum(uthJp1(:, i) .* transpose(exp(1i * (2*pi/h) * (-M:M) * xi2(j)))) * exp(1i * ky * xi2(j));
    end
    ul(j, :)  = temp1;
    urt(j, :) = temp2;
    ur(j, :)  = temp3;
end

%--------------------------------------------------------------------------
function Eps = Epsilon(xi_1, kk, xj, xjp1, N)
%%% Function that creates Epsilon matrix given xi_1, kks, left coordinate
%%% of the strip (xj) and right coordinate of the strip (xjp1)

E = [diag(exp(1i * kk(1 : N) * xi_1 )), zeros(N, N); ...
     zeros(N, N)                      , diag(exp(1i * kk(N+1 : 2*N) * xi_1))];
H = [diag(exp(-1i * kk(1 : N) * xjp1)), zeros(N, N); ...
     zeros(N, N)                      , diag(exp(-1i * kk(N+1 : 2*N) * xj ))];
Eps = E * H;

%--------------------------------------------------------------------------
function [S] = Impedance(Gh, kx)
S = diag(1i * Gh * kx);

%--------------------------------------------------------------------------
function [Lam] = Lambda(xi_1_id, phi, psi, xi_2, h, M, N)
%%% Function Lambda(\xi_1, \xi_2) that creates Lambda matrix for given phi
%%% and psi. \xi_1_id has to be provided and also geoemtric parameters and
%%% M and N are given.

Phi = zeros(2*M+1, 2*N);
Psi = zeros(2*M+1, 2*N);
for m = 1 : 2*M+1
    for n = 1 : 2*N
        mm = m - (M+1);
        e = exp(-1i * (2*pi/h) * mm * xi_2);
        Phi(m, n) = (1/h) * trapz(xi_2, phi(:, xi_1_id, n) .* e);
        Psi(m, n) = (1/h) * trapz(xi_2, psi(:, xi_1_id, n) .* e);
    end
end

Lam = [Phi; Psi];