__kernel void update_current_source(
    __global float *E, const float E0,
    int Nx, int Ny, int Nz
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    if (ix >= Nx) return;
    if (iy >= Ny) return;
    if (iz >= Nz) return;

    const int n_dims = 3;
    const int Nzy = Nz*Ny;
    const int i0 = iz + iy*Nz + ix*Nzy;
    const int i = n_dims*i0;

    E[i+0] += E0;
}

__kernel void update_E(
    __global float *E, __global const float *H,
    __global const float *A0, __global const float *A1,
    int Nx, int Ny, int Nz
) {
    // a0 = 1/(1+sigma_k/e_k*dt)
    // a1 = 1/(e_k*d_xyz) * dt

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    if (ix >= Nx) return;
    if (iy >= Ny) return;
    if (iz >= Nz) return;

    const int n_dims = 3;
    const int Nzy = Nz*Ny;
    const int i0 = iz + iy*Nz + ix*Nzy;
    const int i = n_dims*i0;
    const int dz = n_dims*(((iz+1) % Nz) + iy*Nz + ix*Nzy);
    const int dy = n_dims*(iz + ((iy+1) % Ny)*Nz + ix*Nzy);
    const int dx = n_dims*(iz + iy*Nz + ((ix+1) % Nx)*Nzy);

    // curl(H)
    const float dHz_dy = H[dy+2]-H[i+2];
    const float dHy_dz = H[dz+1]-H[i+1];
    const float dHx_dz = H[dz+0]-H[i+0];
    const float dHz_dx = H[dx+2]-H[i+2];
    const float dHy_dx = H[dx+1]-H[i+1]; 
    const float dHx_dy = H[dy+0]-H[i+0];
    const float cHx = dHz_dy-dHy_dz;
    const float cHy = dHx_dz-dHz_dx;
    const float cHz = dHy_dx-dHx_dy;

    const float a0 = A0[i0];
    const float a1 = A1[i0];

    E[i+0] = a0*(E[i+0] + a1*cHx);
    E[i+1] = a0*(E[i+1] + a1*cHy);
    E[i+2] = a0*(E[i+2] + a1*cHz);
    return;
}

__kernel void update_H(
    __global const float *E, __global float *H,
    float b0,
    int Nx, int Ny, int Nz
) {
    // b0 = 1/(mu_k*d_xyz) * dt
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    if (ix >= Nx) return;
    if (iy >= Ny) return;
    if (iz >= Nz) return;

    const int n_dims = 3;
    const int Nzy = Nz*Ny;
    const int i0 = iz + iy*Nz + ix*Nzy;
    const int i = n_dims*i0;
    const int dz = n_dims*(((iz-1+Nz) % Nz) + iy*Nz + ix*Nzy);
    const int dy = n_dims*(iz + ((iy-1+Ny) % Ny)*Nz + ix*Nzy);
    const int dx = n_dims*(iz + iy*Nz + ((ix-1+Nx) % Nx)*Nzy);

    // curl(E)
    const float dEz_dy = E[i+2]-E[dy+2];
    const float dEy_dz = E[i+1]-E[dz+1];
    const float dEx_dz = E[i+0]-E[dz+0];
    const float dEz_dx = E[i+2]-E[dx+2];
    const float dEy_dx = E[i+1]-E[dx+1]; 
    const float dEx_dy = E[i+0]-E[dy+0];
    const float cEx = dEz_dy-dEy_dz;
    const float cEy = dEx_dz-dEz_dx;
    const float cEz = dEy_dx-dEx_dy;

    H[i+0] = H[i+0] - b0*cEx;
    H[i+1] = H[i+1] - b0*cEy;
    H[i+2] = H[i+2] - b0*cEz;
    return;
}
