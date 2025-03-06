__kernel void update_E(
    __global float *E, __global const float *H, __global float *cH, 
    __global float *A_0, __global float *A_1,
    int Nx, int Ny, int Nz
) {
    // a = 1/2 * Co/ek * sigma_k * dxyz * Z0
    // a_0 = (1-a)/(1+a)
    // a_1 = Co/ek * 1/(1+a)

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

    const float a_0 = A_0[i0];
    const float a_1 = A_1[i0];

    E[i+0] = a_0*(E[i+0] + a_1*cHx);
    E[i+1] = a_0*(E[i+1] + a_1*cHy);
    E[i+2] = a_0*(E[i+2] + a_1*cHz);

    cH[i+0] = cHx;
    cH[i+1] = cHy;
    cH[i+2] = cHz;
    return;
}

__kernel void update_H(
    __global const float *E, __global float *H, __global float *cE,
    float b_0,
    int Nx, int Ny, int Nz
) {
    // a_0 = Co/ek
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

    H[i+0] = H[i+0] - b_0*cEx;
    H[i+1] = H[i+1] - b_0*cEy;
    H[i+2] = H[i+2] - b_0*cEz;

    cE[i+0] = cEx;
    cE[i+1] = cEy;
    cE[i+2] = cEz;
    return;
}
