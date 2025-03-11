struct Params {
    size_x: u32,
    size_y: u32,
    size_z: u32,
    copy_x: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid: array<f32>;
@group(0) @binding(2) var grid_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1, 256)
fn main(@builtin(global_invocation_id) _i: vec3<u32>) {
    let width = textureDimensions(grid_tex).x;
    let height = textureDimensions(grid_tex).y;
    let iy = _i.x;
    let iz = _i.y;
    if (iz >= width) { return; }
    if (iy >= height) { return; }

    let Nx = params.size_x;
    let Ny = params.size_y;
    let Nz = params.size_z;
    let Nzy = Ny*Nz;
    let src_x = params.copy_x;

    let n_dims = u32(3);
    let src_i = n_dims*(src_x*Nzy + iy*Nz + iz);
    let dst_i = vec2<u32>(u32(iz), u32(iy));

    let Ex: f32 = grid[src_i+0];
    let Ey: f32 = grid[src_i+1];
    let Ez: f32 = grid[src_i+2];
    let E_vec: vec3<f32> = vec3<f32>(Ex,Ey,Ez);
    // let colour = vec4<f32>(abs(E_vec), 1.0);
    let v = length(E_vec) * params.scale;
    let colour = vec4<f32>(v,v,v,1.0);
    textureStore(grid_tex, dst_i, colour);
}
