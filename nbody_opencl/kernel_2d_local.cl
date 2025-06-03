__kernel void nbody_step(__global float4* posMass,
                         __global float4* vel,
                         float dt,
                         int n,
                         __local float4* sharedBodies) {

    int i = get_global_id(1);
    if (i >= n) return;

    int local_id_x = get_local_id(0);
    int local_size_x = get_local_size(0);
    int j_start = get_group_id(0) * local_size_x;

    float4 pi = posMass[i];
    float3 acc = (float3)(0.0f, 0.0f, 0.0f);
    float G = 6.67430e-11f;

    int j = j_start + local_id_x;
    if (j < n) {
        sharedBodies[local_id_x] = posMass[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int max_j = min(local_size_x, n - j_start);
    for (int k = 0; k < max_j; ++k) {
        float4 pj = sharedBodies[k];
        if (i == j_start + k) continue;

        float3 r = pj.xyz - pi.xyz;
        float distSqr = dot(r, r) + 1e-6f;
        float invDist = 1.0f / sqrt(distSqr);
        float invDist3 = invDist * invDist * invDist;
        acc += (G * pj.w) * r * invDist3;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 vi = vel[i];
    vi.xyz += acc * dt;
    pi.xyz += vi.xyz * dt;

    posMass[i] = pi;
    vel[i] = vi;
}
