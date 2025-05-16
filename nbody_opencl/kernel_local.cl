__kernel void nbody_step(__global float4* posMass, __global float4* vel, float dt, int n) {
    int i = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    __local float4 sharedBodies[256]; // Tamaño fijo: asegúrate de no usar más de esto

    if (i >= n) return;

    float4 pi = posMass[i];
    float3 acc = (float3)(0.0f, 0.0f, 0.0f);
    float G = 6.67430e-11f;

    for (int block = 0; block < n; block += local_size) {
        int j = block + local_id;
        if (j < n) {
            sharedBodies[local_id] = posMass[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int max_j = min(local_size, n - block);
        for (int k = 0; k < max_j; ++k) {
            float4 pj = sharedBodies[k];
            if (i == block + k) continue;

            float3 r = pj.xyz - pi.xyz;
            float distSqr = dot(r, r) + 1e-6f;
            float invDist = 1.0f / sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;
            acc += (G * pj.w) * r * invDist3;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float4 vi = vel[i];
    vi.xyz += acc * dt;
    pi.xyz += vi.xyz * dt;

    posMass[i] = pi;
    vel[i] = vi;
}
