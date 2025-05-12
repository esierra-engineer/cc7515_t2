__kernel void nbody_step(__global float4* posMass, __global float4* vel, float dt, int n) {
    int i = get_global_id(0);
    if (i >= n) return;

    float4 pi = posMass[i];
    float3 acc = (float3)(0.0f, 0.0f, 0.0f);
    float G = 6.67430e-11f;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        float4 pj = posMass[j];
        float3 r = pj.xyz - pi.xyz;
        float distSqr = dot(r, r) + 1e-6f;
        float invDist = 1.0f / sqrt(distSqr);
        float invDist3 = invDist * invDist * invDist;
        acc += (G * pj.w) * r * invDist3;
    }

    float4 vi = vel[i];
    vi.xyz += acc * dt;
    pi.xyz += vi.xyz * dt;

    posMass[i] = pi;
    vel[i] = vi;
}
