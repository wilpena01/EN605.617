
__kernel void add_cl(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}

__kernel void sub_cl(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] - b[gid];
}


__kernel void mul_cl(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] * b[gid];
}

__kernel void mod_cl(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    if(b[gid]!=0)
        result[gid] = (int)a[gid] / (int)b[gid];
    else
        result[gid] = -9999;
}

__kernel void pow_cl(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    float base = a[gid];
    int exp  = (int)b[gid];
    float r =1; 

    if(exp != 0)
    {
        for(int i = 1; i<exp; i++)
            r += a[gid];
    }
    else
        r = 1;

    result[gid] = (float)r;

}

