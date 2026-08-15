__device__ const int biasx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__device__ const int biasy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__device__ const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
// __device__ const float w[9] =
// {0.444444444f,0.1111111111f,0.1111111111f,0.1111111111f,0.1111111111f,0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f};
__device__ const float si[4] = {1.1, 1.1, 1.1, 1.1};
extern "C" __global__ void fused_lbmkernel(bool *__restrict__ mask, float *__restrict__ f_now,
                                           float *__restrict__ f_out, float *__restrict__ ux, float *__restrict__ uy,
                                           float *__restrict__ rho__, const int totwidth, const int totheight,
                                           const float tau_inv,    // 内部区域的 tau_inv
                                           const float u_in,       // 入口速度
                                           const float tau_inv_bnd // 入口边界的 tau_inv
)
{
#define inv9 0.1111111111f
#define inv36 0.02777777778f
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_idy <= 0 || pixel_idy >= totheight - 1)
        return;

    if (pixel_idx >= totwidth - 1)
        return;

    int pid = pixel_idy * totwidth + pixel_idx;
    int totpixels = totwidth * totheight;

    if (pixel_idx == 0) {
        float f[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            int pullfromx = 0 - biasx[i];
            int pullfromy = pixel_idy - biasy[i];

            if (pullfromx < 0) {
                f[i] = 0.0f;
            } else {
                int sid = pullfromy * totwidth + pullfromx;
                f[i] = mask[sid] ? f_now[opp[i] * totpixels + pid] : f_now[i * totpixels + sid];
                // rho_loc += f[i];
            }
        }
        // ux_loc/=rho_loc;uy_loc/=rho_loc;
        // 平滑过渡
        float dist_to_wall = fminf((float)pixel_idy, (float)(totheight - 1 - pixel_idy));
        float smooth_factor = 1.0f;
        if (dist_to_wall < 50.0f) {
            smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
        }
        float local_u = u_in * smooth_factor;

        // Zou-He 计算
        float rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
        f[1] = f[3] + (0.666666667f) * rho * local_u;
        f[5] = f[7] - 0.5f * (f[2] - f[4]) + (0.16666666666f) * rho * local_u;
        f[8] = f[6] + 0.5f * (f[2] - f[4]) + (0.16666666666f) * rho * local_u;

        ux[pid] = local_u;
        uy[pid] = 0.0f;
        rho__[pid] = rho;
        // float m[9];
        // 碰撞步

        float rho_loc = rho;
        float ux_loc = local_u;
        float uy_loc = 0.0f;
        float m[9];
        float f14 = f[1] + f[2] + f[3] + f[4];
        float f58 = f[5] + f[6] + f[7] + f[8];

        m[0] = rho_loc;
        m[1] = -4.0f * f[0] + 2.0f * (f58)-f14;
        m[2] = 4.0f * f[0] + (f58)-2.0f * (f14);
        // m[3]=f[1]-f[3]+f[5]-f[6]-f[7]+f[8];
        m[3] = rho_loc * ux_loc;
        m[4] = -2.0f * (f[1] - f[3]) + f[5] - f[6] - f[7] + f[8];
        // m[5]=f[2]-f[4]+f[5]+f[6]-f[7]-f[8];
        m[5] = rho_loc * uy_loc;
        m[6] = -2.0f * (f[2] - f[4]) + f[5] + f[6] - f[7] - f[8];
        m[7] = f[1] - f[2] + f[3] - f[4];
        m[8] = f[5] - f[6] + f[7] - f[8];
        float usq = ux_loc * ux_loc + uy_loc * uy_loc;
        float m1eq = (3.0f * usq - 2.0f) * rho_loc;
        float m2eq = (1.0f - 3.0f * usq) * rho_loc;
        float m7eq = rho_loc * (ux_loc * ux_loc - uy_loc * uy_loc);
        float m8eq = rho_loc * ux_loc * uy_loc;
        m[1] -= si[0] * (m[1] - m1eq);
        m[2] -= si[1] * (m[2] - m2eq);
        m[4] -= si[2] * (m[4] + m[3]);
        m[6] -= si[3] * (m[6] + m[5]);
        m[7] -= tau_inv * (m[7] - m7eq);
        m[8] -= tau_inv * (m[8] - m8eq);
        // 预定义常数，用乘法代替除法

        float term0 = m[0] - m[1] + m[2];
        float term1 = 4.0f * m[0] - m[1] - 2.0f * m[2];
        float term2 = 4.0f * m[0] + 2.0f * m[1] + m[2];

        f[0] = term0 * inv9;
        f[1] = (term1 + 6.0f * m[3] - 6.0f * m[4] + 9.0f * m[7]) * inv36;
        f[2] = (term1 + 6.0f * m[5] - 6.0f * m[6] - 9.0f * m[7]) * inv36;
        f[3] = (term1 - 6.0f * m[3] + 6.0f * m[4] + 9.0f * m[7]) * inv36;
        f[4] = (term1 - 6.0f * m[5] + 6.0f * m[6] - 9.0f * m[7]) * inv36;
        f[5] = (term2 + 6.0f * m[3] + 3.0f * m[4] + 6.0f * m[5] + 3.0f * m[6] + 9.0f * m[8]) * inv36;
        f[6] = (term2 - 6.0f * m[3] - 3.0f * m[4] + 6.0f * m[5] + 3.0f * m[6] - 9.0f * m[8]) * inv36;
        f[7] = (term2 - 6.0f * m[3] - 3.0f * m[4] - 6.0f * m[5] - 3.0f * m[6] + 9.0f * m[8]) * inv36;
        f[8] = (term2 + 6.0f * m[3] + 3.0f * m[4] - 6.0f * m[5] - 3.0f * m[6] - 9.0f * m[8]) * inv36;
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            float out_val = fminf(10.0f, fmaxf(0.0f, f[i]));
            f_out[i * totpixels + pid] = out_val;

            if (pixel_idx == totwidth - 2) {
                f_out[i * totpixels + pid + 1] = out_val;
            }
        }
    }

    else {
        if (mask[pid])
            return;
        const float Cs = 0.20f;
        float rho_loc = 0.0f;
        float ux_loc = 0.0f;
        float uy_loc = 0.0f;
        float f[9];

#pragma unroll
        for (int i = 0; i < 9; ++i) {
            int pullfromx = pixel_idx - biasx[i];
            int pullfromy = pixel_idy - biasy[i];
            int sid = pullfromy * totwidth + pullfromx;
            f[i] = mask[sid] ? f_now[opp[i] * totpixels + pid] : f_now[i * totpixels + sid];
            rho_loc += f[i];
            ux_loc += f[i] * biasx[i];
            uy_loc += f[i] * biasy[i];
        }

        ux_loc /= rho_loc;
        uy_loc /= rho_loc;
        ux_loc = fminf(0.4f, fmaxf(-0.4f, ux_loc));
        uy_loc = fminf(0.4f, fmaxf(-0.4f, uy_loc));
        rho_loc = fminf(1.5f, fmaxf(0.0f, rho_loc));
        ux[pid] = ux_loc;
        uy[pid] = uy_loc;
        rho__[pid] = rho_loc;

        // 碰撞步
        float m[9];

        float f14 = f[1] + f[2] + f[3] + f[4];
        float f58 = f[5] + f[6] + f[7] + f[8];

        m[0] = rho_loc;
        m[1] = -4.0f * f[0] + 2.0f * (f58)-f14;
        m[2] = 4.0f * f[0] + (f58)-2.0f * (f14);
        // m[3]=f[1]-f[3]+f[5]-f[6]-f[7]+f[8];
        m[3] = rho_loc * ux_loc;
        m[4] = -2.0f * (f[1] - f[3]) + f[5] - f[6] - f[7] + f[8];
        // m[5]=f[2]-f[4]+f[5]+f[6]-f[7]-f[8];
        m[5] = rho_loc * uy_loc;
        m[6] = -2.0f * (f[2] - f[4]) + f[5] + f[6] - f[7] - f[8];
        m[7] = f[1] - f[2] + f[3] - f[4];
        m[8] = f[5] - f[6] + f[7] - f[8];
        float usq = ux_loc * ux_loc + uy_loc * uy_loc;
        float m1eq = (3.0f * usq - 2.0f) * rho_loc;
        float m2eq = (1.0f - 3.0f * usq) * rho_loc;
        float m7eq = rho_loc * (ux_loc * ux_loc - uy_loc * uy_loc);
        float m8eq = rho_loc * ux_loc * uy_loc;
        // --- Smagorinsky LES model ---
        float m7_neq = m[7] - m7eq;
        float m8_neq = m[8] - m8eq;
        float Q = sqrtf(m7_neq * m7_neq + m8_neq * m8_neq + 1e-8f);
        const float tau_0 = 1.0f / tau_inv;
        float tau_eff = 0.5f * (tau_0 + sqrtf(tau_0 * tau_0 + 18.0f * Cs * Cs * Q / rho_loc));
        float tau_inv_eff = 1.0f / tau_eff;

        // MRT relaxation with Smagorinsky-modified stress relaxation
        m[1] -= si[0] * (m[1] - m1eq);
        m[2] -= si[1] * (m[2] - m2eq);
        m[4] -= si[2] * (m[4] + m[3]);
        m[6] -= si[3] * (m[6] + m[5]);
        m[7] -= tau_inv_eff * (m[7] - m7eq);
        m[8] -= tau_inv_eff * (m[8] - m8eq);

        float term0 = m[0] - m[1] + m[2];
        float term1 = 4.0f * m[0] - m[1] - 2.0f * m[2];
        float term2 = 4.0f * m[0] + 2.0f * m[1] + m[2];

        f[0] = term0 * inv9;
        f[1] = (term1 + 6.0f * m[3] - 6.0f * m[4] + 9.0f * m[7]) * inv36;
        f[2] = (term1 + 6.0f * m[5] - 6.0f * m[6] - 9.0f * m[7]) * inv36;
        f[3] = (term1 - 6.0f * m[3] + 6.0f * m[4] + 9.0f * m[7]) * inv36;
        f[4] = (term1 - 6.0f * m[5] + 6.0f * m[6] - 9.0f * m[7]) * inv36;
        f[5] = (term2 + 6.0f * m[3] + 3.0f * m[4] + 6.0f * m[5] + 3.0f * m[6] + 9.0f * m[8]) * inv36;
        f[6] = (term2 - 6.0f * m[3] - 3.0f * m[4] + 6.0f * m[5] + 3.0f * m[6] - 9.0f * m[8]) * inv36;
        f[7] = (term2 - 6.0f * m[3] - 3.0f * m[4] - 6.0f * m[5] - 3.0f * m[6] + 9.0f * m[8]) * inv36;
        f[8] = (term2 + 6.0f * m[3] + 3.0f * m[4] - 6.0f * m[5] - 3.0f * m[6] - 9.0f * m[8]) * inv36;
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            // 限幅防止在刚启动时因为极端边界产生负数或爆炸
            float out_val = fminf(10.0f, fmaxf(0.0f, f[i]));
            f_out[i * totpixels + pid] = out_val;

            // 【精华】：顺手处理 right_out 逻辑，零开销！
            if (pixel_idx == totwidth - 2) {
                f_out[i * totpixels + pid + 1] = out_val;
            }
        }
    }
}

extern "C" __global__ void visualizekernel(const float *__restrict__ ux, const float *__restrict__ uy,
                                           const float *__restrict__ rho_, unsigned char *__restrict__ image,
                                           const bool *__restrict__ mask, const int totwidth, const int totheight,
                                           const float vort_scale)
{

    // int totpixels = totwidth * totheight;
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;
    int pid = pixel_idx + pixel_idy * totwidth;

    if (pixel_idx <= 0 || pixel_idx >= totwidth - 1 || pixel_idy <= 0 || pixel_idy >= totheight - 1)
        return;
    image[pid * 4 + 3] = 255;
    if (mask[pid]) {
        image[pid * 4] = 0;
        image[pid * 4 + 1] = 0;
        image[pid * 4 + 2] = 0;
        return;
    }
    int pid_right = pixel_idy * totwidth + (pixel_idx + 1);
    int pid_left = pixel_idy * totwidth + (pixel_idx - 1);
    int pid_top = (pixel_idy + 1) * totwidth + pixel_idx;
    int pid_bot = (pixel_idy - 1) * totwidth + pixel_idx;
    // float rhoo = rho_[pid];
    float vort = ((uy[pid_right] - uy[pid_left]) - (ux[pid_top] - ux[pid_bot])) * vort_scale;
    // 如果你喜欢你同学的赛博朋克风格，可以用下面这三行替换上面三行：
    // float r = fminf(fmaxf(fabsf(vort), 0.0f), 1.0f);
    // float g = r;
    // float b = 0.5f;
    // float speed = sqrtf(ux[pid]*ux[pid] + uy[pid]*uy[pid]);
    // float r = rhoo;// 放大 5 倍观察
    // float g = r;
    // float b = 0.5f; // 这就是你看到的蓝色背景来源

    float r = fminf(fmaxf(1.0f + vort, 0.0f), 1.0f);
    float g = fminf(fmaxf(1.0f - fabsf(vort), 0.0f), 1.0f);
    float b = fminf(fmaxf(1.0f - vort, 0.0f), 1.0f);
    image[pid * 4 + 0] = (unsigned char)(r * 255);
    image[pid * 4 + 1] = (unsigned char)(g * 255);
    image[pid * 4 + 2] = (unsigned char)(b * 255);
}