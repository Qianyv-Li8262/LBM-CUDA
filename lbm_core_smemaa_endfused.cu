__device__ const int biasx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__device__ const int biasy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__device__ const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
__device__ const float w[9] = {0.444444444f,  0.1111111111f, 0.1111111111f, 0.1111111111f, 0.1111111111f,
                               0.0277777778f, 0.0277777778f, 0.0277777778f, 0.0277777778f};

constexpr int T = 6;
constexpr int SQUARE_A = 24;
constexpr int SMEM_SIZE = SQUARE_A + 2 * T;
constexpr int AREA = SMEM_SIZE * SMEM_SIZE;

extern "C" __global__ void fused_lbmkernel(const bool *__restrict__ mask, const float *__restrict__ f_now,
                                           float *__restrict__ f_out, const int totwidth, const int totheight,
                                           const float tau_inv, const float u_in)
{
    static_assert((T & 1) == 0 && T >= 2, "T must be an even integer >= 2");

    __shared__ float sregion[9][SMEM_SIZE][SMEM_SIZE];
    __shared__ unsigned char smask[SMEM_SIZE][SMEM_SIZE];

    const int total_area = totwidth * totheight;
    const int block_stride = blockDim.x * blockDim.y;
    const int block_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_x = SQUARE_A * blockIdx.x;
    const int block_y = SQUARE_A * blockIdx.y;

    const int outlet_fluid_smem_x = (totwidth - 2) - (block_x - T);
    const int outlet_ghost_smem_x = outlet_fluid_smem_x + 1;
    const bool has_outlet = outlet_fluid_smem_x >= 0 && outlet_fluid_smem_x < SMEM_SIZE && outlet_ghost_smem_x >= 0 &&
                            outlet_ghost_smem_x < SMEM_SIZE;

    // ========================================================================
    // A0 SPECIALIZATION
    // GMEM canonical -> registers -> BGK -> reversed SMEM.
    // 同一个 traversal 顺便把完整 mask halo 放入 shared；A0 本身不读邻居，所以无需前置 barrier。
    // ========================================================================
    for (int i = block_tid; i < AREA; i += block_stride) {
        const int smem_x = i % SMEM_SIZE;
        const int smem_y = i / SMEM_SIZE;
        const int thread_x = block_x - T + smem_x;
        const int thread_y = block_y - T + smem_y;
        const bool in_domain = thread_x >= 0 && thread_x < totwidth && thread_y >= 0 && thread_y < totheight;

        bool is_solid = true;
        int linear_idx = 0;
        if (in_domain) {
            linear_idx = thread_y * totwidth + thread_x;
            is_solid = mask[linear_idx];
            if (thread_x == totwidth - 1)
                is_solid = false; // open outlet ghost
            smask[smem_y][smem_x] = is_solid ? 1 : 0;
        } else {
            smask[smem_y][smem_x] = 1;
        }

        if (!in_domain || thread_y <= 0 || thread_y >= totheight - 1 || thread_x >= totwidth - 1 || is_solid)
            continue;

        float f[9];
#pragma unroll
        for (int read = 0; read < 9; ++read)
            f[read] = f_now[read * total_area + linear_idx];

        float rho = 0.0f, ux_loc = 0.0f, uy_loc = 0.0f;
        if (thread_x == 0) {
            float dist_to_wall = fminf((float)thread_y, (float)(totheight - 1 - thread_y));
            float smooth_factor = 1.0f;
            if (dist_to_wall < 50.0f)
                smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
            float local_u = u_in * smooth_factor;
            rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
            f[1] = f[3] + 0.666666667f * rho * local_u;
            f[5] = f[7] - 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
            f[8] = f[6] + 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
            ux_loc = local_u;
        } else {
#pragma unroll
            for (int use = 0; use < 9; ++use) {
                rho += f[use];
                ux_loc += f[use] * biasx[use];
                uy_loc += f[use] * biasy[use];
            }
            ux_loc /= rho;
            uy_loc /= rho;
        }

        const float usq = ux_loc * ux_loc + uy_loc * uy_loc;
#pragma unroll
        for (int use = 0; use < 9; ++use) {
            float cu = biasx[use] * ux_loc + biasy[use] * uy_loc;
            float feq = w[use] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
            f[use] += tau_inv * (feq - f[use]);
        }
#pragma unroll
        for (int store = 0; store < 9; ++store)
            sregion[opp[store]][smem_y][smem_x] = fminf(10.0f, fmaxf(0.0f, f[store]));
    }

    // A0 和完整 smask 都 ready。
    __syncthreads();

    // A 后只需要 reversed outlet ghost，供紧随其后的 B 使用。
    if (has_outlet) {
        for (int line = block_tid; line < SMEM_SIZE; line += block_stride) {
            int thread_y = block_y - T + line;
            if (thread_y <= 0 || thread_y >= totheight - 1)
                continue;
#pragma unroll
            for (int j = 0; j < 9; ++j)
                sregion[j][line][outlet_ghost_smem_x] = sregion[j][line][outlet_fluid_smem_x];
        }
    }
    __syncthreads();

    // ========================================================================
    // MIDDLE: B -> A -> B -> A ...
    // A0 已经做掉第一步，Bfinal 留给最后一步，因此这里一共 T/2-1 对 B/A。
    // B 后不复制 canonical outlet：后面的 A 完全 local，不需要 ghost；A 后再生成 reversed ghost 即可。
    // ========================================================================
#pragma unroll
    for (int times = 0; times < T / 2 - 1; ++times) {
        // --------------------------------------------------------------------
        // B middle: reversed SMEM -> canonical SMEM
        // --------------------------------------------------------------------
        int margin = 2 * times + 1;
        int calculation_size = SMEM_SIZE - 2 * margin;
        int calculation_area = calculation_size * calculation_size;

        for (int i = block_tid; i < calculation_area; i += block_stride) {
            int smem_x = margin + i % calculation_size;
            int smem_y = margin + i / calculation_size;
            int thread_x = block_x - T + smem_x;
            int thread_y = block_y - T + smem_y;

            if (thread_x < 0 || thread_x >= totwidth || thread_y <= 0 || thread_y >= totheight - 1)
                continue;
            if (thread_x >= totwidth - 1 || smask[smem_y][smem_x])
                continue;

            float f[9];
#pragma unroll
            for (int read = 0; read < 9; ++read) {
                int source_x = smem_x - biasx[read];
                int source_y = smem_y - biasy[read];
                f[read] =
                    smask[source_y][source_x] ? sregion[read][smem_y][smem_x] : sregion[opp[read]][source_y][source_x];
            }

            float rho = 0.0f, ux_loc = 0.0f, uy_loc = 0.0f;
            if (thread_x == 0) {
                float dist_to_wall = fminf((float)thread_y, (float)(totheight - 1 - thread_y));
                float smooth_factor = 1.0f;
                if (dist_to_wall < 50.0f)
                    smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
                float local_u = u_in * smooth_factor;
                rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
                f[1] = f[3] + 0.666666667f * rho * local_u;
                f[5] = f[7] - 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
                f[8] = f[6] + 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
                ux_loc = local_u;
            } else {
#pragma unroll
                for (int use = 0; use < 9; ++use) {
                    rho += f[use];
                    ux_loc += f[use] * biasx[use];
                    uy_loc += f[use] * biasy[use];
                }
                ux_loc /= rho;
                uy_loc /= rho;
            }

            float usq = ux_loc * ux_loc + uy_loc * uy_loc;
#pragma unroll
            for (int use = 0; use < 9; ++use) {
                float cu = biasx[use] * ux_loc + biasy[use] * uy_loc;
                float feq = w[use] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
                f[use] += tau_inv * (feq - f[use]);
            }

#pragma unroll
            for (int store = 0; store < 9; ++store) {
                int dest_x = smem_x + biasx[store];
                int dest_y = smem_y + biasy[store];
                float out_val = fminf(10.0f, fmaxf(0.0f, f[store]));
                bool left_open_boundary = thread_x == 0 && biasx[store] < 0;
                if (left_open_boundary)
                    sregion[store][dest_y][dest_x] = out_val;
                else if (smask[dest_y][dest_x])
                    sregion[opp[store]][smem_y][smem_x] = out_val;
                else
                    sregion[store][dest_y][dest_x] = out_val;
            }
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // A middle: canonical SMEM -> reversed SMEM
        // --------------------------------------------------------------------
        margin = 2 * times + 2;
        calculation_size = SMEM_SIZE - 2 * margin;
        calculation_area = calculation_size * calculation_size;

        for (int i = block_tid; i < calculation_area; i += block_stride) {
            int smem_x = margin + i % calculation_size;
            int smem_y = margin + i / calculation_size;
            int thread_x = block_x - T + smem_x;
            int thread_y = block_y - T + smem_y;

            if (thread_x < 0 || thread_x >= totwidth || thread_y <= 0 || thread_y >= totheight - 1)
                continue;
            if (thread_x >= totwidth - 1 || smask[smem_y][smem_x])
                continue;

            float f[9];
#pragma unroll
            for (int read = 0; read < 9; ++read)
                f[read] = sregion[read][smem_y][smem_x];

            float rho = 0.0f, ux_loc = 0.0f, uy_loc = 0.0f;
            if (thread_x == 0) {
                float dist_to_wall = fminf((float)thread_y, (float)(totheight - 1 - thread_y));
                float smooth_factor = 1.0f;
                if (dist_to_wall < 50.0f)
                    smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
                float local_u = u_in * smooth_factor;
                rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
                f[1] = f[3] + 0.666666667f * rho * local_u;
                f[5] = f[7] - 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
                f[8] = f[6] + 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
                ux_loc = local_u;
            } else {
#pragma unroll
                for (int use = 0; use < 9; ++use) {
                    rho += f[use];
                    ux_loc += f[use] * biasx[use];
                    uy_loc += f[use] * biasy[use];
                }
                ux_loc /= rho;
                uy_loc /= rho;
            }

            float usq = ux_loc * ux_loc + uy_loc * uy_loc;
#pragma unroll
            for (int use = 0; use < 9; ++use) {
                float cu = biasx[use] * ux_loc + biasy[use] * uy_loc;
                float feq = w[use] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
                f[use] += tau_inv * (feq - f[use]);
            }
#pragma unroll
            for (int store = 0; store < 9; ++store)
                sregion[opp[store]][smem_y][smem_x] = fminf(10.0f, fmaxf(0.0f, f[store]));
        }
        __syncthreads();

        // A 后生成下一 B 所需的 reversed outlet ghost。
        if (has_outlet) {
            for (int line = block_tid; line < calculation_size; line += block_stride) {
                int smem_y = margin + line;
                int thread_y = block_y - T + smem_y;
                if (thread_y <= 0 || thread_y >= totheight - 1)
                    continue;
#pragma unroll
                for (int j = 0; j < 9; ++j)
                    sregion[j][smem_y][outlet_ghost_smem_x] = sregion[j][smem_y][outlet_fluid_smem_x];
            }
        }
        __syncthreads();
    }

    // ========================================================================
    // B FINAL SPECIALIZATION
    // reversed SMEM -> registers -> BGK -> directly scatter to f_out.
    // 不再生成 final canonical sregion，也没有最后一次 shared store/read。
    // ========================================================================
    constexpr int FINAL_MARGIN = T - 1;
    constexpr int FINAL_SIZE = SQUARE_A + 2;
    constexpr int FINAL_AREA = FINAL_SIZE * FINAL_SIZE;

    for (int i = block_tid; i < FINAL_AREA; i += block_stride) {
        int smem_x = FINAL_MARGIN + i % FINAL_SIZE;
        int smem_y = FINAL_MARGIN + i / FINAL_SIZE;
        int thread_x = block_x - T + smem_x;
        int thread_y = block_y - T + smem_y;

        if (thread_x < 0 || thread_x >= totwidth || thread_y <= 0 || thread_y >= totheight - 1)
            continue;
        if (thread_x >= totwidth - 1 || smask[smem_y][smem_x])
            continue;

        float f[9];
#pragma unroll
        for (int read = 0; read < 9; ++read) {
            int source_x = smem_x - biasx[read];
            int source_y = smem_y - biasy[read];
            f[read] =
                smask[source_y][source_x] ? sregion[read][smem_y][smem_x] : sregion[opp[read]][source_y][source_x];
        }

        float rho = 0.0f, ux_loc = 0.0f, uy_loc = 0.0f;
        if (thread_x == 0) {
            float dist_to_wall = fminf((float)thread_y, (float)(totheight - 1 - thread_y));
            float smooth_factor = 1.0f;
            if (dist_to_wall < 50.0f)
                smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
            float local_u = u_in * smooth_factor;
            rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
            f[1] = f[3] + 0.666666667f * rho * local_u;
            f[5] = f[7] - 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
            f[8] = f[6] + 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
            ux_loc = local_u;
        } else {
#pragma unroll
            for (int use = 0; use < 9; ++use) {
                rho += f[use];
                ux_loc += f[use] * biasx[use];
                uy_loc += f[use] * biasy[use];
            }
            ux_loc /= rho;
            uy_loc /= rho;
        }

        float usq = ux_loc * ux_loc + uy_loc * uy_loc;
#pragma unroll
        for (int use = 0; use < 9; ++use) {
            float cu = biasx[use] * ux_loc + biasy[use] * uy_loc;
            float feq = w[use] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
            f[use] += tau_inv * (feq - f[use]);
        }

#pragma unroll
        for (int store = 0; store < 9; ++store) {
            int dest_smem_x = smem_x + biasx[store];
            int dest_smem_y = smem_y + biasy[store];
            int target_x, target_y, target_q;
            float out_val = fminf(10.0f, fmaxf(0.0f, f[store]));

            // x=0 向域外飞出的 population 直接丢弃；下一 A0 的 incoming 由 Zou-He 重建。
            if (thread_x == 0 && biasx[store] < 0)
                continue;

            if (smask[dest_smem_y][dest_smem_x]) {
                // bounce-back: canonical target = self, opp(q)
                target_x = thread_x;
                target_y = thread_y;
                target_q = opp[store];
            } else {
                // normal propagation: canonical target = self + c_q, q
                target_x = thread_x + biasx[store];
                target_y = thread_y + biasy[store];
                target_q = store;
            }

            // W-1 的普通 streamed 值不是我们的 outlet 定义；ghost 由 W-2 的 canonical 值复制得到。
            if (target_x == totwidth - 1)
                continue;

            bool target_in_core = target_x >= block_x && target_x < block_x + SQUARE_A && target_y >= block_y &&
                                  target_y < block_y + SQUARE_A;
            if (!target_in_core || target_x < 0 || target_x >= totwidth || target_y < 0 || target_y >= totheight)
                continue;

            int target_idx = target_y * totwidth + target_x;
            f_out[target_q * total_area + target_idx] = out_val;

            // 保持原 zero-gradient outlet: f_q(W-1,y) = f_q(W-2,y)。
            // 只有 W-2 的 owner CTA 会走到这里，因此不会产生跨 CTA 写冲突。
            if (target_x == totwidth - 2)
                f_out[target_q * total_area + target_y * totwidth + (totwidth - 1)] = out_val;
        }
    }
}

// ============================================================================
// 宏观量只在需要可视化时提取一次，而不是每个 4-step simulation kernel 都写 ux/uy/rho。
// x=0 的 incoming PDFs 在 canonical buffer 中不要求有效，因此这里也现场做一次 Zou-He。
// ============================================================================
extern "C" __global__ void macro_kernel(const float *__restrict__ f_now, float *__restrict__ ux, float *__restrict__ uy,
                                        float *__restrict__ rho__, const bool *__restrict__ mask, const int totwidth,
                                        const int totheight, const float u_in)
{
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_idx < 0 || pixel_idx >= totwidth - 1 || pixel_idy <= 0 || pixel_idy >= totheight - 1)
        return;

    int pid = pixel_idy * totwidth + pixel_idx;
    if (mask[pid])
        return;
    int total_area = totwidth * totheight;

    float f[9];
#pragma unroll
    for (int i = 0; i < 9; ++i)
        f[i] = f_now[i * total_area + pid];

    float rho = 0.0f, ux_loc = 0.0f, uy_loc = 0.0f;
    if (pixel_idx == 0) {
        float dist_to_wall = fminf((float)pixel_idy, (float)(totheight - 1 - pixel_idy));
        float smooth_factor = 1.0f;
        if (dist_to_wall < 50.0f)
            smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
        float local_u = u_in * smooth_factor;
        rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
        f[1] = f[3] + 0.666666667f * rho * local_u;
        f[5] = f[7] - 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
        f[8] = f[6] + 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
        ux_loc = local_u;
    } else {
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            rho += f[i];
            ux_loc += f[i] * biasx[i];
            uy_loc += f[i] * biasy[i];
        }
        ux_loc /= rho;
        uy_loc /= rho;
    }

    ux[pid] = ux_loc;
    uy[pid] = uy_loc;
    rho__[pid] = rho;
}

extern "C" __global__ void visualizekernel(const float *__restrict__ ux, const float *__restrict__ uy,
                                           const float *__restrict__ rho_, unsigned char *__restrict__ image,
                                           const bool *__restrict__ mask, const int totwidth, const int totheight,
                                           const float vort_scale)
{
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_idx <= 0 || pixel_idx >= totwidth - 1 || pixel_idy <= 0 || pixel_idy >= totheight - 1)
        return;

    int pid = pixel_idx + pixel_idy * totwidth;
    image[pid * 4 + 3] = 255;
    if (mask[pid]) {
        image[pid * 4] = 0;
        image[pid * 4 + 1] = 0;
        image[pid * 4 + 2] = 0;
        return;
    }

    int pid_right = pixel_idy * totwidth + pixel_idx + 1;
    int pid_left = pixel_idy * totwidth + pixel_idx - 1;
    int pid_top = (pixel_idy + 1) * totwidth + pixel_idx;
    int pid_bot = (pixel_idy - 1) * totwidth + pixel_idx;
    float vort = ((uy[pid_right] - uy[pid_left]) - (ux[pid_top] - ux[pid_bot])) * vort_scale;
    float r = fminf(fmaxf(1.0f + vort, 0.0f), 1.0f);
    float g = fminf(fmaxf(1.0f - fabsf(vort), 0.0f), 1.0f);
    float b = fminf(fmaxf(1.0f - vort, 0.0f), 1.0f);
    image[pid * 4 + 0] = (unsigned char)(r * 255);
    image[pid * 4 + 1] = (unsigned char)(g * 255);
    image[pid * 4 + 2] = (unsigned char)(b * 255);
}
