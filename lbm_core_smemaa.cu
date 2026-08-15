__device__ const int biasx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__device__ const int biasy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__device__ const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
__device__ const float w[9] = {0.444444444f,  0.1111111111f, 0.1111111111f, 0.1111111111f, 0.1111111111f,
                               0.0277777778f, 0.0277777778f, 0.0277777778f, 0.0277777778f};
#define inv9 0.1111111111f
#define inv36 0.02777777778f
constexpr int T = 4;
constexpr int SQUARE_A = 24;
constexpr int SMEM_SIZE = SQUARE_A + 2 * T;
constexpr int AREA = SMEM_SIZE * SMEM_SIZE;
extern "C" __global__ void fused_lbmkernel(bool *__restrict__ mask, const float *__restrict__ f_now,
                                           float *__restrict__ f_out, float *__restrict__ ux, float *__restrict__ uy,
                                           float *__restrict__ rho__, const int totwidth, const int totheight,
                                           const float tau_inv, const float u_in)
{
    static_assert((T & 1) == 0, "T must be even for AA A/B pairs");
    __shared__ float sregion[9][SMEM_SIZE][SMEM_SIZE];
    __shared__ char smask[SMEM_SIZE][SMEM_SIZE];
    int total_area = totheight * totwidth;
    int block_stride = blockDim.x * blockDim.y;
    int block_tid = threadIdx.y * blockDim.x + threadIdx.x;
    /*
        blockDim 只决定 worker 数量。

        每个 CTA 真正负责的输出 core
        始终是 SQUARE_A x SQUARE_A。
    */
    int block_x = SQUARE_A * blockIdx.x;
    int block_y = SQUARE_A * blockIdx.y;
    // ============================================================
    // Load 28 x 28 region
    // ============================================================
    for (int i = block_tid; i < AREA; i += block_stride) {
        int smem_x = i % SMEM_SIZE;
        int smem_y = i / SMEM_SIZE;
        int thread_x = block_x - T + smem_x;
        int thread_y = block_y - T + smem_y;
        bool in_domain = thread_x >= 0 && thread_x < totwidth && thread_y >= 0 && thread_y < totheight;
        if (in_domain) {
            int linear_idx = thread_y * totwidth + thread_x;
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                sregion[j][smem_y][smem_x] =f_now[j * total_area + linear_idx];
            }
            /*
                W-1 是 open outlet ghost，
                即便用户 mask 里意外给它标了 true，
                AA 内也不能把它当 bounce-back wall。
            */
            if (thread_x == totwidth - 1) {
                smask[smem_y][smem_x] = 0.0f;
            } else {
                smask[smem_y][smem_x] = mask[linear_idx] ? 1.0f : 0.0f;
            }
        } else {
            /*
                域外暂时按 solid placeholder。

                左侧 x < 0 是个例外：
                对 x=0 的 incoming PDFs，
                B phase 后面会用 Zou-He 覆盖，
                所以不会真的使用这里的 bounce-back 值。
            */
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                sregion[j][smem_y][smem_x] = 0.0f;
            }
            smask[smem_y][smem_x] = 1.0f;
        }
    }
    __syncthreads();
    // ============================================================
    //
    // T = 6:
    //
    // A0 : 28 x 28
    // B0 : 26 x 26
    //       -> 24 x 24 canonical valid
    //
    // A1 : 24 x 24
    // B1 : 22 x 22
    //       -> 20 x 20 canonical valid
    //
    // A2 : 20 x 20
    // B2 : 18 x 18
    //       -> 16 x 16 canonical valid
    //
    // ============================================================
#pragma unroll
    for (int times = 0; times < T / 2; ++times) {
        // ========================================================
        // A PHASE
        //
        // canonical
        //      ->
        // collision
        //      ->
        // reversed
        //
        // read:
        //     sregion[q][self]
        //
        // store:
        //     sregion[opp[q]][self]
        // ========================================================
        int margin = 2 * times;
        int calculation_size = SMEM_SIZE - 2 * margin;
        int calculation_area = calculation_size * calculation_size;
        for (int i = block_tid; i < calculation_area; i += block_stride) {
            int smem_x = margin + i % calculation_size;
            int smem_y = margin + i / calculation_size;
            int thread_x = block_x - T + smem_x;
            int thread_y = block_y - T + smem_y;
            // --------------------------------------------
            // physical-domain protection
            // --------------------------------------------
            if (thread_x < 0 || thread_x >= totwidth || thread_y < 0 || thread_y >= totheight) {
                continue;
            }
            /*
                保持原 kernel 语义：

                y = 0
                y = H-1

                本身不进行 collision。
            */
            if (thread_y <= 0 || thread_y >= totheight - 1) {
                continue;
            }
            /*
                W-1 是 outlet ghost，
                自己不 collision。
            */
            if (thread_x >= totwidth - 1) {
                continue;
            }
            /*
                obstacle cell 本身不更新。
            */
            if (smask[smem_y][smem_x]) {
                continue;
            }
            float f[9];
            // --------------------------------------------
            // A gather: canonical local
            // --------------------------------------------
#pragma unroll
            for (int read = 0; read < 9; ++read) {
                f[read] = sregion[read][smem_y][smem_x];
            }
            float rho = 0.0f;
            float ux_loc = 0.0f;
            float uy_loc = 0.0f;
            // ====================================================
            // A-PHASE ZOU-HE
            // ====================================================
            if (thread_x == 0) {
                /*
                    左入口。

                    物理未知 incoming populations：

                        f1 : E
                        f5 : NE
                        f8 : SE

                    A phase 当前 f[] 是 canonical logical
                    populations，所以公式与普通 Zou-He
                    完全一样。
                */
                float dist_to_wall = fminf((float)thread_y, (float)(totheight - 1 - thread_y));
                float smooth_factor = 1.0f;
                if (dist_to_wall < 50.0f) {
                    smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
                }
                float local_u = u_in * smooth_factor;
                rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
                f[1] = f[3] + 0.666666667f * rho * local_u;
                f[5] = f[7] - 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
                f[8] = f[6] + 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
                /*
                    保持你旧 kernel 的入口语义：
                    不重新从 f[] 求速度，
                    而直接使用指定边界速度。
                */
                ux_loc = local_u;
                uy_loc = 0.0f;
            } else {
                // ----------------------------------------
                // ordinary fluid macro variables
                // ----------------------------------------
#pragma unroll
                for (int use = 0; use < 9; ++use) {
                    rho += f[use];
                    ux_loc += f[use] * biasx[use];
                    uy_loc += f[use] * biasy[use];
                }
                ux_loc /= rho;
                uy_loc /= rho;
            }
            // ====================================================
            // BGK
            // ====================================================
            float usq = ux_loc * ux_loc + uy_loc * uy_loc;
#pragma unroll
            for (int use = 0; use < 9; ++use) {
                float cu = biasx[use] * ux_loc + biasy[use] * uy_loc;
                float feq = w[use] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
                f[use] += tau_inv * (feq - f[use]);
            }
            // ====================================================
            // A store
            //
            // canonical -> reversed
            // ====================================================
#pragma unroll
            for (int store = 0; store < 9; ++store) {
                float out_val = fminf(10.0f, fmaxf(0.0f, f[store]));
                sregion[opp[store]][smem_y][smem_x] = out_val;
            }
        }
        /*
            所有 cell 的 A collision 必须全部完成，
            outlet 才能复制 W-2 的 reversed state。
        */
        __syncthreads();
        // ========================================================
        // A-PHASE OUTLET
        //
        // W-2 : reversed
        // W-1 : copied reversed ghost
        //
        // ghost[q] = fluid[q]
        // ========================================================
        int outlet_fluid_smem_x = (totwidth - 2) - (block_x - T);
        int outlet_ghost_smem_x = outlet_fluid_smem_x + 1;
        bool has_outlet = outlet_fluid_smem_x >= 0 && outlet_fluid_smem_x < SMEM_SIZE && outlet_ghost_smem_x >= 0 &&
                          outlet_ghost_smem_x < SMEM_SIZE;
        if (has_outlet) {
            /*
                A 的 reversed valid range：

                    [margin,
                     SMEM_SIZE-margin)

                B 对角 pull 需要其中的全部 rows，
                所以这里按 A valid range copy。
            */
            for (int line = block_tid; line < calculation_size; line += block_stride) {
                int smem_y = margin + line;
                int thread_y = block_y - T + smem_y;
                /*
                    保持旧 kernel：
                    y=0/H-1 outlet ghost 不更新。
                */
                if (thread_y <= 0 || thread_y >= totheight - 1) {
                    continue;
                }
#pragma unroll
                for (int j = 0; j < 9; ++j) {
                    sregion[j][smem_y][outlet_ghost_smem_x] = sregion[j][smem_y][outlet_fluid_smem_x];
                }
            }
        }
        /*
            B phase 读取 ghost 前，
            必须等 outlet copy 完。
        */
        __syncthreads();
        // ========================================================
        // B PHASE
        //
        // reversed
        //     ->
        // neighbor gather
        //     ->
        // collision
        //     ->
        // neighbor scatter
        //     ->
        // canonical
        //
        // gather normal:
        //
        //     sregion[opp[q]][self-c_q]
        //
        // scatter normal:
        //
        //     sregion[q][self+c_q]
        // ========================================================
        margin = 2 * times + 1;
        calculation_size = SMEM_SIZE - 2 * margin;
        calculation_area = calculation_size * calculation_size;
        for (int i = block_tid; i < calculation_area; i += block_stride) {
            int smem_x = margin + i % calculation_size;
            int smem_y = margin + i / calculation_size;
            int thread_x = block_x - T + smem_x;
            int thread_y = block_y - T + smem_y;
            if (thread_x < 0 || thread_x >= totwidth || thread_y < 0 || thread_y >= totheight) {
                continue;
            }
            if (thread_y <= 0 || thread_y >= totheight - 1) {
                continue;
            }
            /*
                outlet ghost 本身不 collision。
            */
            if (thread_x >= totwidth - 1) {
                continue;
            }
            if (smask[smem_y][smem_x]) {
                continue;
            }
            float f[9];
            // ====================================================
            // B gather
            // ====================================================
#pragma unroll
            for (int read = 0; read < 9; ++read) {
                int source_x = smem_x - biasx[read];
                int source_y = smem_y - biasy[read];
                /*
                    x=0 从外部来的三个方向：

                        q=1,5,8

                    source_x 会落到 x=-1。

                    这里的值只是 placeholder，
                    后面 Zou-He 会覆盖。

                    因为域外 smask=1，
                    当前会暂时走 bounce-back 分支。
                */
                if (smask[source_y][source_x]) {
                    /*
                        AA bounce-back:

                        after A:
                            postcollision f_opp(q)
                            位于 self.slot[q]

                        因此 wall 返回的 incoming q：
                    */
                    f[read] = sregion[read][smem_y][smem_x];
                } else {
                    /*
                        normal reversed AA pull
                    */
                    f[read] = sregion[opp[read]][source_y][source_x];
                }
            }
            float rho = 0.0f;
            float ux_loc = 0.0f;
            float uy_loc = 0.0f;
            // ====================================================
            // B-PHASE ZOU-HE
            // ====================================================
            if (thread_x == 0) {
                /*
                    到这里：

                    f0 f2 f3 f4 f6 f7
                    已经是正确 physical populations。

                    f1 f5 f8 来自 x=-1，
                    当前只是 placeholder。

                    直接覆盖即可。
                */
                float dist_to_wall = fminf((float)thread_y, (float)(totheight - 1 - thread_y));
                float smooth_factor = 1.0f;
                if (dist_to_wall < 50.0f) {
                    smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
                }
                float local_u = u_in * smooth_factor;
                rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
                f[1] = f[3] + 0.666666667f * rho * local_u;
                f[5] = f[7] - 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
                f[8] = f[6] + 0.5f * (f[2] - f[4]) + 0.16666666666f * rho * local_u;
                ux_loc = local_u;
                uy_loc = 0.0f;
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
            // ====================================================
            // BGK
            // ====================================================
            float usq = ux_loc * ux_loc + uy_loc * uy_loc;
#pragma unroll
            for (int use = 0; use < 9; ++use) {
                float cu = biasx[use] * ux_loc + biasy[use] * uy_loc;
                float feq = w[use] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
                f[use] += tau_inv * (feq - f[use]);
            }
            // ====================================================
            // B scatter
            // ====================================================
#pragma unroll
            for (int store = 0; store < 9; ++store) {
                int dest_x = smem_x + biasx[store];
                int dest_y = smem_y + biasy[store];
                float out_val = fminf(10.0f, fmaxf(0.0f, f[store]));
                /*
                    左入口：
                    q=3,6,7 会飞出 x=-1。

                    这不是墙，所以绝对不能 bounce。

                    让它写进 shared ghost 当垃圾桶即可。
                    下一步需要从左侧进入的 q=1,5,8
                    会重新被 Zou-He 构造。
                */
                bool left_open_boundary = thread_x == 0 && biasx[store] < 0;
                if (left_open_boundary) {
                    sregion[store][dest_y][dest_x] = out_val;
                } else if (smask[dest_y][dest_x]) {
                    /*
                        AA bounce-back:

                        outgoing q 撞 solid，
                        下一 canonical state 中
                        直接成为 self 的 opp(q)。
                    */
                    sregion[opp[store]][smem_y][smem_x] = out_val;
                } else {
                    /*
                        normal AA propagation.

                        注意这里就是 +bias，
                        正号没有写反。
                    */
                    sregion[store][dest_y][dest_x] = out_val;
                }
            }
        }
        /*
            先等所有 B scatter 完。
        */
        __syncthreads();
        // ========================================================
        // B-PHASE OUTLET
        //
        // B 完成后 central valid region 已经变回 canonical。
        //
        // 再次：
        //
        //     ghost[q] = fluid[q]
        //
        // 只是这一次复制的是 canonical representation。
        // ========================================================
        int next_margin = 2 * (times + 1);
        int next_calculation_size = SMEM_SIZE - 2 * next_margin;
        if (has_outlet) {
            /*
                B 后只有 next_margin 内的区域
                是完整 canonical valid。
            */
            for (int line = block_tid; line < next_calculation_size; line += block_stride) {
                int smem_y = next_margin + line;
                int thread_y = block_y - T + smem_y;
                if (thread_y <= 0 || thread_y >= totheight - 1) {
                    continue;
                }
#pragma unroll
                for (int j = 0; j < 9; ++j) {
                    sregion[j][smem_y][outlet_ghost_smem_x] = sregion[j][smem_y][outlet_fluid_smem_x];
                }
            }
        }
        /*
            outlet canonical ghost 准备完以后，
            才允许下一 A pair 开始。
        */
        __syncthreads();
    }
    // ============================================================
    // After T=6:
    //
    // central 16 x 16 is canonical and valid.
    //
    // shared:
    //
    //     x,y = [T, T+15]
    //           [6,21]
    //
    // global:
    //
    //     x = block_x ... block_x+15
    // ============================================================
    constexpr int CORE_AREA = SQUARE_A * SQUARE_A;
    for (int i = block_tid; i < CORE_AREA; i += block_stride) {
        int core_x = i % SQUARE_A;
        int core_y = i / SQUARE_A;
        int smem_x = T + core_x;
        int smem_y = T + core_y;
        int thread_x = block_x + core_x;
        int thread_y = block_y + core_y;
        if (thread_x < 0 || thread_x >= totwidth || thread_y < 0 || thread_y >= totheight) {
            continue;
        }
        int linear_idx = thread_y * totwidth + thread_x;
        /*
            T 是偶数，因此退出时一定 canonical。

            包括 W-1 ghost：
            最后一轮 B outlet copy 已经保证
            ghost == W-2。
        */
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            f_out[j * total_area + linear_idx] = sregion[j][smem_y][smem_x];
        }
        /*
            与旧 kernel 一样：
            outlet ghost 本身不输出 macro variables。

            wall / obstacle 也不输出。
        */
        if (!smask[smem_y][smem_x] && thread_x < totwidth - 1 && thread_y > 0 && thread_y < totheight - 1) {
            float rho = 0.0f;
            float ux_loc = 0.0f;
            float uy_loc = 0.0f;
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                float fj = sregion[j][smem_y][smem_x];
                rho += fj;
                ux_loc += fj * biasx[j];
                uy_loc += fj * biasy[j];
            }
            ux_loc /= rho;
            uy_loc /= rho;
            ux[linear_idx] = ux_loc;
            uy[linear_idx] = uy_loc;
            rho__[linear_idx] = rho;
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