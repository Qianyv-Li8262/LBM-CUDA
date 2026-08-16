__device__ const int biasx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__device__ const int biasy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__device__ const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
__device__ const float w[9] = {0.444444444f,  0.1111111111f, 0.1111111111f, 0.1111111111f, 0.1111111111f,
                               0.0277777778f, 0.0277777778f, 0.0277777778f, 0.0277777778f};

constexpr int T = 6;
constexpr int SQUARE_A = 24;
constexpr int SMEM_SIZE = SQUARE_A + 2 * T;
constexpr int AREA = SMEM_SIZE * SMEM_SIZE;

// Per-physical-timestep traversal policy.
// false: compact logical-square traversal (minimum predication, may cross physical rows)
// true : physical-SMEM-order traversal (bank-friendly, trades conflicts for predicated-off lanes)
// Stage == margin: 0:A0, 1:B, 2:A, ..., T-1:B-final.
static_assert(T == 6, "Update kPhysicalTraversal when changing T");
static constexpr bool kPhysicalTraversal[T] = {
    false, // stage 0, size 36: already bank-clean
    false,  // stage 1, size 34
    false, // stage 2, size 32: already bank-clean
    false,  // stage 3, size 30
    false,  // stage 4, size 28
    false   // stage 5, size 26 (B final)
};

template <int I> struct stage_tag {
    static constexpr int value = I;
};

template <int I, int End, class F> __device__ __forceinline__ void static_for(const F &f)
{
    if constexpr (I < End) {
        f(stage_tag<I>{});
        static_for<I + 1, End>(f);
    }
}

template <int Stage, class F>
__device__ __forceinline__ void traverse_stage(int block_tid, int block_stride, const F &body)
{
    static_assert(Stage >= 0 && Stage < T, "invalid temporal stage");

    constexpr int margin = Stage;
    constexpr int calculation_size = SMEM_SIZE - 2 * margin;
    constexpr int calculation_area = calculation_size * calculation_size;
    constexpr bool physical_order = kPhysicalTraversal[Stage];

    if constexpr (!physical_order) {
        // Compact logical-square order: no inactive holes, but a warp can cross
        // a logical row boundary that is not a physical SMEM row boundary.
        for (int i = block_tid; i < calculation_area; i += block_stride) {
            const int smem_x = margin + i % calculation_size;
            const int smem_y = margin + i / calculation_size;
            body(smem_x, smem_y);
        }
    } else {
        // Walk the smallest contiguous physical-address span covering the active square.
        // Consecutive lanes therefore touch consecutive float words in sregion.
        constexpr int first = margin * SMEM_SIZE + margin;
        constexpr int span = (calculation_size - 1) * SMEM_SIZE + calculation_size;
        constexpr int x_begin = margin;
        constexpr int x_end = margin + calculation_size;

        for (int i = block_tid; i < span; i += block_stride) {
            const int addr = first + i;
            const int smem_x = addr % SMEM_SIZE;
            const int smem_y = addr / SMEM_SIZE;

            // Physical-row holes are the price paid for bank-clean SIMD scheduling.
            if (smem_x < x_begin || smem_x >= x_end)
                continue;

            body(smem_x, smem_y);
        }
    }
}

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
    auto a0_body = [&](int smem_x, int smem_y) {
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
            return;

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
    };

    traverse_stage<0>(block_tid, block_stride, a0_body);

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
    // Stage number == shrinking margin. static_for makes Stage a compile-time value,
    // so kPhysicalTraversal[Stage] becomes an if-constexpr policy with no runtime branch.
    // ========================================================================
    auto b_middle_body = [&](int smem_x, int smem_y) {
        const int thread_x = block_x - T + smem_x;
        const int thread_y = block_y - T + smem_y;

        if (thread_x < 0 || thread_x >= totwidth || thread_y <= 0 || thread_y >= totheight - 1)
            return;
        if (thread_x >= totwidth - 1 || smask[smem_y][smem_x])
            return;

        float f[9];
#pragma unroll
        for (int read = 0; read < 9; ++read) {
            const int source_x = smem_x - biasx[read];
            const int source_y = smem_y - biasy[read];
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

        const float usq = ux_loc * ux_loc + uy_loc * uy_loc;
#pragma unroll
        for (int use = 0; use < 9; ++use) {
            float cu = biasx[use] * ux_loc + biasy[use] * uy_loc;
            float feq = w[use] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
            f[use] += tau_inv * (feq - f[use]);
        }

#pragma unroll
        for (int store = 0; store < 9; ++store) {
            const int dest_x = smem_x + biasx[store];
            const int dest_y = smem_y + biasy[store];
            const float out_val = fminf(10.0f, fmaxf(0.0f, f[store]));
            const bool left_open_boundary = thread_x == 0 && biasx[store] < 0;
            if (left_open_boundary)
                sregion[store][dest_y][dest_x] = out_val;
            else if (smask[dest_y][dest_x])
                sregion[opp[store]][smem_y][smem_x] = out_val;
            else
                sregion[store][dest_y][dest_x] = out_val;
        }
    };

    auto a_middle_body = [&](int smem_x, int smem_y) {
        const int thread_x = block_x - T + smem_x;
        const int thread_y = block_y - T + smem_y;

        if (thread_x < 0 || thread_x >= totwidth || thread_y <= 0 || thread_y >= totheight - 1)
            return;
        if (thread_x >= totwidth - 1 || smask[smem_y][smem_x])
            return;

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
    };

    auto run_middle_stage = [&](auto tag) {
        constexpr int Stage = decltype(tag)::value;
        static_assert(Stage > 0 && Stage < T - 1, "middle stage out of range");

        if constexpr (Stage & 1) {
            // B: reversed -> canonical
            traverse_stage<Stage>(block_tid, block_stride, b_middle_body);
            __syncthreads();
        } else {
            // A: canonical -> reversed
            traverse_stage<Stage>(block_tid, block_stride, a_middle_body);
            __syncthreads();

            // A 后生成下一 B 所需的 reversed outlet ghost。
            if (has_outlet) {
                constexpr int margin = Stage;
                constexpr int calculation_size = SMEM_SIZE - 2 * margin;
                for (int line = block_tid; line < calculation_size; line += block_stride) {
                    const int smem_y = margin + line;
                    const int thread_y = block_y - T + smem_y;
                    if (thread_y <= 0 || thread_y >= totheight - 1)
                        continue;
#pragma unroll
                    for (int j = 0; j < 9; ++j)
                        sregion[j][smem_y][outlet_ghost_smem_x] = sregion[j][smem_y][outlet_fluid_smem_x];
                }
            }
            __syncthreads();
        }
    };

    // Stages 1 ... T-2 are instantiated independently at compile time.
    static_for<1, T - 1>(run_middle_stage);

    // ========================================================================
    // B FINAL SPECIALIZATION
    // reversed SMEM -> registers -> BGK -> directly scatter to f_out.
    // 不再生成 final canonical sregion，也没有最后一次 shared store/read。
    // ========================================================================
    auto b_final_body = [&](int smem_x, int smem_y) {
        const int thread_x = block_x - T + smem_x;
        const int thread_y = block_y - T + smem_y;

        if (thread_x < 0 || thread_x >= totwidth || thread_y <= 0 || thread_y >= totheight - 1)
            return;
        if (thread_x >= totwidth - 1 || smask[smem_y][smem_x])
            return;

        float f[9];
#pragma unroll
        for (int read = 0; read < 9; ++read) {
            const int source_x = smem_x - biasx[read];
            const int source_y = smem_y - biasy[read];
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

        const float usq = ux_loc * ux_loc + uy_loc * uy_loc;
#pragma unroll
        for (int use = 0; use < 9; ++use) {
            float cu = biasx[use] * ux_loc + biasy[use] * uy_loc;
            float feq = w[use] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
            f[use] += tau_inv * (feq - f[use]);
        }

#pragma unroll
        for (int store = 0; store < 9; ++store) {
            const int dest_smem_x = smem_x + biasx[store];
            const int dest_smem_y = smem_y + biasy[store];
            int target_x, target_y, target_q;
            const float out_val = fminf(10.0f, fmaxf(0.0f, f[store]));

            // x=0 向域外飞出的 population 直接丢弃；下一 A0 的 incoming 由 Zou-He 重建。
            if (thread_x == 0 && biasx[store] < 0)
                continue;

            if (smask[dest_smem_y][dest_smem_x]) {
                target_x = thread_x;
                target_y = thread_y;
                target_q = opp[store];
            } else {
                target_x = thread_x + biasx[store];
                target_y = thread_y + biasy[store];
                target_q = store;
            }

            if (target_x == totwidth - 1)
                continue;

            const bool target_in_core = target_x >= block_x && target_x < block_x + SQUARE_A && target_y >= block_y &&
                                        target_y < block_y + SQUARE_A;
            if (!target_in_core || target_x < 0 || target_x >= totwidth || target_y < 0 || target_y >= totheight)
                continue;

            const int target_idx = target_y * totwidth + target_x;
            f_out[target_q * total_area + target_idx] = out_val;

            if (target_x == totwidth - 2)
                f_out[target_q * total_area + target_y * totwidth + (totwidth - 1)] = out_val;
        }
    };

    traverse_stage<T - 1>(block_tid, block_stride, b_final_body);
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
