#include <cublasLt.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw status;
    }
}

struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t output = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;

    ~Plan() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (output) cublasLtMatrixLayoutDestroy(output);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }
};

void layout(cublasLtMatrixLayout_t* descriptor, int64_t rows, int64_t columns, int32_t batches) {
    check(cublasLtMatrixLayoutCreate(descriptor, CUDA_R_32F, rows, columns, rows));
    int64_t stride = rows * columns;
    check(cublasLtMatrixLayoutSetAttribute(*descriptor, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                          &batches, sizeof(batches)));
    check(cublasLtMatrixLayoutSetAttribute(*descriptor, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &stride, sizeof(stride)));
}

uint32_t option(const cublasLtMatmulAlgo_t& algorithm, cublasLtMatmulAlgoConfigAttributes_t attribute) {
    uint32_t value = 0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, &value, sizeof(value), nullptr));
    return value;
}

bool fp32_fma(const cublasLtMatmulAlgo_t& algorithm) {
    uint64_t flags = 0;
    check(cublasLtMatmulAlgoCapGetAttribute(&algorithm, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                           &flags, sizeof(flags), nullptr));
    return (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK) == CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA
        && (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK) == CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F
        && (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) == CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F
        && option(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) <= 1
        && option(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) == CUBLASLT_REDUCTION_SCHEME_NONE;
}

void find_algorithms(Plan& plan, uint64_t workspace,
                     uint32_t align_a, uint32_t align_b, uint32_t align_c) {
    check(cublasLtMatmulPreferenceCreate(&plan.preference));
    check(cublasLtMatmulPreferenceSetAttribute(plan.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace, sizeof(workspace)));
    const std::array<cublasLtMatmulPreferenceAttributes_t, 4> attributes{
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
    };
    const std::array<uint32_t, 4> alignments{align_a, align_b, align_c, align_c};
    for (std::size_t i = 0; i < attributes.size(); ++i) {
        check(cublasLtMatmulPreferenceSetAttribute(plan.preference, attributes[i], &alignments[i],
                                                  sizeof(alignments[i])));
    }
    std::array<cublasLtMatmulHeuristicResult_t, 64> candidates{};
    int found = 0;
    check(cublasLtMatmulAlgoGetHeuristic(plan.handle, plan.operation, plan.a, plan.b,
                                       plan.output, plan.output, plan.preference,
                                       candidates.size(), candidates.data(), &found));
    for (int i = 0; i < found; ++i) {
        const auto& candidate = candidates[i];
        if (candidate.state == CUBLAS_STATUS_SUCCESS && fp32_fma(candidate.algo)) {
            plan.algorithms.push_back(candidate);
        }
    }
}

}  // namespace

extern "C" Plan* axial_lt_create(int32_t batch, int64_t m, int64_t k, int64_t n,
                                  uint64_t workspace, uint32_t align_a, uint32_t align_b,
                                  uint32_t align_c, int* status) {
    try {
        auto plan = std::make_unique<Plan>();
        check(cublasLtCreate(&plan->handle));
        check(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        // Interpret contiguous row-major operands as transposed column-major
        // matrices, in the same operand order used by ATen for contiguous C.
        layout(&plan->a, n, k, batch);
        layout(&plan->b, k, m, batch);
        layout(&plan->output, n, m, batch);
        find_algorithms(*plan, workspace, align_a, align_b, align_c);
        *status = CUBLAS_STATUS_SUCCESS;
        return plan.release();
    } catch (cublasStatus_t error) {
        *status = error;
    } catch (...) {
        *status = CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return nullptr;
}

extern "C" Plan* linear_lt_create(int64_t rows, int64_t input, int64_t output,
                                   int32_t packed_weight, const void* bias, int32_t relu,
                                   uint64_t workspace, int* status) {
    try {
        auto plan = std::make_unique<Plan>();
        check(cublasLtCreate(&plan->handle));
        check(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        // Column-major D is the original row-major Linear output transposed.
        // A is W, while B is the original row-major activation transposed.
        cublasOperation_t trans_a = packed_weight ? CUBLAS_OP_N : CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &trans_a, sizeof(trans_a)));
        if (bias || relu) {
            cublasLtEpilogue_t epilogue = bias
                ? (relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS)
                : CUBLASLT_EPILOGUE_RELU;
            check(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
        }
        if (bias) {
            check(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &bias, sizeof(bias)));
        }
        layout(&plan->a, packed_weight ? output : input, packed_weight ? input : output, 1);
        layout(&plan->b, input, rows, 1);
        layout(&plan->output, output, rows, 1);
        find_algorithms(*plan, workspace, 16, 16, 16);
        *status = CUBLAS_STATUS_SUCCESS;
        return plan.release();
    } catch (cublasStatus_t error) {
        *status = error;
    } catch (...) {
        *status = CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return nullptr;
}

extern "C" std::size_t axial_lt_version() {
    return cublasLtGetVersion();
}

extern "C" int linear_lt_set_bias(Plan* plan, const void* bias) {
    return cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                         &bias, sizeof(bias));
}

extern "C" int axial_lt_count(const Plan* plan) {
    return static_cast<int>(plan->algorithms.size());
}

extern "C" int axial_lt_info(const Plan* plan, int index, uint32_t* id, uint32_t* tile,
                             uint32_t* stages, uint64_t* workspace) {
    if (index < 0 || index >= axial_lt_count(plan)) return CUBLAS_STATUS_INVALID_VALUE;
    const auto& candidate = plan->algorithms[index];
    try {
        *id = option(candidate.algo, CUBLASLT_ALGO_CONFIG_ID);
        *tile = option(candidate.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
        *stages = option(candidate.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
        *workspace = candidate.workspaceSize;
        return CUBLAS_STATUS_SUCCESS;
    } catch (cublasStatus_t error) {
        return error;
    }
}

extern "C" int axial_lt_run(const Plan* plan, int index, const void* left, const void* right,
                            void* output, void* workspace, uint64_t size, cudaStream_t stream) {
    if (index < 0 || index >= axial_lt_count(plan)) return CUBLAS_STATUS_INVALID_VALUE;
    const auto& candidate = plan->algorithms[index];
    if (candidate.workspaceSize > size) return CUBLAS_STATUS_INVALID_VALUE;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasLtMatmul(plan->handle, plan->operation, &alpha, right, plan->a, left, plan->b,
                          &beta, output, plan->output, output, plan->output, &candidate.algo,
                          workspace, size, stream);
}

extern "C" void axial_lt_destroy(Plan* plan) {
    delete plan;
}
