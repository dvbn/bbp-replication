#include <Python.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

#if defined(PYIOLS_HAVE_SCIPY_OPENBLAS64)
using PyiolsLapackInt = std::int64_t;
extern "C" {
void scipy_dgelsd_64_(
    const PyiolsLapackInt* m,
    const PyiolsLapackInt* n,
    const PyiolsLapackInt* nrhs,
    double* a,
    const PyiolsLapackInt* lda,
    double* b,
    const PyiolsLapackInt* ldb,
    double* s,
    const double* rcond,
    PyiolsLapackInt* rank,
    double* work,
    const PyiolsLapackInt* lwork,
    PyiolsLapackInt* iwork,
    PyiolsLapackInt* info);
}
#define PYIOLS_DGELSD scipy_dgelsd_64_
#elif defined(PYIOLS_HAVE_LAPACK)
using PyiolsLapackInt = int;
extern "C" {
void dgelsd_(
    const PyiolsLapackInt* m,
    const PyiolsLapackInt* n,
    const PyiolsLapackInt* nrhs,
    double* a,
    const PyiolsLapackInt* lda,
    double* b,
    const PyiolsLapackInt* ldb,
    double* s,
    const double* rcond,
    PyiolsLapackInt* rank,
    double* work,
    const PyiolsLapackInt* lwork,
    PyiolsLapackInt* iwork,
    PyiolsLapackInt* info);
}
#define PYIOLS_DGELSD dgelsd_
#endif

namespace pyiols::kernel {

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

double stable_l2_norm(
    const Eigen::Ref<const Eigen::VectorXd>& values) {
    double norm = 0.0;
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        norm = std::hypot(norm, std::abs(values(idx)));
    }
    return norm;
}

Eigen::MatrixXd min_norm_lstsq_eigen(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        Eigen::MatrixXd(a),
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    const double max_sv = singular.size() == 0 ? 0.0 : singular.maxCoeff();
    const double cutoff =
        std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(a.rows(), a.cols())) * max_sv;
    Eigen::MatrixXd projected = svd.matrixU().transpose() * b;
    for (Eigen::Index row = 0; row < projected.rows(); ++row) {
        const double sv = singular(row);
        if (sv > cutoff) {
            projected.row(row) /= sv;
        } else {
            projected.row(row).setZero();
        }
    }
    return svd.matrixV() * projected;
}

Eigen::MatrixXd min_norm_lstsq(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b) {
    if (a.rows() == 0 || a.cols() == 0) {
        throw std::invalid_argument("least-squares matrix must be non-empty");
    }
    if (a.rows() != b.rows()) {
        throw std::invalid_argument("least-squares row dimensions do not match");
    }
#if defined(PYIOLS_HAVE_SCIPY_OPENBLAS64) || defined(PYIOLS_HAVE_LAPACK)
    if (a.rows() >
            static_cast<Eigen::Index>(std::numeric_limits<PyiolsLapackInt>::max()) ||
        a.cols() >
            static_cast<Eigen::Index>(std::numeric_limits<PyiolsLapackInt>::max()) ||
        b.cols() >
            static_cast<Eigen::Index>(std::numeric_limits<PyiolsLapackInt>::max())) {
        throw std::overflow_error("least-squares dimensions exceed LAPACK int range");
    }
    const PyiolsLapackInt m = static_cast<PyiolsLapackInt>(a.rows());
    const PyiolsLapackInt n = static_cast<PyiolsLapackInt>(a.cols());
    const PyiolsLapackInt nrhs = static_cast<PyiolsLapackInt>(b.cols());
    const PyiolsLapackInt lda = std::max<PyiolsLapackInt>(1, m);
    const PyiolsLapackInt ldb = std::max<PyiolsLapackInt>(1, std::max(m, n));
    const PyiolsLapackInt minmn = std::min(m, n);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> a_lapack =
        a;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> b_lapack =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(
            ldb,
            nrhs);
    b_lapack.topRows(m) = b;
    Eigen::VectorXd singular(minmn);

    const double rcond =
        std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(a.rows(), a.cols()));
    PyiolsLapackInt rank = 0;
    PyiolsLapackInt info = 0;

    const PyiolsLapackInt smlsiz = 25;
    PyiolsLapackInt nlvl = 0;
    if (minmn > smlsiz + 1) {
        nlvl = static_cast<PyiolsLapackInt>(
            std::log(static_cast<double>(minmn) / static_cast<double>(smlsiz + 1)) /
            std::log(2.0)) +
               1;
    }
    const PyiolsLapackInt liwork =
        std::max<PyiolsLapackInt>(
            1,
            3 * minmn * std::max<PyiolsLapackInt>(0, nlvl) + 11 * minmn);
    std::vector<PyiolsLapackInt> iwork(static_cast<std::size_t>(liwork), 0);

    double work_query = 0.0;
    PyiolsLapackInt lwork = -1;
    PYIOLS_DGELSD(
        &m,
        &n,
        &nrhs,
        a_lapack.data(),
        &lda,
        b_lapack.data(),
        &ldb,
        singular.data(),
        &rcond,
        &rank,
        &work_query,
        &lwork,
        iwork.data(),
        &info);
    if (info != 0 || !std::isfinite(work_query) || work_query <= 0.0) {
        throw std::runtime_error("LAPACK DGELSD workspace query failed");
    }
    lwork = static_cast<PyiolsLapackInt>(std::ceil(work_query));
    std::vector<double> work(
        static_cast<std::size_t>(std::max<PyiolsLapackInt>(1, lwork)),
        0.0);
    PYIOLS_DGELSD(
        &m,
        &n,
        &nrhs,
        a_lapack.data(),
        &lda,
        b_lapack.data(),
        &ldb,
        singular.data(),
        &rcond,
        &rank,
        work.data(),
        &lwork,
        iwork.data(),
        &info);
    if (info != 0) {
        throw std::runtime_error("LAPACK DGELSD least-squares solve failed");
    }
    return Eigen::MatrixXd(b_lapack.topRows(n));
#else
    return min_norm_lstsq_eigen(a, b);
#endif
}

Eigen::MatrixXd min_norm_lstsq_anderson(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b) {
    return min_norm_lstsq(a, b);
}

std::optional<Eigen::VectorXd> lstsq_full_rank_qr_fast_path(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    Eigen::Index min_cols = 8) {
    if (a.rows() != b.rows() || a.rows() < a.cols() ||
        a.cols() < std::max<Eigen::Index>(min_cols, 1)) {
        return std::nullopt;
    }
    if (!a.allFinite() || !b.allFinite()) {
        return std::nullopt;
    }
    const Eigen::MatrixXd a_owned(a);
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a_owned);
    if (qr.info() != Eigen::Success || qr.rank() != a.cols()) {
        return std::nullopt;
    }
    const double max_pivot = qr.maxPivot();
    if (!std::isfinite(max_pivot) || max_pivot <= 0.0) {
        return std::nullopt;
    }
    double min_pivot = std::numeric_limits<double>::infinity();
    const Eigen::Index pivot_count = std::min(a.rows(), a.cols());
    for (Eigen::Index idx = 0; idx < pivot_count; ++idx) {
        const double pivot = std::abs(qr.matrixR()(idx, idx));
        if (!std::isfinite(pivot)) {
            return std::nullopt;
        }
        min_pivot = std::min(min_pivot, pivot);
    }
    if (!std::isfinite(min_pivot) || min_pivot <= max_pivot * 1.0e-10) {
        return std::nullopt;
    }
    Eigen::VectorXd beta = qr.solve(b);
    if (!beta.allFinite()) {
        return std::nullopt;
    }
    return beta;
}

std::optional<Eigen::VectorXd> solve_well_conditioned_spd_gram_fast_path(
    const Eigen::Ref<const Eigen::MatrixXd>& gram,
    const Eigen::Ref<const Eigen::VectorXd>& rhs,
    Eigen::Index min_cols);

std::optional<Eigen::VectorXd> lstsq_well_conditioned_spd_fast_path(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::VectorXd>& b) {
    if (a.rows() != b.rows() || a.rows() < a.cols() || a.cols() < 32) {
        return std::nullopt;
    }
    if (!a.allFinite() || !b.allFinite()) {
        return std::nullopt;
    }
    const Eigen::MatrixXd gram = a.transpose() * a;
    const Eigen::VectorXd rhs = a.transpose() * b;
    if (!gram.allFinite() || !rhs.allFinite()) {
        return std::nullopt;
    }
    return solve_well_conditioned_spd_gram_fast_path(gram, rhs, 32);
}

std::optional<Eigen::VectorXd> solve_well_conditioned_spd_gram_fast_path(
    const Eigen::Ref<const Eigen::MatrixXd>& gram,
    const Eigen::Ref<const Eigen::VectorXd>& rhs,
    Eigen::Index min_cols) {
    if (gram.rows() != gram.cols() || gram.rows() != rhs.size() ||
        gram.cols() < min_cols) {
        return std::nullopt;
    }
    if (!gram.allFinite() || !rhs.allFinite()) {
        return std::nullopt;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        gram,
        Eigen::EigenvaluesOnly);
    if (spectrum.info() != Eigen::Success) {
        return std::nullopt;
    }
    const Eigen::VectorXd eigenvalues = spectrum.eigenvalues();
    if (eigenvalues.size() == 0) {
        return std::nullopt;
    }
    const double max_eigen = eigenvalues.maxCoeff();
    const double min_eigen = eigenvalues.minCoeff();
    if (!std::isfinite(max_eigen) || !std::isfinite(min_eigen) ||
        max_eigen <= 0.0 || min_eigen <= max_eigen * 1.0e-8) {
        return std::nullopt;
    }
    Eigen::LLT<Eigen::MatrixXd> llt(gram);
    if (llt.info() != Eigen::Success) {
        return std::nullopt;
    }
    Eigen::VectorXd beta = llt.solve(rhs);
    if (!beta.allFinite()) {
        return std::nullopt;
    }
    const double rhs_scale = std::max(stable_l2_norm(rhs), 1.0);
    const Eigen::VectorXd normal_residual_vec = gram * beta - rhs;
    const double normal_residual =
        stable_l2_norm(normal_residual_vec) / rhs_scale;
    if (!std::isfinite(normal_residual) || normal_residual > 1.0e-8) {
        return std::nullopt;
    }
    return beta;
}

}  // namespace pyiols::kernel

namespace {

using pyiols::kernel::RowMatrixXd;

constexpr double kFloat64Max = std::numeric_limits<double>::max();
const double kFloat64LogMax = std::log(kFloat64Max);

Eigen::VectorXd vector_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> array,
    const char* name) {
    const py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::invalid_argument(std::string(name) + " must be one-dimensional");
    }
    Eigen::VectorXd out(info.shape[0]);
    const auto* ptr = static_cast<const double*>(info.ptr);
    if (info.shape[0] > 0) {
        std::memcpy(
            out.data(),
            ptr,
            static_cast<std::size_t>(info.shape[0]) * sizeof(double));
    }
    return out;
}

RowMatrixXd matrix_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> array,
    const char* name) {
    const py::buffer_info info = array.request();
    if (info.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be two-dimensional");
    }
    RowMatrixXd out(info.shape[0], info.shape[1]);
    const auto* ptr = static_cast<const double*>(info.ptr);
    if (info.shape[0] > 0 && info.shape[1] > 0) {
        std::memcpy(
            out.data(),
            ptr,
            static_cast<std::size_t>(info.shape[0] * info.shape[1]) *
                sizeof(double));
    }
    return out;
}

std::vector<std::int64_t> groups_from_array(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    Eigen::Index expected_rows,
    std::int64_t n_groups) {
    const py::buffer_info info = groups.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("groups must be one-dimensional");
    }
    if (info.shape[0] != expected_rows) {
        throw std::invalid_argument("values and groups must have the same rows");
    }
    const auto* ptr = static_cast<const std::int64_t*>(info.ptr);
    std::vector<std::int64_t> out(static_cast<std::size_t>(info.shape[0]));
    for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
        if (ptr[row] < 0 || ptr[row] >= n_groups) {
            throw std::invalid_argument("group labels must be in [0, n_groups)");
        }
        out[static_cast<std::size_t>(row)] = ptr[row];
    }
    return out;
}

py::array_t<double> vector_to_array(const Eigen::Ref<const Eigen::VectorXd>& vec) {
    py::array_t<double> out({vec.size()});
    if (vec.size() > 0) {
        std::memcpy(
            out.mutable_data(),
            vec.data(),
            static_cast<std::size_t>(vec.size()) * sizeof(double));
    }
    return out;
}

py::array_t<double> matrix_to_array(const Eigen::Ref<const Eigen::MatrixXd>& mat) {
    py::array_t<double> out({mat.rows(), mat.cols()});
    auto unchecked = out.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < unchecked.shape(0); ++row) {
        for (py::ssize_t col = 0; col < unchecked.shape(1); ++col) {
            unchecked(row, col) = mat(row, col);
        }
    }
    return out;
}

py::object matrix_or_vector_to_array(
    const Eigen::Ref<const Eigen::MatrixXd>& mat,
    bool as_vector) {
    if (!as_vector) {
        return matrix_to_array(mat);
    }
    return vector_to_array(mat.col(0));
}

Eigen::MatrixXd matrix_from_values(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    bool& was_vector) {
    const py::buffer_info info = values.request();
    if (info.ndim == 1) {
        was_vector = true;
        Eigen::MatrixXd out(info.shape[0], 1);
        const auto* ptr = static_cast<const double*>(info.ptr);
        for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
            out(row, 0) = ptr[row];
        }
        return out;
    }
    if (info.ndim == 2) {
        was_vector = false;
        return matrix_from_array(values, "values");
    }
    throw std::invalid_argument("values must be one- or two-dimensional");
}

struct DemeanWorkspaceCore {
    Eigen::MatrixXd scratch;
    Eigen::MatrixXd previous;
    Eigen::MatrixXd sums;
    Eigen::VectorXd denom;
};

struct DemeanVectorWorkspaceCore {
    Eigen::VectorXd scratch;
    Eigen::VectorXd previous;
    Eigen::VectorXd sums;
    Eigen::VectorXd denom;
    std::vector<Eigen::VectorXd> bundle_sums;
    std::vector<Eigen::VectorXd> bundle_denom;
    int last_sweep_iterations = 0;
};

struct GroupBundleCore {
    std::vector<std::vector<std::int64_t>> groups;
    std::vector<std::int64_t> counts;
    std::vector<Eigen::VectorXd> unweighted_inverse_counts;
};

GroupBundleCore group_bundle_from_objects(
    py::object groups_obj,
    py::object n_groups_obj,
    Eigen::Index expected_rows,
    const char* empty_group_message);

void demean_for_group_bundle_into_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const GroupBundleCore& bundle,
    const Eigen::VectorXd* weights,
    double tol,
    int maxiter,
    Eigen::MatrixXd& out,
    DemeanWorkspaceCore& workspace,
    bool* converged = nullptr);

void demean_oneway_matrix_into_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const Eigen::VectorXd* weights,
    Eigen::MatrixXd& out,
    DemeanWorkspaceCore& workspace) {
    workspace.sums.resize(static_cast<Eigen::Index>(n_groups), values.cols());
    workspace.sums.setZero();
    workspace.denom.resize(static_cast<Eigen::Index>(n_groups));
    workspace.denom.setZero();
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        const auto group = static_cast<Eigen::Index>(
            groups[static_cast<std::size_t>(row)]);
        const double weight = weights != nullptr ? (*weights)(row) : 1.0;
        workspace.sums.row(group) += weight * values.row(row);
        workspace.denom(group) += weight;
    }

    for (Eigen::Index group = 0; group < workspace.sums.rows(); ++group) {
        if (weights != nullptr) {
            workspace.sums.row(group) /=
                std::max(workspace.denom(group), 1.0e-300);
        } else {
            const double divisor = std::max(workspace.denom(group), 1.0);
            workspace.sums.row(group) /= divisor;
        }
    }

    out.resize(values.rows(), values.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        out.row(row) = values.row(row) -
                       workspace.sums.row(groups[static_cast<std::size_t>(row)]);
    }
}

void demean_oneway_vector_unweighted_with_buffers_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    Eigen::VectorXd& out,
    Eigen::VectorXd& sums,
    Eigen::VectorXd& denom,
    const Eigen::VectorXd* inverse_counts = nullptr) {
    sums.resize(static_cast<Eigen::Index>(n_groups));
    sums.setZero();
    if (inverse_counts != nullptr &&
        inverse_counts->size() == static_cast<Eigen::Index>(n_groups)) {
        for (Eigen::Index row = 0; row < values.size(); ++row) {
            const auto group = static_cast<Eigen::Index>(
                groups[static_cast<std::size_t>(row)]);
            sums(group) += values(row);
        }
        for (Eigen::Index group = 0; group < sums.size(); ++group) {
            sums(group) *= (*inverse_counts)(group);
        }
    } else {
        denom.resize(static_cast<Eigen::Index>(n_groups));
        denom.setZero();
        for (Eigen::Index row = 0; row < values.size(); ++row) {
            const auto group = static_cast<Eigen::Index>(
                groups[static_cast<std::size_t>(row)]);
            sums(group) += values(row);
            denom(group) += 1.0;
        }

        for (Eigen::Index group = 0; group < sums.size(); ++group) {
            const double divisor = std::max(denom(group), 1.0);
            if (divisor > 1e-15) {
                sums(group) /= divisor;
            } else {
                sums(group) = 0.0;
            }
        }
    }

    out.resize(values.size());
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        out(row) =
            values(row) - sums(groups[static_cast<std::size_t>(row)]);
    }
}

void demean_oneway_vector_unweighted_into_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    Eigen::VectorXd& out,
    DemeanVectorWorkspaceCore& workspace,
    const Eigen::VectorXd* inverse_counts = nullptr) {
    demean_oneway_vector_unweighted_with_buffers_core(
        values,
        groups,
        n_groups,
        out,
        workspace.sums,
        workspace.denom,
        inverse_counts);
}

void demean_vector_for_group_bundle_unweighted_into_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const GroupBundleCore& bundle,
    double tol,
    int maxiter,
    Eigen::VectorXd& out,
    DemeanVectorWorkspaceCore& workspace,
    bool* converged = nullptr);
Eigen::VectorXd demean_vector_for_group_bundle_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const GroupBundleCore& bundle,
    const std::optional<Eigen::VectorXd>& weights,
    double tol,
    int maxiter);

Eigen::MatrixXd demean_oneway_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const std::optional<Eigen::VectorXd>& weights) {
    Eigen::MatrixXd out;
    DemeanWorkspaceCore workspace;
    const Eigen::VectorXd* weights_ptr = weights.has_value() ? &(*weights) : nullptr;
    demean_oneway_matrix_into_core(
        values,
        groups,
        n_groups,
        weights_ptr,
        out,
        workspace);
    return out;
}

py::array_t<double> exp_guard(
    py::array_t<double, py::array::c_style | py::array::forcecast> values) {
    const py::buffer_info info = values.request();
    py::array_t<double> out(info.shape);
    const auto* src = static_cast<const double*>(info.ptr);
    auto* dst = static_cast<double*>(out.mutable_data());
    for (py::ssize_t idx = 0; idx < info.size; ++idx) {
        const double value = src[idx];
        if (std::isinf(value) && value > 0.0) {
            dst[idx] = kFloat64Max;
        } else if (value > kFloat64LogMax) {
            dst[idx] = kFloat64Max;
        } else {
            dst[idx] = std::exp(value);
        }
    }
    return out;
}

py::array_t<double> group_sum(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    std::int64_t n_groups) {
    const Eigen::VectorXd values_vec = vector_from_array(values, "values");
    const auto group_vec = groups_from_array(groups, values_vec.size(), n_groups);
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < values_vec.size(); ++row) {
        out(group_vec[row]) += values_vec(row);
    }
    return vector_to_array(out);
}

py::array_t<double> group_logsumexp(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    std::int64_t n_groups) {
    const Eigen::VectorXd values_vec = vector_from_array(values, "values");
    const auto group_vec = groups_from_array(groups, values_vec.size(), n_groups);
    Eigen::VectorXd max_by_group =
        Eigen::VectorXd::Constant(n_groups, -std::numeric_limits<double>::infinity());
    std::vector<bool> has_nan(static_cast<std::size_t>(n_groups), false);
    for (Eigen::Index row = 0; row < values_vec.size(); ++row) {
        const auto group = static_cast<Eigen::Index>(group_vec[row]);
        const double value = values_vec(row);
        if (std::isnan(value)) {
            has_nan[static_cast<std::size_t>(group)] = true;
            continue;
        }
        if (value > max_by_group(group)) {
            max_by_group(group) = value;
        }
    }

    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_groups, -std::numeric_limits<double>::infinity());
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < values_vec.size(); ++row) {
        const auto group = static_cast<Eigen::Index>(group_vec[row]);
        const double max_value = max_by_group(group);
        const double value = values_vec(row);
        if (has_nan[static_cast<std::size_t>(group)] || !std::isfinite(max_value)) {
            continue;
        }
        if (std::isfinite(value)) {
            sums(group) += std::exp(value - max_value);
        }
    }
    for (Eigen::Index group = 0; group < n_groups; ++group) {
        if (has_nan[static_cast<std::size_t>(group)]) {
            continue;
        }
        if (std::isinf(max_by_group(group)) && max_by_group(group) > 0.0) {
            out(group) = std::numeric_limits<double>::infinity();
        } else if (std::isfinite(max_by_group(group)) && sums(group) > 0.0) {
            out(group) = max_by_group(group) + std::log(sums(group));
        }
    }
    return vector_to_array(out);
}

py::object demean_oneway(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    std::int64_t n_groups,
    py::object weights_obj) {
    bool was_vector = false;
    const Eigen::MatrixXd values_mat = matrix_from_values(values, was_vector);
    const auto group_vec = groups_from_array(groups, values_mat.rows(), n_groups);
    std::optional<Eigen::VectorXd> weights;
    if (!weights_obj.is_none()) {
        weights = vector_from_array(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                weights_obj),
            "weights");
        if (weights->size() != values_mat.rows()) {
            throw std::invalid_argument("weights and groups must have the same shape");
        }
    }
    const Eigen::MatrixXd demeaned =
        demean_oneway_matrix(values_mat, group_vec, n_groups, weights);
    return matrix_or_vector_to_array(demeaned, was_vector);
}

py::dict demean_oneway_with_instruments(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    std::int64_t n_groups,
    py::object instruments_obj) {
    bool values_was_vector = false;
    const Eigen::MatrixXd values_mat = matrix_from_values(values, values_was_vector);
    const auto group_vec = groups_from_array(groups, values_mat.rows(), n_groups);
    const Eigen::MatrixXd demeaned_values =
        demean_oneway_matrix(values_mat, group_vec, n_groups, std::nullopt);
    py::dict out;
    out["values"] = matrix_or_vector_to_array(demeaned_values, values_was_vector);
    if (instruments_obj.is_none()) {
        out["instruments"] = py::none();
        return out;
    }
    bool instruments_was_vector = false;
    const Eigen::MatrixXd instruments = matrix_from_values(
        py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
            instruments_obj),
        instruments_was_vector);
    if (instruments.rows() != values_mat.rows()) {
        throw std::invalid_argument(
            "instruments and values must have the same number of rows");
    }
    const Eigen::MatrixXd demeaned_instruments =
        demean_oneway_matrix(instruments, group_vec, n_groups, std::nullopt);
    out["instruments"] =
        matrix_or_vector_to_array(demeaned_instruments, instruments_was_vector);
    return out;
}

py::object lstsq_min_norm(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b) {
    const Eigen::MatrixXd a_mat = matrix_from_array(a, "a");
    bool b_was_vector = false;
    const Eigen::MatrixXd b_mat = matrix_from_values(b, b_was_vector);
    const Eigen::MatrixXd solution = pyiols::kernel::min_norm_lstsq(a_mat, b_mat);
    return matrix_or_vector_to_array(solution, b_was_vector);
}

py::object anderson_lstsq(
    py::array_t<double, py::array::c_style | py::array::forcecast> f_mat,
    py::array_t<double, py::array::c_style | py::array::forcecast> residual) {
    const Eigen::MatrixXd f_mat_core = matrix_from_array(f_mat, "f_mat");
    bool residual_was_vector = false;
    const Eigen::MatrixXd residual_core =
        matrix_from_values(residual, residual_was_vector);
    const Eigen::MatrixXd solution =
        pyiols::kernel::min_norm_lstsq_anderson(f_mat_core, residual_core);
    return matrix_or_vector_to_array(solution, residual_was_vector);
}

bool cpp_env_flag_enabled_core(const char* name);

struct I2SLSControlCore {
    int max_iter = 1200;
    double tol = 5.0e-8;
    double phase2_rho = 1.0;
    double damping = 1.0;
    double rank_tol = std::numeric_limits<double>::quiet_NaN();
    int z_full_rank = -1;
    int first_stage_rank = -1;
    int x_breve_rank = -1;
    int anderson_m = 5;
    double delta_log_min = -5.0;
    double delta_log_max = 5.0;
    int delta_grid_size = 5;
    double rho_log_min = -7.0;
    double rho_log_max = 20.0;
    int rho_grid_size = 30;
    int tau_path_steps = 8;
    bool has_min_dtau = false;
    double min_dtau = 0.0;
    int max_path_evaluations = 200;
    double speed_i2sls_phase1_tol = std::numeric_limits<double>::quiet_NaN();
    bool speed_i2sls_combined_path_state = false;
    bool speed_i2sls_path_anderson = false;
    bool speed_i2sls_inplace_recenter = false;
    bool speed_trust_full_rank = false;
    bool speed_i2sls_small_lstsq = false;
    bool speed_i2sls_gram_lstsq = false;
    bool speed_i2sls_small_qr_lstsq = false;
    bool speed_i2sls_compact_diagnostics = false;
    bool speed_i2sls_vector_pseudo = false;
    bool speed_i2sls_omit_predictions = false;
};

struct EigenVectorPairRingCore {
    std::vector<Eigen::VectorXd> x;
    std::vector<Eigen::VectorXd> f;
    std::size_t capacity = 0;
    std::size_t start = 0;
    std::size_t count = 0;

    // Callers consume references immediately; do not hold x_at/f_at/f_back
    // references across another push because a full ring overwrites slots.
    void reset(std::size_t new_capacity) {
        capacity = new_capacity;
        start = 0;
        count = 0;
        x.clear();
        f.clear();
        x.reserve(capacity);
        f.reserve(capacity);
    }

    void clear() {
        start = 0;
        count = 0;
        x.clear();
        f.clear();
    }

    // runtime_v2 (direct fixed-tau route only): enlarge the ring in place when
    // the Anderson memory adapts upward, keeping the logical order of the
    // retained pairs. The NumPy reference trims its history lists to the
    // adapted aa_m + 1, so after adaptation it uses up to aa_m lags; a fixed
    // ring silently kept the initial memory. No-op unless new_capacity is larger.
    void grow(std::size_t new_capacity) {
        if (new_capacity <= capacity) {
            return;
        }
        std::vector<Eigen::VectorXd> x_ordered;
        std::vector<Eigen::VectorXd> f_ordered;
        x_ordered.reserve(new_capacity);
        f_ordered.reserve(new_capacity);
        for (std::size_t logical = 0; logical < count; ++logical) {
            x_ordered.emplace_back(x_at(logical));
            f_ordered.emplace_back(f_at(logical));
        }
        x = std::move(x_ordered);
        f = std::move(f_ordered);
        capacity = new_capacity;
        start = 0;
    }

    void push(const Eigen::VectorXd& x_value, const Eigen::VectorXd& f_value) {
        if (capacity == 0) {
            return;
        }
        if (count < capacity) {
            x.emplace_back(x_value);
            f.emplace_back(f_value);
            ++count;
            return;
        }
        x[start] = x_value;
        f[start] = f_value;
        start = (start + 1) % capacity;
    }

    std::size_t size() const {
        return count;
    }

    std::size_t physical_index(std::size_t logical_index) const {
        assert(capacity > 0);
        assert(logical_index < count);
        return (start + logical_index) % capacity;
    }

    const Eigen::VectorXd& x_at(std::size_t logical_index) const {
        return x[physical_index(logical_index)];
    }

    const Eigen::VectorXd& f_at(std::size_t logical_index) const {
        return f[physical_index(logical_index)];
    }

    const Eigen::VectorXd& f_back() const {
        assert(count > 0);
        return f_at(count - 1);
    }
};

struct I2SLSIterationCore {
    bool converged = false;
    Eigen::VectorXd beta;
    int iterations = 0;
    std::string failure_reason;
    double final_criterion = std::numeric_limits<double>::infinity();
    int anderson_m_used = 0;
    std::vector<double> kappa_trajectory;
};

struct I2SLSNoFEResultCore {
    bool converged = false;
    Eigen::VectorXd beta;
    Eigen::VectorXd fitted;
    Eigen::VectorXd residual;
    bool predictions_omitted = false;
    std::string failure_reason;
    int first_stage_rank = 0;
    int x_breve_rank = 0;
    int z_full_rank = 0;
    int z_full_columns = 0;
    int phase1_iters = 0;
    int phase1_accepted = 0;
    int phase2_iters = 0;
    int phase2_restarts = 0;
    double final_rho = std::numeric_limits<double>::quiet_NaN();
    double final_criterion = std::numeric_limits<double>::infinity();
    int anderson_m_used = 0;
    double score_residual_max_abs_scaled = std::numeric_limits<double>::quiet_NaN();
    double score_tolerance = std::numeric_limits<double>::quiet_NaN();
    std::string score_validation_status = "not_checked";
    std::vector<double> rho_path;
    std::vector<double> kappa_trajectory;
};

struct I2SLSPathStateCore {
    bool valid = false;
    Eigen::VectorXd linear;
    Eigen::VectorXd linear_base;
    Eigen::VectorXd mu;
    Eigen::VectorXd residual;
    Eigen::VectorXd d_sqrt;
    Eigen::VectorXd ratio;
    Eigen::VectorXd ratio_base;
    Eigen::MatrixXd x_breve;
    int first_stage_rank = 0;
    int x_breve_rank = 0;
    bool weighted_within = false;
};

struct I2SLSPathStepCore {
    bool converged = false;
    Eigen::VectorXd beta;
    double rho = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    std::string failure_reason;
    double final_criterion = std::numeric_limits<double>::infinity();
    double intercept_adjustment = 0.0;
    double intercept_raw = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> kappa_trajectory;
    I2SLSPathStateCore state;
};

struct I2SLSPathRecordCore {
    double tau = 2.0;
    bool has_from_tau = false;
    double from_tau = 0.0;
    bool converged = false;
    Eigen::VectorXd beta;
    bool has_beta = false;
    bool has_rho = false;
    double rho = 0.0;
    int iterations = 0;
    std::string failure_reason;
    double final_criterion = std::numeric_limits<double>::quiet_NaN();
    int x_breve_rank = 0;
    std::string step_failure_reason;
    double intercept_adjustment = 0.0;
    double intercept_raw = std::numeric_limits<double>::quiet_NaN();
};

struct I2SLSNoFEPathResultCore {
    bool converged = false;
    Eigen::VectorXd beta;
    double tau_attained = 2.0;
    bool has_tau_failed = false;
    double tau_failed = 0.0;
    std::string failure_reason;
    bool has_final_rho = false;
    double final_rho = 0.0;
    double min_dtau = 0.0;
    int n_path_evaluations = 0;
    int n_path_successes = 0;
    int n_path_failures = 0;
    std::vector<I2SLSPathRecordCore> path;
};

I2SLSControlCore i2sls_control_from_mapping(py::object control_obj) {
    I2SLSControlCore control;
    control.speed_i2sls_combined_path_state =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_COMBINED_PATH_STATE");
    control.speed_i2sls_path_anderson =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_PATH_ANDERSON");
    control.speed_i2sls_inplace_recenter =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_INPLACE_RECENTER");
    control.speed_trust_full_rank =
        cpp_env_flag_enabled_core("PYIOLS_SPEED_TRUST_FULL_RANK");
    control.speed_i2sls_small_lstsq =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_SMALL_LSTSQ");
    control.speed_i2sls_gram_lstsq =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_GRAM_LSTSQ");
    control.speed_i2sls_small_qr_lstsq =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_SMALL_QR_LSTSQ");
    control.speed_i2sls_compact_diagnostics =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_COMPACT_DIAGNOSTICS");
    control.speed_i2sls_vector_pseudo =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_VECTOR_PSEUDO");
    control.speed_i2sls_omit_predictions =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_OMIT_PREDICTIONS");
    if (control_obj.is_none()) {
        return control;
    }
    const py::dict mapping = py::cast<py::dict>(control_obj);
    auto get_int = [&mapping](const char* key, int current) {
        if (mapping.contains(py::str(key))) {
            return py::cast<int>(mapping[py::str(key)]);
        }
        return current;
    };
    auto get_bool = [&mapping](const char* key, bool current) {
        if (mapping.contains(py::str(key))) {
            return py::cast<bool>(mapping[py::str(key)]);
        }
        return current;
    };
    auto get_double = [&mapping](const char* key, double current) {
        if (mapping.contains(py::str(key))) {
            return py::cast<double>(mapping[py::str(key)]);
        }
        return current;
    };
    control.max_iter = get_int("max_iter", control.max_iter);
    control.tol = get_double("tol", control.tol);
    control.phase2_rho =
        get_double("phase2_rho", get_double("phase2_rho_hint", control.phase2_rho));
    control.damping = get_double("damping", control.damping);
    control.rank_tol = get_double("rank_tol", control.rank_tol);
    control.z_full_rank = get_int("z_full_rank", control.z_full_rank);
    control.first_stage_rank = get_int("first_stage_rank", control.first_stage_rank);
    control.x_breve_rank = get_int("x_breve_rank", control.x_breve_rank);
    control.anderson_m = get_int("anderson_m", control.anderson_m);
    control.delta_log_min = get_double("delta_log_min", control.delta_log_min);
    control.delta_log_max = get_double("delta_log_max", control.delta_log_max);
    control.delta_grid_size = get_int("delta_grid_size", control.delta_grid_size);
    control.rho_log_min = get_double("rho_log_min", control.rho_log_min);
    control.rho_log_max = get_double("rho_log_max", control.rho_log_max);
    control.rho_grid_size = get_int("rho_grid_size", control.rho_grid_size);
    control.tau_path_steps = get_int("tau_path_steps", control.tau_path_steps);
    if (mapping.contains(py::str("min_dtau")) && !mapping[py::str("min_dtau")].is_none()) {
        control.has_min_dtau = true;
        control.min_dtau = py::cast<double>(mapping[py::str("min_dtau")]);
    }
    control.max_path_evaluations =
        get_int("max_path_evaluations", control.max_path_evaluations);
    control.speed_i2sls_phase1_tol =
        get_double("speed_i2sls_phase1_tol", control.speed_i2sls_phase1_tol);
    control.speed_i2sls_compact_diagnostics =
        get_bool(
            "speed_i2sls_compact_diagnostics",
            control.speed_i2sls_compact_diagnostics);
    control.speed_i2sls_vector_pseudo =
        get_bool("speed_i2sls_vector_pseudo", control.speed_i2sls_vector_pseudo);
    control.speed_i2sls_omit_predictions =
        get_bool(
            "speed_i2sls_omit_predictions",
            control.speed_i2sls_omit_predictions);
    return control;
}

std::vector<double> i2sls_exp_grid(double log_min, double log_max, int size) {
    std::vector<double> out;
    const int count = std::max(size, 1);
    out.reserve(static_cast<std::size_t>(count));
    if (count == 1) {
        out.push_back(std::exp(log_min));
        return out;
    }
    for (int idx = 0; idx < count; ++idx) {
        const double frac = static_cast<double>(idx) / static_cast<double>(count - 1);
        out.push_back(std::exp(log_min + frac * (log_max - log_min)));
    }
    return out;
}

std::vector<double> i2sls_rho_grid(const I2SLSControlCore& control) {
    std::vector<double> ordered;
    auto add = [&ordered](double value) {
        if (!std::isfinite(value) || value < 0.0) {
            return;
        }
        for (const double seen : ordered) {
            if (std::abs(value - seen) <= 1.0e-14 * std::max(std::abs(seen), 1.0)) {
                return;
            }
        }
        ordered.push_back(value);
    };
    add(control.phase2_rho);
    for (const double value :
         i2sls_exp_grid(control.rho_log_min, control.rho_log_max,
                        control.rho_grid_size)) {
        add(value);
    }
    return ordered;
}

int matrix_rank_svd_core(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double rank_tol) {
    if (matrix.size() == 0) {
        return 0;
    }
    const Eigen::MatrixXd matrix_copy(matrix);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix_copy);
    const Eigen::VectorXd singular = svd.singularValues();
    const double max_sv = singular.size() == 0 ? 0.0 : singular.maxCoeff();
    const double cutoff =
        std::isfinite(rank_tol) && rank_tol > 0.0
            ? rank_tol
            : std::max<double>(
                  static_cast<double>(matrix.rows()) *
                      std::numeric_limits<double>::epsilon() * max_sv,
                  1.0e-12);
    int rank = 0;
    for (Eigen::Index idx = 0; idx < singular.size(); ++idx) {
        if (singular(idx) > cutoff) {
            ++rank;
        }
    }
    return rank;
}

double matrix_rank_qr_cutoff_core(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const Eigen::ColPivHouseholderQR<Eigen::MatrixXd>& qr,
    double rank_tol) {
    if (std::isfinite(rank_tol) && rank_tol > 0.0) {
        return rank_tol;
    }
    if (matrix.size() == 0) {
        return 1.0e-12;
    }
    const Eigen::Index diag_size =
        std::min(qr.matrixR().rows(), qr.matrixR().cols());
    const double r_scale =
        diag_size > 0 ? std::abs(qr.matrixR()(0, 0)) : 0.0;
    return std::max(
        static_cast<double>(matrix.rows()) *
            std::numeric_limits<double>::epsilon() * r_scale,
        1.0e-12);
}

int matrix_rank_qr_core(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double rank_tol) {
    if (matrix.size() == 0) {
        return 0;
    }
    const Eigen::MatrixXd matrix_copy(matrix);
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{matrix_copy};
    if (qr.info() != Eigen::Success) {
        return 0;
    }
    const Eigen::Index diag_size =
        std::min(qr.matrixR().rows(), qr.matrixR().cols());
    const double cutoff = matrix_rank_qr_cutoff_core(matrix, qr, rank_tol);
    int rank = 0;
    for (Eigen::Index idx = 0; idx < diag_size; ++idx) {
        if (std::abs(qr.matrixR()(idx, idx)) > cutoff) {
            ++rank;
        }
    }
    return rank;
}

std::tuple<Eigen::VectorXd, bool> i2sls_apply_transform_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const GroupBundleCore* transform_bundle,
    double transform_tol,
    int transform_maxiter) {
    if (transform_bundle == nullptr || transform_bundle->groups.empty()) {
        return {Eigen::VectorXd(values), true};
    }
    Eigen::VectorXd transformed;
    DemeanVectorWorkspaceCore workspace;
    bool converged = false;
    demean_vector_for_group_bundle_unweighted_into_core(
        values,
        *transform_bundle,
        transform_tol,
        transform_maxiter,
        transformed,
        workspace,
        &converged);
    return {transformed, converged};
}

bool i2sls_has_transform_bundle_core(const GroupBundleCore* transform_bundle) {
    return transform_bundle != nullptr && !transform_bundle->groups.empty();
}

std::optional<Eigen::VectorXd> i2sls_small_lstsq_fast_path_core(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::VectorXd>& b) {
    if (a.rows() != b.rows() || a.rows() <= a.cols() || a.cols() != 2) {
        return std::nullopt;
    }
    if (!a.allFinite() || !b.allFinite()) {
        return std::nullopt;
    }
    const Eigen::MatrixXd gram = a.transpose() * a;
    const Eigen::VectorXd rhs = a.transpose() * b;
    if (!gram.allFinite() || !rhs.allFinite()) {
        return std::nullopt;
    }
    Eigen::VectorXd beta(a.cols());
    const double g00 = gram(0, 0);
    const double g01 = gram(0, 1);
    const double g11 = gram(1, 1);
    const double scale =
        std::max({std::abs(g00), std::abs(g01), std::abs(g11), 1.0});
    const double det = g00 * g11 - g01 * g01;
    if (!std::isfinite(det) || det <= scale * scale * 1.0e-10) {
        return std::nullopt;
    }
    beta(0) = (rhs(0) * g11 - g01 * rhs(1)) / det;
    beta(1) = (g00 * rhs(1) - g01 * rhs(0)) / det;
    if (!beta.allFinite()) {
        return std::nullopt;
    }
    const double rhs_scale =
        std::max(pyiols::kernel::stable_l2_norm(rhs), 1.0);
    const Eigen::VectorXd normal_residual_vec = gram * beta - rhs;
    const double normal_residual =
        pyiols::kernel::stable_l2_norm(normal_residual_vec) / rhs_scale;
    if (!std::isfinite(normal_residual) || normal_residual > 1.0e-8) {
        return std::nullopt;
    }
    return beta;
}

struct I2SLSPrecomputedSmallLstsqCore {
    bool valid = false;
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;
    double g00 = 0.0;
    double g01 = 0.0;
    double g11 = 0.0;
    double det = 0.0;

    std::optional<Eigen::VectorXd> solve(
        const Eigen::Ref<const Eigen::MatrixXd>& a,
        const Eigen::Ref<const Eigen::VectorXd>& b) const {
        if (!valid || a.rows() != rows || a.cols() != cols ||
            b.rows() != rows || cols != 2) {
            return std::nullopt;
        }
        const Eigen::VectorXd rhs = a.transpose() * b;
        if (!rhs.allFinite()) {
            return std::nullopt;
        }
        Eigen::VectorXd beta(cols);
        beta(0) = (rhs(0) * g11 - g01 * rhs(1)) / det;
        beta(1) = (g00 * rhs(1) - g01 * rhs(0)) / det;
        if (!beta.allFinite()) {
            return std::nullopt;
        }
        Eigen::Vector2d normal_residual_vec;
        normal_residual_vec(0) = g00 * beta(0) + g01 * beta(1) - rhs(0);
        normal_residual_vec(1) = g01 * beta(0) + g11 * beta(1) - rhs(1);
        const double rhs_scale =
            std::max(pyiols::kernel::stable_l2_norm(rhs), 1.0);
        const double normal_residual =
            pyiols::kernel::stable_l2_norm(normal_residual_vec) / rhs_scale;
        if (!std::isfinite(normal_residual) || normal_residual > 1.0e-8) {
            return std::nullopt;
        }
        return beta;
    }

    std::optional<Eigen::VectorXd> solve_rhs(double rhs0, double rhs1) const {
        if (!valid || cols != 2 || !std::isfinite(rhs0) ||
            !std::isfinite(rhs1)) {
            return std::nullopt;
        }
        Eigen::VectorXd beta(cols);
        beta(0) = (rhs0 * g11 - g01 * rhs1) / det;
        beta(1) = (g00 * rhs1 - g01 * rhs0) / det;
        if (!beta.allFinite()) {
            return std::nullopt;
        }
        Eigen::Vector2d normal_residual_vec;
        normal_residual_vec(0) = g00 * beta(0) + g01 * beta(1) - rhs0;
        normal_residual_vec(1) = g01 * beta(0) + g11 * beta(1) - rhs1;
        const double rhs_scale = std::max(std::hypot(rhs0, rhs1), 1.0);
        const double normal_residual =
            pyiols::kernel::stable_l2_norm(normal_residual_vec) / rhs_scale;
        if (!std::isfinite(normal_residual) || normal_residual > 1.0e-8) {
            return std::nullopt;
        }
        return beta;
    }
};

std::optional<I2SLSPrecomputedSmallLstsqCore>
i2sls_precompute_small_lstsq_core(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const I2SLSControlCore& control) {
    if (!control.speed_i2sls_small_lstsq || a.rows() <= a.cols() ||
        a.cols() != 2 || !a.allFinite()) {
        return std::nullopt;
    }
    const Eigen::MatrixXd gram = a.transpose() * a;
    if (!gram.allFinite()) {
        return std::nullopt;
    }
    const double g00 = gram(0, 0);
    const double g01 = gram(0, 1);
    const double g11 = gram(1, 1);
    const double scale =
        std::max({std::abs(g00), std::abs(g01), std::abs(g11), 1.0});
    const double det = g00 * g11 - g01 * g01;
    if (!std::isfinite(det) || det <= scale * scale * 1.0e-10) {
        return std::nullopt;
    }
    I2SLSPrecomputedSmallLstsqCore out;
    out.valid = true;
    out.rows = a.rows();
    out.cols = a.cols();
    out.g00 = g00;
    out.g01 = g01;
    out.g11 = g11;
    out.det = det;
    return out;
}

std::optional<Eigen::VectorXd> i2sls_gram_lstsq_fast_path_core(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::VectorXd>& b) {
    if (a.rows() != b.rows() || a.rows() < a.cols() ||
        a.cols() < 3 || a.cols() > 64) {
        return std::nullopt;
    }
    if (!a.allFinite() || !b.allFinite()) {
        return std::nullopt;
    }
    const Eigen::MatrixXd gram = a.transpose() * a;
    const Eigen::VectorXd rhs = a.transpose() * b;
    if (!gram.allFinite() || !rhs.allFinite()) {
        return std::nullopt;
    }
    return pyiols::kernel::solve_well_conditioned_spd_gram_fast_path(
        gram,
        rhs,
        3);
}

Eigen::MatrixXd i2sls_min_norm_lstsq_core(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b,
    const I2SLSControlCore* control = nullptr);

Eigen::VectorXd i2sls_min_norm_lstsq_vector_core(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const I2SLSControlCore* control = nullptr) {
    const bool small_lstsq_enabled =
        control != nullptr
            ? control->speed_i2sls_small_lstsq
            : cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_SMALL_LSTSQ");
    if (small_lstsq_enabled) {
        auto small = i2sls_small_lstsq_fast_path_core(a, b);
        if (small.has_value()) {
            return std::move(*small);
        }
    }
    const bool gram_lstsq_enabled =
        control != nullptr
            ? control->speed_i2sls_gram_lstsq
            : cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_GRAM_LSTSQ");
    if (gram_lstsq_enabled) {
        auto gram = i2sls_gram_lstsq_fast_path_core(a, b);
        if (gram.has_value()) {
            return std::move(*gram);
        }
    }
    const Eigen::Index qr_min_cols =
        control != nullptr && control->speed_i2sls_small_qr_lstsq ? 3 : 8;
    auto fast =
        pyiols::kernel::lstsq_full_rank_qr_fast_path(
            a,
            b,
            qr_min_cols);
    if (fast.has_value()) {
        return std::move(*fast);
    }
    return pyiols::kernel::min_norm_lstsq_eigen(a, Eigen::MatrixXd(b)).col(0);
}

Eigen::VectorXd i2sls_min_norm_lstsq_vector_precomputed_core(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const I2SLSControlCore* control,
    const I2SLSPrecomputedSmallLstsqCore* precomputed) {
    if (precomputed != nullptr) {
        auto beta = precomputed->solve(a, b);
        if (beta.has_value()) {
            return std::move(*beta);
        }
    }
    return i2sls_min_norm_lstsq_vector_core(a, b, control);
}

Eigen::MatrixXd i2sls_min_norm_lstsq_core(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b,
    const I2SLSControlCore* control) {
    if (b.cols() == 1) {
        Eigen::MatrixXd out(a.cols(), 1);
        out.col(0) = i2sls_min_norm_lstsq_vector_core(a, b.col(0), control);
        return out;
    }
    return pyiols::kernel::min_norm_lstsq_eigen(a, b);
}

std::tuple<Eigen::VectorXd, bool, std::string>
i2sls_phi_step_two_col_identity_core(
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::MatrixXd>& x_breve,
    int phase,
    double param,
    const I2SLSPrecomputedSmallLstsqCore& precomputed_lstsq) {
    if (!precomputed_lstsq.valid || beta.size() != 2 || x_raw.cols() != 2 ||
        x_breve.cols() != 2 || x_raw.rows() != y.size() ||
        x_breve.rows() != y.size()) {
        return {Eigen::VectorXd(beta), false, "invalid_two_col_phi_shape"};
    }
    const double beta0 = beta(0);
    const double beta1 = beta(1);
    const double eta = phase == 0 ? 0.0 : 1.0 / (1.0 + param);
    const double inv_phase1_scale = phase == 0 ? 1.0 / (1.0 + param) : 0.0;
    double rhs0 = 0.0;
    double rhs1 = 0.0;
    for (Eigen::Index row = 0; row < y.size(); ++row) {
        const double linear = x_raw(row, 0) * beta0 + x_raw(row, 1) * beta1;
        const double mu =
            std::exp(std::min(std::max(linear, -745.0), 709.0));
        const double ratio = y(row) / std::max(mu, 1.0e-300);
        double pseudo = 0.0;
        if (phase == 0) {
            const double adjustment = (ratio - 1.0) * inv_phase1_scale;
            if (adjustment <= -1.0 || !std::isfinite(adjustment)) {
                return {
                    Eigen::VectorXd(beta),
                    false,
                    "invalid_phase1_log_adjustment"};
            }
            pseudo = linear + std::log1p(adjustment);
        } else {
            pseudo = linear + eta * (ratio - 1.0);
        }
        if (!std::isfinite(pseudo)) {
            return {Eigen::VectorXd(beta), false, "nonfinite_pseudo_response"};
        }
        rhs0 += x_breve(row, 0) * pseudo;
        rhs1 += x_breve(row, 1) * pseudo;
    }
    auto beta_new = precomputed_lstsq.solve_rhs(rhs0, rhs1);
    if (!beta_new.has_value()) {
        return {Eigen::VectorXd(beta), false, "nonfinite_phi_beta"};
    }
    return {std::move(*beta_new), true, ""};
}

std::tuple<Eigen::VectorXd, bool, std::string>
i2sls_phi_step_vector_pseudo_core(
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::MatrixXd>& x_breve,
    const GroupBundleCore* transform_bundle,
    double transform_tol,
    int transform_maxiter,
    int phase,
    double param,
    const I2SLSControlCore& control,
    const I2SLSPrecomputedSmallLstsqCore* precomputed_lstsq) {
    const Eigen::VectorXd linear = x_raw * beta;
    const Eigen::VectorXd mu =
        linear.array().min(709.0).max(-745.0).exp().matrix();
    const Eigen::ArrayXd ratio =
        y.array() / mu.array().max(1.0e-300);
    Eigen::VectorXd pseudo(linear.size());
    if (phase == 0) {
        const Eigen::ArrayXd adjustment = (ratio - 1.0) / (1.0 + param);
        if (!adjustment.isFinite().all() || !(adjustment > -1.0).all()) {
            return {
                Eigen::VectorXd(beta),
                false,
                "invalid_phase1_log_adjustment"};
        }
        pseudo = (linear.array() + adjustment.log1p()).matrix();
    } else {
        const double eta = 1.0 / (1.0 + param);
        pseudo = (linear.array() + eta * (ratio - 1.0)).matrix();
    }
    if (!pseudo.allFinite()) {
        return {Eigen::VectorXd(beta), false, "nonfinite_pseudo_response"};
    }
    Eigen::VectorXd beta_new;
    if (i2sls_has_transform_bundle_core(transform_bundle)) {
        const auto [pseudo_fit, transform_converged] = i2sls_apply_transform_core(
            pseudo,
            transform_bundle,
            transform_tol,
            transform_maxiter);
        if (!transform_converged) {
            return {Eigen::VectorXd(beta), false, "i2sls_transform_not_converged"};
        }
        if (!pseudo_fit.allFinite()) {
            return {
                Eigen::VectorXd(beta),
                false,
                "nonfinite_transformed_pseudo_response"};
        }
        beta_new =
            i2sls_min_norm_lstsq_vector_precomputed_core(
                x_breve,
                pseudo_fit,
                &control,
                precomputed_lstsq);
    } else {
        beta_new =
            i2sls_min_norm_lstsq_vector_precomputed_core(
                x_breve,
                pseudo,
                &control,
                precomputed_lstsq);
    }
    if (!beta_new.allFinite()) {
        return {Eigen::VectorXd(beta), false, "nonfinite_phi_beta"};
    }
    return {beta_new, true, ""};
}

std::tuple<Eigen::VectorXd, bool, std::string> i2sls_phi_step_core(
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::MatrixXd>& x_breve,
    const GroupBundleCore* transform_bundle,
    double transform_tol,
    int transform_maxiter,
    int phase,
    double param,
    const I2SLSControlCore& control,
    const I2SLSPrecomputedSmallLstsqCore* precomputed_lstsq) {
    if (cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_I2SLS_FUSED_TWO_COL") &&
        !i2sls_has_transform_bundle_core(transform_bundle) &&
        precomputed_lstsq != nullptr && precomputed_lstsq->valid &&
        x_raw.cols() == 2 && x_breve.cols() == 2 && beta.size() == 2) {
        return i2sls_phi_step_two_col_identity_core(
            beta,
            y,
            x_raw,
            x_breve,
            phase,
            param,
            *precomputed_lstsq);
    }
    if (control.speed_i2sls_vector_pseudo) {
        return i2sls_phi_step_vector_pseudo_core(
            beta,
            y,
            x_raw,
            x_breve,
            transform_bundle,
            transform_tol,
            transform_maxiter,
            phase,
            param,
            control,
            precomputed_lstsq);
    }
    const Eigen::VectorXd linear = x_raw * beta;
    Eigen::VectorXd mu(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        mu(row) = std::exp(std::min(std::max(linear(row), -745.0), 709.0));
    }
    Eigen::VectorXd pseudo(linear.size());
    if (phase == 0) {
        for (Eigen::Index row = 0; row < linear.size(); ++row) {
            const double ratio = y(row) / std::max(mu(row), 1.0e-300);
            const double adjustment = (ratio - 1.0) / (1.0 + param);
            if (adjustment <= -1.0 || !std::isfinite(adjustment)) {
                return {
                    Eigen::VectorXd(beta),
                    false,
                    "invalid_phase1_log_adjustment"};
            }
            pseudo(row) = linear(row) + std::log1p(adjustment);
        }
    } else {
        const double eta = 1.0 / (1.0 + param);
        for (Eigen::Index row = 0; row < linear.size(); ++row) {
            const double ratio = y(row) / std::max(mu(row), 1.0e-300);
            pseudo(row) = linear(row) + eta * (ratio - 1.0);
        }
    }
    if (!pseudo.allFinite()) {
        return {Eigen::VectorXd(beta), false, "nonfinite_pseudo_response"};
    }
    Eigen::VectorXd beta_new;
    if (i2sls_has_transform_bundle_core(transform_bundle)) {
        const auto [pseudo_fit, transform_converged] = i2sls_apply_transform_core(
            pseudo,
            transform_bundle,
            transform_tol,
            transform_maxiter);
        if (!transform_converged) {
            return {Eigen::VectorXd(beta), false, "i2sls_transform_not_converged"};
        }
        if (!pseudo_fit.allFinite()) {
            return {
                Eigen::VectorXd(beta),
                false,
                "nonfinite_transformed_pseudo_response"};
        }
        beta_new =
            i2sls_min_norm_lstsq_vector_precomputed_core(
                x_breve,
                pseudo_fit,
                &control,
                precomputed_lstsq);
    } else {
        beta_new =
            i2sls_min_norm_lstsq_vector_precomputed_core(
                x_breve,
                pseudo,
                &control,
                precomputed_lstsq);
    }
    if (!beta_new.allFinite()) {
        return {Eigen::VectorXd(beta), false, "nonfinite_phi_beta"};
    }
    return {beta_new, true, ""};
}

std::optional<Eigen::VectorXd> i2sls_anderson_candidate_core(
    const Eigen::Ref<const Eigen::VectorXd>& beta_phi,
    const EigenVectorPairRingCore& history,
    const I2SLSControlCore& control) {
    if (history.size() < 2) {
        return std::nullopt;
    }
    const std::size_t m_cur = history.size() - 1;
    const Eigen::VectorXd& f_current = history.f_back();
    Eigen::MatrixXd f_mat(f_current.size(), static_cast<Eigen::Index>(m_cur));
    for (std::size_t j = 1; j <= m_cur; ++j) {
        f_mat.col(static_cast<Eigen::Index>(j - 1)) =
            f_current - history.f_at(history.size() - j - 1);
    }
    Eigen::VectorXd gamma;
    try {
        gamma = i2sls_min_norm_lstsq_vector_core(f_mat, f_current, &control);
    } catch (const std::exception&) {
        return std::nullopt;
    }
    Eigen::VectorXd beta_aa = beta_phi;
    for (std::size_t idx = 0; idx < m_cur; ++idx) {
        const std::size_t previous_idx = history.size() - idx - 2;
        const Eigen::VectorXd phi_prev =
            history.x_at(previous_idx) + history.f_at(previous_idx);
        beta_aa += gamma(static_cast<Eigen::Index>(idx)) * (phi_prev - beta_phi);
    }
    if (!beta_aa.allFinite()) {
        return std::nullopt;
    }
    return beta_aa;
}

I2SLSIterationCore i2sls_iterate_fixed_point_core(
    const Eigen::Ref<const Eigen::VectorXd>& beta_start,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::MatrixXd>& x_breve,
    const GroupBundleCore* transform_bundle,
    double transform_tol,
    int transform_maxiter,
    int phase,
    double param,
    const I2SLSControlCore& control,
    const I2SLSPrecomputedSmallLstsqCore* precomputed_lstsq) {
    I2SLSIterationCore out;
    Eigen::VectorXd beta = beta_start;
    const int aa_m = phase == 1 ? std::max(control.anderson_m, 0) : 0;
    EigenVectorPairRingCore history;
    if (aa_m > 0) {
        history.reset(static_cast<std::size_t>(aa_m + 1));
    }
    if (!control.speed_i2sls_compact_diagnostics) {
        out.kappa_trajectory.reserve(
            static_cast<std::size_t>(std::max(control.max_iter - 1, 0)));
    }
    double previous_criterion = std::numeric_limits<double>::quiet_NaN();
    for (int iteration = 1; iteration <= control.max_iter; ++iteration) {
        auto [beta_phi, valid, phi_failure] =
            i2sls_phi_step_core(
                beta,
                y,
                x_raw,
                x_breve,
                transform_bundle,
                transform_tol,
                transform_maxiter,
                phase,
                param,
                control,
                precomputed_lstsq);
        if (!valid) {
            out.failure_reason =
                phi_failure.empty() ? "invalid_phi_step" : phi_failure;
            break;
        }
        Eigen::VectorXd candidate = beta_phi;
        if (aa_m > 0) {
            const Eigen::VectorXd f_k = beta_phi - beta;
            history.push(beta, f_k);
            const auto beta_aa =
                i2sls_anderson_candidate_core(
                    beta_phi,
                    history,
                    control);
            if (beta_aa.has_value()) {
                auto [beta_aa_phi, aa_valid, aa_failure] =
                    i2sls_phi_step_core(
                        *beta_aa,
                        y,
                        x_raw,
                        x_breve,
                        transform_bundle,
                        transform_tol,
                        transform_maxiter,
                        phase,
                        param,
                        control,
                        precomputed_lstsq);
                (void) aa_failure;
                if (aa_valid) {
                    const double phi_step = pyiols::kernel::stable_l2_norm(beta_phi - beta);
                    const double aa_step = pyiols::kernel::stable_l2_norm(beta_aa_phi - *beta_aa);
                    if (aa_step <= phi_step) {
                        candidate = *beta_aa;
                    }
                }
            }
        }
        Eigen::VectorXd beta_new =
            (1.0 - control.damping) * beta + control.damping * candidate;
        const double step = pyiols::kernel::stable_l2_norm(beta_new - beta);
        const double scale = std::max(pyiols::kernel::stable_l2_norm(beta), 1.0);
        const double criterion = step / scale;
        if (!control.speed_i2sls_compact_diagnostics &&
            std::isfinite(previous_criterion) && previous_criterion > 0.0) {
            out.kappa_trajectory.push_back(criterion / previous_criterion);
        }
        previous_criterion = criterion;
        beta = beta_new;
        out.iterations = iteration;
        out.final_criterion = criterion;
        if (criterion <= control.tol) {
            out.converged = true;
            out.beta = beta;
            out.anderson_m_used = aa_m;
            return out;
        }
    }
    if (out.failure_reason.empty()) {
        out.failure_reason =
            phase == 1 ? "phase2_evaluation_budget_exhausted" : "not_converged";
    }
    out.beta = beta;
    out.anderson_m_used = aa_m;
    return out;
}

I2SLSNoFEResultCore solve_i2sls_nofe_tau2_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::MatrixXd>& z_full,
    const Eigen::Ref<const Eigen::MatrixXd>& x_breve,
    const GroupBundleCore* transform_bundle,
    double transform_tol,
    int transform_maxiter,
    const I2SLSControlCore& control) {
    I2SLSNoFEResultCore out;
    out.z_full_columns = static_cast<int>(z_full.cols());
    if (y.size() != x_raw.rows() || y.size() != z_full.rows()) {
        throw std::invalid_argument("i2SLS endog/exog/Z rows do not match");
    }
    if (x_breve.rows() != x_raw.rows() || x_breve.cols() != x_raw.cols()) {
        throw std::invalid_argument("x_breve must match exog shape");
    }
    Eigen::VectorXd log1p_y(y.size());
    for (Eigen::Index row = 0; row < y.size(); ++row) {
        if (y(row) < 0.0 || !std::isfinite(y(row))) {
            throw std::invalid_argument("endog must be finite and non-negative");
        }
        log1p_y(row) = std::log1p(y(row));
    }
    out.z_full_rank =
        control.z_full_rank >= 0
            ? control.z_full_rank
            : matrix_rank_svd_core(z_full, control.rank_tol);
    if (out.z_full_rank == 0) {
        out.failure_reason = "rank_deficient_first_stage_at_anchor";
        out.beta = Eigen::VectorXd::Constant(
            x_raw.cols(), std::numeric_limits<double>::quiet_NaN());
        out.fitted = Eigen::VectorXd::Constant(
            y.size(), std::numeric_limits<double>::quiet_NaN());
        out.residual = out.fitted;
        return out;
    }
    out.first_stage_rank =
        control.first_stage_rank >= 0
            ? control.first_stage_rank
            : matrix_rank_svd_core(z_full.transpose() * x_raw, control.rank_tol);
    out.x_breve_rank =
        control.x_breve_rank >= 0
            ? control.x_breve_rank
            : matrix_rank_svd_core(x_breve, control.rank_tol);
    if (out.first_stage_rank < x_raw.cols() || out.x_breve_rank < x_raw.cols()) {
        out.failure_reason = "rank_deficient_first_stage_at_anchor";
        out.beta = Eigen::VectorXd::Constant(
            x_raw.cols(), std::numeric_limits<double>::quiet_NaN());
        out.fitted = Eigen::VectorXd::Constant(
            y.size(), std::numeric_limits<double>::quiet_NaN());
        out.residual = out.fitted;
        return out;
    }
    const auto [log1p_y_fit, log1p_y_transform_converged] =
        i2sls_apply_transform_core(
            log1p_y,
            transform_bundle,
            transform_tol,
            transform_maxiter);
    if (!log1p_y_transform_converged) {
        out.failure_reason = "i2sls_transform_not_converged";
        out.beta = Eigen::VectorXd::Constant(
            x_raw.cols(), std::numeric_limits<double>::quiet_NaN());
        out.fitted = Eigen::VectorXd::Constant(
            y.size(), std::numeric_limits<double>::quiet_NaN());
        out.residual = out.fitted;
        return out;
    }
    const std::optional<I2SLSPrecomputedSmallLstsqCore> precomputed_lstsq =
        i2sls_precompute_small_lstsq_core(x_breve, control);
    const I2SLSPrecomputedSmallLstsqCore* precomputed_lstsq_ptr =
        precomputed_lstsq.has_value() ? &(*precomputed_lstsq) : nullptr;
    Eigen::VectorXd beta =
        i2sls_min_norm_lstsq_vector_precomputed_core(
            x_breve,
            log1p_y_fit,
            &control,
            precomputed_lstsq_ptr);
    out.rho_path.reserve(
        static_cast<std::size_t>(std::max(control.rho_grid_size, 0)));
    if (!control.speed_i2sls_compact_diagnostics) {
        out.kappa_trajectory.reserve(static_cast<std::size_t>(
            std::max(control.max_iter, 0) *
            std::max(control.delta_grid_size + control.rho_grid_size + 2, 1)));
    }
    I2SLSControlCore phase1_control = control;
    if (std::isfinite(control.speed_i2sls_phase1_tol) &&
        control.speed_i2sls_phase1_tol > control.tol) {
        phase1_control.tol = control.speed_i2sls_phase1_tol;
    }
    for (const double delta :
         i2sls_exp_grid(control.delta_log_min, control.delta_log_max,
                        control.delta_grid_size)) {
        I2SLSIterationCore phase1 = i2sls_iterate_fixed_point_core(
            beta,
            y,
            x_raw,
            x_breve,
            transform_bundle,
            transform_tol,
            transform_maxiter,
            0,
            delta,
            phase1_control,
            precomputed_lstsq_ptr);
        out.phase1_iters += phase1.iterations;
        if (!control.speed_i2sls_compact_diagnostics) {
            out.kappa_trajectory.insert(
                out.kappa_trajectory.end(),
                phase1.kappa_trajectory.begin(),
                phase1.kappa_trajectory.end());
        }
        if (phase1.converged) {
            beta = phase1.beta;
            out.phase1_accepted += 1;
        }
    }
    if (out.phase1_accepted == 0) {
        out.failure_reason = "phase1_candidate_unusable";
        out.beta = Eigen::VectorXd::Constant(
            x_raw.cols(), std::numeric_limits<double>::quiet_NaN());
        out.fitted = Eigen::VectorXd::Constant(
            y.size(), std::numeric_limits<double>::quiet_NaN());
        out.residual = out.fitted;
        return out;
    }
    for (const double rho : i2sls_rho_grid(control)) {
        out.rho_path.push_back(rho);
        I2SLSIterationCore phase2 = i2sls_iterate_fixed_point_core(
            beta,
            y,
            x_raw,
            x_breve,
            transform_bundle,
            transform_tol,
            transform_maxiter,
            1,
            rho,
            control,
            precomputed_lstsq_ptr);
        out.phase2_iters += phase2.iterations;
        if (!control.speed_i2sls_compact_diagnostics) {
            out.kappa_trajectory.insert(
                out.kappa_trajectory.end(),
                phase2.kappa_trajectory.begin(),
                phase2.kappa_trajectory.end());
        }
        if (phase2.converged) {
            beta = phase2.beta;
            out.converged = true;
            out.final_rho = rho;
            out.final_criterion = phase2.final_criterion;
            out.anderson_m_used = phase2.anderson_m_used;
            break;
        }
        out.phase2_restarts += 1;
    }
    if (!out.converged) {
        out.failure_reason = "phase2_evaluation_budget_exhausted";
        out.beta = Eigen::VectorXd::Constant(
            x_raw.cols(), std::numeric_limits<double>::quiet_NaN());
        out.fitted = Eigen::VectorXd::Constant(
            y.size(), std::numeric_limits<double>::quiet_NaN());
        out.residual = out.fitted;
        return out;
    }
    out.beta = beta;
    if (control.speed_i2sls_omit_predictions) {
        out.predictions_omitted = true;
        out.score_validation_status = "omitted_speed_compact_diagnostics";
        return out;
    }
    const Eigen::VectorXd linear = x_raw * beta;
    out.fitted.resize(y.size());
    for (Eigen::Index row = 0; row < y.size(); ++row) {
        out.fitted(row) =
            std::exp(std::min(std::max(linear(row), -745.0), 709.0));
    }
    out.residual = y - out.fitted;
    if (control.speed_i2sls_compact_diagnostics) {
        out.score_validation_status = "omitted_speed_compact_diagnostics";
        return out;
    }
    const auto [score_residual, score_transform_converged] =
        i2sls_apply_transform_core(
            out.residual,
            transform_bundle,
            transform_tol,
            transform_maxiter);
    Eigen::VectorXd score;
    if (score_transform_converged) {
        score = z_full.transpose() * score_residual;
    } else {
        score = Eigen::VectorXd::Constant(
            z_full.cols(),
            std::numeric_limits<double>::infinity());
    }
    out.score_residual_max_abs_scaled =
        score.cwiseAbs().maxCoeff() / std::max<double>(static_cast<double>(y.size()), 1.0);
    out.score_tolerance = std::max(10.0 * control.tol, 1.0e-8);
    out.score_validation_status =
        out.score_residual_max_abs_scaled <= out.score_tolerance
            ? "passed"
            : "diagnostic_warning";
    return out;
}

std::vector<double> i2sls_rho_grid_with_hint_core(
    const I2SLSControlCore& control,
    double rho_hint) {
    std::vector<double> ordered;
    ordered.reserve(
        static_cast<std::size_t>(std::max(control.rho_grid_size + 1, 1)));
    auto add = [&ordered](double value) {
        if (!std::isfinite(value) || value < 0.0) {
            return;
        }
        for (const double seen : ordered) {
            if (std::abs(value - seen) <= 1.0e-14 * std::max(std::abs(seen), 1.0)) {
                return;
            }
        }
        ordered.push_back(value);
    };
    if (std::isfinite(rho_hint) && rho_hint >= 0.0) {
        add(rho_hint);
    }
    for (const double value : i2sls_rho_grid(control)) {
        add(value);
    }
    return ordered;
}

std::vector<double> i2sls_tau_path_schedule_core(
    double tau_target,
    const I2SLSControlCore& control) {
    std::vector<double> out;
    if (std::abs(tau_target - 2.0) <= 1.0e-12) {
        return out;
    }
    const int steps = std::max(control.tau_path_steps, 1);
    out.reserve(static_cast<std::size_t>(steps));
    for (int idx = 1; idx <= steps; ++idx) {
        const double progress = static_cast<double>(idx) / static_cast<double>(steps);
        const double shaped = 1.0 - std::pow(1.0 - progress, 2.0);
        out.push_back(2.0 + (tau_target - 2.0) * shaped);
    }
    return out;
}

double i2sls_rank_tolerance_core(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double rank_tol) {
    if (std::isfinite(rank_tol) && rank_tol > 0.0) {
        return rank_tol;
    }
    if (matrix.size() == 0) {
        return 1.0e-12;
    }
    const Eigen::MatrixXd matrix_copy(matrix);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix_copy);
    const Eigen::VectorXd singular = svd.singularValues();
    const double max_sv = singular.size() == 0 ? 0.0 : singular.maxCoeff();
    return std::max(
        static_cast<double>(matrix.rows()) *
            std::numeric_limits<double>::epsilon() * max_sv,
        1.0e-12);
}

I2SLSPathStateCore i2sls_path_state_for_beta_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::MatrixXd>& z_full,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    double tau,
    const GroupBundleCore* transform_bundle,
    double transform_tol,
    int transform_maxiter,
    const I2SLSControlCore& control) {
    I2SLSPathStateCore state;
    if (x_raw.rows() != y.size() || z_full.rows() != y.size() ||
        x_raw.cols() != beta.size()) {
        return state;
    }
    state.linear = x_raw * beta;
    state.d_sqrt =
        (0.5 * (2.0 - tau) * state.linear.array())
            .min(709.0)
            .max(-745.0)
            .exp()
            .matrix();
    state.mu = state.linear.array().min(709.0).max(-745.0).exp().matrix();
    state.residual = y - state.mu;
    if (!state.d_sqrt.allFinite() || !state.mu.allFinite()) {
        return I2SLSPathStateCore{};
    }
    state.ratio = (y.array() / state.mu.array().max(1.0e-300)).matrix();
    if (!state.ratio.allFinite()) {
        return I2SLSPathStateCore{};
    }

    Eigen::MatrixXd x_base;
    Eigen::MatrixXd z_base;
    state.linear_base = state.linear;
    state.ratio_base = state.ratio;
    if (transform_bundle != nullptr && !transform_bundle->groups.empty()) {
        Eigen::VectorXd weights = state.d_sqrt.array().square().matrix();
        if (!weights.allFinite()) {
            return I2SLSPathStateCore{};
        }
        bool x_converged = false;
        bool z_converged = false;
        bool ratio_converged = false;
        if (control.speed_i2sls_combined_path_state) {
            Eigen::MatrixXd combined(
                y.size(), x_raw.cols() + z_full.cols() + 1);
            combined.leftCols(x_raw.cols()) = x_raw;
            combined.middleCols(x_raw.cols(), z_full.cols()) = z_full;
            combined.col(x_raw.cols() + z_full.cols()) = state.ratio;
            Eigen::MatrixXd combined_base;
            DemeanWorkspaceCore combined_workspace;
            bool combined_converged = false;
            demean_for_group_bundle_into_core(
                combined,
                *transform_bundle,
                &weights,
                transform_tol,
                transform_maxiter,
                combined_base,
                combined_workspace,
                &combined_converged);
            x_converged = combined_converged;
            z_converged = combined_converged;
            ratio_converged = combined_converged;
            if (combined_converged) {
                x_base = combined_base.leftCols(x_raw.cols());
                z_base = combined_base.middleCols(x_raw.cols(), z_full.cols());
                state.ratio_base =
                    combined_base.col(x_raw.cols() + z_full.cols());
            }
        } else {
            DemeanWorkspaceCore x_workspace;
            DemeanWorkspaceCore z_workspace;
            DemeanWorkspaceCore ratio_workspace;
            demean_for_group_bundle_into_core(
                x_raw,
                *transform_bundle,
                &weights,
                transform_tol,
                transform_maxiter,
                x_base,
                x_workspace,
                &x_converged);
            demean_for_group_bundle_into_core(
                z_full,
                *transform_bundle,
                &weights,
                transform_tol,
                transform_maxiter,
                z_base,
                z_workspace,
                &z_converged);
            Eigen::MatrixXd ratio_mat(state.ratio.rows(), 1);
            ratio_mat.col(0) = state.ratio;
            Eigen::MatrixXd ratio_base_mat;
            demean_for_group_bundle_into_core(
                ratio_mat,
                *transform_bundle,
                &weights,
                transform_tol,
                transform_maxiter,
                ratio_base_mat,
                ratio_workspace,
                &ratio_converged);
            if (ratio_converged) {
                state.ratio_base = ratio_base_mat.col(0);
            }
        }
        if (!x_converged || !z_converged || !ratio_converged) {
            return I2SLSPathStateCore{};
        }
        state.linear_base = x_base * beta;
        state.weighted_within = true;
        if (!x_base.allFinite() || !z_base.allFinite() ||
            !state.ratio_base.allFinite() || !state.linear_base.allFinite()) {
            return I2SLSPathStateCore{};
        }
    }

    Eigen::MatrixXd x_star(y.size(), x_raw.cols());
    Eigen::MatrixXd z_star(y.size(), z_full.cols());
    if (state.weighted_within) {
        x_star = x_base.array().colwise() * state.d_sqrt.array();
        z_star = z_base.array().colwise() * state.d_sqrt.array();
    } else {
        x_star = x_raw.array().colwise() * state.d_sqrt.array();
        z_star = z_full.array().colwise() * state.d_sqrt.array();
    }
    if (!x_star.allFinite() || !z_star.allFinite()) {
        return I2SLSPathStateCore{};
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{z_star};
    if (qr.info() != Eigen::Success) {
        return I2SLSPathStateCore{};
    }
    const Eigen::Index q_cols = std::min(z_star.rows(), z_star.cols());
    const double cutoff = matrix_rank_qr_cutoff_core(z_star, qr, control.rank_tol);
    int rank = 0;
    for (Eigen::Index idx = 0; idx < q_cols; ++idx) {
        if (std::abs(qr.matrixR()(idx, idx)) > cutoff) {
            ++rank;
        }
    }
    if (rank == 0) {
        return I2SLSPathStateCore{};
    }
    const Eigen::MatrixXd q_rank =
        qr.householderQ() *
        Eigen::MatrixXd::Identity(z_star.rows(), static_cast<Eigen::Index>(rank));
    state.x_breve = q_rank * (q_rank.transpose() * x_star);
    const bool trust_full_rank =
        control.speed_trust_full_rank &&
        rank >= x_raw.cols() &&
        control.x_breve_rank >= x_raw.cols() &&
        control.first_stage_rank >= x_raw.cols();
    if (trust_full_rank) {
        state.x_breve_rank = control.x_breve_rank;
        state.first_stage_rank = control.first_stage_rank;
    } else {
        state.x_breve_rank = matrix_rank_qr_core(state.x_breve, control.rank_tol);
        if (state.x_breve_rank < x_raw.cols()) {
            return I2SLSPathStateCore{};
        }
        state.first_stage_rank = static_cast<int>(x_raw.cols());
    }
    state.valid = true;
    return state;
}

std::pair<Eigen::VectorXd, double> i2sls_recenter_intercept_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    double tau) {
    if (x_raw.cols() == 0 || beta.size() == 0 || x_raw.rows() != y.size() ||
        x_raw.cols() != beta.size()) {
        return {Eigen::VectorXd(beta), 0.0};
    }
    constexpr double intercept_atol = 1.0e-8;
    constexpr double intercept_rtol = 1.0e-5;
    for (Eigen::Index row = 0; row < x_raw.rows(); ++row) {
        if (std::abs(x_raw(row, 0) - 1.0) >
            intercept_atol + intercept_rtol) {
            return {Eigen::VectorXd(beta), 0.0};
        }
    }
    if (y.size() == 0) {
        return {Eigen::VectorXd(beta), 0.0};
    }
    const Eigen::VectorXd linear = x_raw * beta;
    Eigen::VectorXd numerator =
        Eigen::VectorXd::Constant(linear.size(), -std::numeric_limits<double>::infinity());
    Eigen::VectorXd denominator(linear.size());
    for (Eigen::Index row = 0; row < y.size(); ++row) {
        if (y(row) > 0.0) {
            numerator(row) = std::log(y(row)) + (1.0 - tau) * linear(row);
        }
        denominator(row) = (2.0 - tau) * linear(row);
    }
    auto logsumexp = [](const Eigen::VectorXd& values) -> double {
        double max_value = -std::numeric_limits<double>::infinity();
        for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
            if (std::isfinite(values(idx)) && values(idx) > max_value) {
                max_value = values(idx);
            }
        }
        if (!std::isfinite(max_value)) {
            return max_value;
        }
        double sum = 0.0;
        for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
            if (std::isfinite(values(idx))) {
                sum += std::exp(values(idx) - max_value);
            }
        }
        return max_value + std::log(sum);
    };
    const double log_numerator = logsumexp(numerator);
    const double log_denominator = logsumexp(denominator);
    if (!std::isfinite(log_numerator) || !std::isfinite(log_denominator)) {
        return {Eigen::VectorXd(beta), 0.0};
    }
    const double adjustment = log_numerator - log_denominator;
    Eigen::VectorXd beta_recentered(beta);
    beta_recentered(0) += adjustment;
    return {beta_recentered, adjustment};
}

bool i2sls_recenter_path_state_inplace_core(
    I2SLSPathStateCore& state,
    double adjustment,
    double tau) {
    if (adjustment == 0.0) {
        return true;
    }
    const double mu_scale = std::exp(adjustment);
    const double d_scale = std::exp(0.5 * (2.0 - tau) * adjustment);
    const double ratio_scale = std::exp(-adjustment);
    if (!std::isfinite(mu_scale) || !std::isfinite(d_scale) ||
        !std::isfinite(ratio_scale)) {
        return false;
    }
    state.linear.array() += adjustment;
    state.linear_base.array() += adjustment;
    state.residual.array() += state.mu.array() * (1.0 - mu_scale);
    state.mu *= mu_scale;
    state.d_sqrt *= d_scale;
    state.ratio *= ratio_scale;
    state.ratio_base *= ratio_scale;
    state.x_breve *= d_scale;
    return state.linear.allFinite() && state.linear_base.allFinite() &&
           state.mu.allFinite() && state.residual.allFinite() &&
           state.d_sqrt.allFinite() && state.ratio.allFinite() &&
           state.ratio_base.allFinite() && state.x_breve.allFinite();
}

I2SLSPathStepCore i2sls_iterate_nofe_tau_step_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::MatrixXd>& z_full,
    const Eigen::Ref<const Eigen::VectorXd>& beta_start,
    double tau,
    double rho_hint,
    const GroupBundleCore* transform_bundle,
    double transform_tol,
    int transform_maxiter,
    const I2SLSControlCore& control) {
    for (const double rho : i2sls_rho_grid_with_hint_core(control, rho_hint)) {
        Eigen::VectorXd beta = beta_start;
        const int aa_m =
            control.speed_i2sls_path_anderson
                ? std::max(control.anderson_m, 0)
                : 0;
        EigenVectorPairRingCore history;
        if (aa_m > 0) {
            history.reset(static_cast<std::size_t>(aa_m + 1));
        }
        double criterion_previous = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> kappa_trajectory;
        kappa_trajectory.reserve(
            static_cast<std::size_t>(std::max(control.max_iter - 1, 0)));
        double final_criterion = std::numeric_limits<double>::infinity();
        for (int iteration = 1; iteration <= control.max_iter; ++iteration) {
            const I2SLSPathStateCore state =
                i2sls_path_state_for_beta_core(
                    y,
                    x_raw,
                    z_full,
                    beta,
                    tau,
                    transform_bundle,
                    transform_tol,
                    transform_maxiter,
                    control);
            if (!state.valid) {
                break;
            }
            const double eta = 1.0 / (1.0 + rho);
            Eigen::VectorXd y_star(y.size());
            if (state.weighted_within) {
                y_star =
                    (state.d_sqrt.array() *
                     (state.linear_base.array() + eta * state.ratio_base.array()))
                        .matrix();
            } else {
                y_star =
                    (state.d_sqrt.array() *
                     (state.linear.array() + eta * (state.ratio.array() - 1.0)))
                        .matrix();
            }
            if (!y_star.allFinite()) {
                break;
            }
            const Eigen::VectorXd beta_phi =
                i2sls_min_norm_lstsq_vector_core(
                    state.x_breve,
                    y_star,
                    &control);
            if (!beta_phi.allFinite()) {
                break;
            }
            Eigen::VectorXd candidate = beta_phi;
            if (aa_m > 0) {
                const Eigen::VectorXd f_k = beta_phi - beta;
                history.push(beta, f_k);
                const auto beta_aa =
                    i2sls_anderson_candidate_core(
                        beta_phi,
                        history,
                        control);
                if (beta_aa.has_value() && beta_aa->allFinite()) {
                    candidate = *beta_aa;
                }
            }
            const Eigen::VectorXd beta_next =
                (1.0 - control.damping) * beta + control.damping * candidate;
            const double step = pyiols::kernel::stable_l2_norm(beta_next - beta);
            const double scale =
                std::max(pyiols::kernel::stable_l2_norm(beta), 1.0);
            final_criterion = step / scale;
            if (std::isfinite(criterion_previous) && criterion_previous > 0.0) {
                kappa_trajectory.push_back(final_criterion / criterion_previous);
            }
            criterion_previous = final_criterion;
            beta = beta_next;
            if (final_criterion <= control.tol) {
                const I2SLSPathStateCore final_state =
                    i2sls_path_state_for_beta_core(
                        y,
                        x_raw,
                        z_full,
                        beta,
                        tau,
                        transform_bundle,
                        transform_tol,
                        transform_maxiter,
                        control);
                if (!final_state.valid) {
                    break;
                }
                I2SLSPathStepCore out;
                out.converged = true;
                out.beta = beta;
                out.rho = rho;
                out.iterations = iteration;
                out.final_criterion = final_criterion;
                out.kappa_trajectory = kappa_trajectory;
                out.state = final_state;
                return out;
            }
        }
    }
    I2SLSPathStepCore out;
    out.converged = false;
    out.beta = beta_start;
    out.iterations = control.max_iter;
    out.failure_reason = "phase2_evaluation_budget_exhausted";
    return out;
}

I2SLSPathRecordCore i2sls_make_path_record_core(
    double tau,
    double from_tau,
    const I2SLSPathStepCore& step,
    const std::string& failure_reason_override = "") {
    I2SLSPathRecordCore record;
    record.tau = tau;
    record.has_from_tau = true;
    record.from_tau = from_tau;
    record.converged = step.converged;
    record.iterations = step.iterations;
    if (step.converged) {
        record.beta = step.beta;
        record.has_beta = true;
        if (std::isfinite(step.rho)) {
            record.has_rho = true;
            record.rho = step.rho;
        }
        record.final_criterion = step.final_criterion;
        record.x_breve_rank = step.state.x_breve_rank;
        record.intercept_adjustment = step.intercept_adjustment;
        record.intercept_raw = step.intercept_raw;
    } else {
        record.failure_reason = failure_reason_override.empty()
                                    ? step.failure_reason
                                    : failure_reason_override;
        record.step_failure_reason = step.failure_reason;
    }
    return record;
}

I2SLSNoFEPathResultCore solve_i2sls_nofe_tau_path_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& x_raw,
    const Eigen::Ref<const Eigen::MatrixXd>& z_full,
    const Eigen::Ref<const Eigen::VectorXd>& beta_anchor,
    double rho_anchor,
    double tau_target,
    const GroupBundleCore* transform_bundle,
    double transform_tol,
    int transform_maxiter,
    const I2SLSControlCore& control) {
    if (y.size() != x_raw.rows() || y.size() != z_full.rows()) {
        throw std::invalid_argument("i2SLS endog/exog/Z rows do not match");
    }
    if (beta_anchor.size() != x_raw.cols()) {
        throw std::invalid_argument("beta_anchor must match exog columns");
    }
    I2SLSNoFEPathResultCore out;
    Eigen::VectorXd beta_current = beta_anchor;
    double rho_current = rho_anchor;
    double tau_current = 2.0;
    const double delta = std::abs(tau_target - 2.0);
    const double min_dtau =
        control.has_min_dtau
            ? control.min_dtau
            : std::max(
                  delta / std::max(
                              static_cast<double>(control.tau_path_steps) * 128.0,
                              1.0),
                  1.0e-3);
    const double path_tol = std::max(0.1 * min_dtau, 1.0e-12);
    std::optional<double> failed_tau;
    std::string failure_reason;
    out.path.reserve(static_cast<std::size_t>(
        std::max(control.tau_path_steps, 1)));

    for (const double scheduled_goal : i2sls_tau_path_schedule_core(tau_target, control)) {
        double goal = scheduled_goal;
        while (std::abs(goal - tau_current) > path_tol) {
            if (out.n_path_evaluations >= control.max_path_evaluations) {
                failure_reason = "max_path_evaluations_reached";
                failed_tau = goal;
                break;
            }
            I2SLSPathStepCore step = i2sls_iterate_nofe_tau_step_core(
                y,
                x_raw,
                z_full,
                beta_current,
                goal,
                rho_current,
                transform_bundle,
                transform_tol,
                transform_maxiter,
                control);
            out.n_path_evaluations += 1;
            if (step.converged) {
                step.intercept_raw =
                    step.beta.size() > 0
                        ? step.beta(0)
                        : std::numeric_limits<double>::quiet_NaN();
                const auto recentered =
                    i2sls_recenter_intercept_core(y, x_raw, step.beta, goal);
                step.beta = recentered.first;
                step.intercept_adjustment = recentered.second;
                if (step.intercept_adjustment != 0.0) {
                    const bool can_update_inplace =
                        control.speed_i2sls_inplace_recenter &&
                        (transform_bundle == nullptr ||
                         transform_bundle->groups.empty());
                    if (!can_update_inplace ||
                        !i2sls_recenter_path_state_inplace_core(
                            step.state,
                            step.intercept_adjustment,
                            goal)) {
                        const I2SLSPathStateCore recentered_state =
                            i2sls_path_state_for_beta_core(
                                y,
                                x_raw,
                                z_full,
                                step.beta,
                                goal,
                                transform_bundle,
                                transform_tol,
                                transform_maxiter,
                                control);
                        if (recentered_state.valid) {
                            step.state = recentered_state;
                        }
                    }
                }
                beta_current = step.beta;
                rho_current = step.rho;
                out.path.push_back(
                    i2sls_make_path_record_core(goal, tau_current, step));
                tau_current = goal;
                out.n_path_successes += 1;
                break;
            }
            out.n_path_failures += 1;
            if (std::abs(goal - tau_current) <= min_dtau) {
                failure_reason = "min_dtau_reached";
                failed_tau = goal;
                out.path.push_back(
                    i2sls_make_path_record_core(
                        goal,
                        tau_current,
                        step,
                        failure_reason));
                break;
            }
            goal = tau_current + 0.5 * (goal - tau_current);
        }
        if (!failure_reason.empty()) {
            break;
        }
    }

    out.beta = beta_current;
    out.tau_attained = tau_current;
    out.converged =
        failure_reason.empty() && std::abs(tau_current - tau_target) <= path_tol;
    if (failed_tau.has_value()) {
        out.has_tau_failed = true;
        out.tau_failed = *failed_tau;
    }
    out.failure_reason = failure_reason;
    if (std::isfinite(rho_current)) {
        out.has_final_rho = true;
        out.final_rho = rho_current;
    }
    out.min_dtau = min_dtau;
    return out;
}

py::dict i2sls_nofe_result_to_dict(const I2SLSNoFEResultCore& result) {
    py::dict out;
    out["kernel_backend"] = "cpp";
    out["cpp_symbol"] = "solve_i2sls_nofe_tau2";
    out["converged"] = result.converged;
    out["failure_reason"] =
        result.failure_reason.empty() ? py::none() : py::cast(result.failure_reason);
    out["beta"] = vector_to_array(result.beta);
    out["predictions_omitted"] = result.predictions_omitted;
    out["fittedvalues"] = result.predictions_omitted
                               ? py::array_t<double>(0)
                               : vector_to_array(result.fitted);
    out["resid_response"] = result.predictions_omitted
                                 ? py::array_t<double>(0)
                                 : vector_to_array(result.residual);
    py::dict diagnostics;
    diagnostics["first_stage_rank"] = result.first_stage_rank;
    diagnostics["x_breve_rank"] = result.x_breve_rank;
    diagnostics["z_full_rank"] = result.z_full_rank;
    diagnostics["z_full_columns"] = result.z_full_columns;
    diagnostics["phase1_iters"] = result.phase1_iters;
    diagnostics["phase1_accepted"] = result.phase1_accepted;
    diagnostics["phase2_iters"] = result.phase2_iters;
    diagnostics["phase2_restarts"] = result.phase2_restarts;
    diagnostics["final_rho"] = std::isfinite(result.final_rho)
                                   ? py::cast(result.final_rho)
                                   : py::none();
    diagnostics["final_criterion"] = result.final_criterion;
    diagnostics["anderson_m_used"] = result.anderson_m_used;
    diagnostics["score_residual_max_abs_scaled"] =
        result.score_residual_max_abs_scaled;
    diagnostics["score_tolerance"] = result.score_tolerance;
    diagnostics["score_validation_status"] = result.score_validation_status;
    py::list rho_path;
    for (const double rho : result.rho_path) {
        rho_path.append(rho);
    }
    diagnostics["rho_path"] = rho_path;
    if (result.score_validation_status == "omitted_speed_compact_diagnostics") {
        diagnostics["kappa_trajectory_status"] =
            "omitted_speed_compact_diagnostics";
    } else {
        py::list kappa_trajectory;
        for (const double value : result.kappa_trajectory) {
            kappa_trajectory.append(value);
        }
        diagnostics["kappa_trajectory"] = kappa_trajectory;
    }
    out["diagnostics"] = diagnostics;
    return out;
}

py::dict i2sls_path_record_to_dict(const I2SLSPathRecordCore& record) {
    py::dict out;
    out["tau"] = record.tau;
    out["from_tau"] = record.has_from_tau ? py::cast(record.from_tau) : py::none();
    out["converged"] = record.converged;
    if (record.has_beta) {
        out["beta"] = vector_to_array(record.beta);
    } else {
        out["beta"] = py::none();
    }
    out["rho"] = record.has_rho ? py::cast(record.rho) : py::none();
    out["iterations"] = record.iterations;
    out["failure_reason"] =
        record.failure_reason.empty() ? py::none() : py::cast(record.failure_reason);
    py::dict diagnostics;
    if (std::isfinite(record.final_criterion)) {
        diagnostics["final_criterion"] = record.final_criterion;
    }
    if (record.x_breve_rank > 0) {
        diagnostics["x_breve_rank"] = record.x_breve_rank;
    }
    diagnostics["intercept_recentered"] = record.intercept_adjustment != 0.0;
    diagnostics["intercept_recenter_adjustment"] = record.intercept_adjustment;
    if (std::isfinite(record.intercept_raw) && record.intercept_adjustment != 0.0) {
        diagnostics["intercept_raw"] = record.intercept_raw;
    }
    if (!record.step_failure_reason.empty()) {
        diagnostics["step_failure_reason"] = record.step_failure_reason;
    }
    out["diagnostics"] = diagnostics;
    return out;
}

py::dict i2sls_nofe_tau_path_to_dict(const I2SLSNoFEPathResultCore& result) {
    py::dict out;
    out["kernel_backend"] = "cpp";
    out["cpp_symbol"] = "solve_i2sls_nofe_tau_path";
    out["converged"] = result.converged;
    out["beta"] = vector_to_array(result.beta);
    out["tau_attained"] = result.tau_attained;
    out["tau_failed"] =
        result.has_tau_failed ? py::cast(result.tau_failed) : py::none();
    out["failure_reason"] =
        result.failure_reason.empty() ? py::none() : py::cast(result.failure_reason);
    out["final_rho"] =
        result.has_final_rho ? py::cast(result.final_rho) : py::none();
    out["min_dtau"] = result.min_dtau;
    out["n_path_evaluations"] = result.n_path_evaluations;
    out["n_path_successes"] = result.n_path_successes;
    out["n_path_failures"] = result.n_path_failures;
    py::list records;
    for (const I2SLSPathRecordCore& record : result.path) {
        records.append(i2sls_path_record_to_dict(record));
    }
    out["path"] = py::tuple(records);
    return out;
}

py::dict solve_i2sls_nofe_tau2_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    py::array_t<double, py::array::c_style | py::array::forcecast> z_full,
    py::object control_obj,
    py::object x_breve_obj,
    py::object transform_groups_obj,
    py::object transform_counts_obj,
    double transform_tol,
    int transform_maxiter) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    const RowMatrixXd z_full_mat = matrix_from_array(z_full, "z_full");
    const I2SLSControlCore control = i2sls_control_from_mapping(control_obj);
    if (x_breve_obj.is_none()) {
        throw std::invalid_argument("x_breve is required");
    }
    const RowMatrixXd x_breve_mat = matrix_from_array(
        py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
            x_breve_obj),
        "x_breve");
    std::optional<GroupBundleCore> transform_bundle;
    if (transform_groups_obj.is_none() != transform_counts_obj.is_none()) {
        throw std::invalid_argument(
            "i2sls transform_groups and transform_counts must be supplied together");
    }
    if (!transform_groups_obj.is_none() && !transform_counts_obj.is_none()) {
        py::sequence groups_seq = py::cast<py::sequence>(transform_groups_obj);
        if (py::len(groups_seq) > 0) {
            transform_bundle = group_bundle_from_objects(
                transform_groups_obj,
                transform_counts_obj,
                endog_vec.size(),
                "i2sls transform groups may be empty only for identity topology");
        } else if (py::len(py::cast<py::sequence>(transform_counts_obj)) > 0) {
            throw std::invalid_argument(
                "i2sls transform groups/counts length mismatch");
        }
    }
    I2SLSNoFEResultCore result;
    {
        py::gil_scoped_release release;
        result = solve_i2sls_nofe_tau2_core(
            endog_vec,
            exog_mat,
            z_full_mat,
            x_breve_mat,
            transform_bundle.has_value() ? &(*transform_bundle) : nullptr,
            transform_tol,
            transform_maxiter,
            control);
    }
    return i2sls_nofe_result_to_dict(result);
}

py::dict solve_i2sls_nofe_tau_path_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    py::array_t<double, py::array::c_style | py::array::forcecast> z_full,
    double tau_target,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta_anchor,
    py::object rho_anchor_obj,
    py::object control_obj,
    py::object transform_groups_obj,
    py::object transform_counts_obj,
    double transform_tol,
    int transform_maxiter) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    const RowMatrixXd z_full_mat = matrix_from_array(z_full, "z_full");
    const Eigen::VectorXd beta_anchor_vec =
        vector_from_array(beta_anchor, "beta_anchor");
    double rho_anchor = std::numeric_limits<double>::quiet_NaN();
    if (!rho_anchor_obj.is_none()) {
        rho_anchor = py::cast<double>(rho_anchor_obj);
    }
    const I2SLSControlCore control = i2sls_control_from_mapping(control_obj);
    std::optional<GroupBundleCore> transform_bundle;
    if (transform_groups_obj.is_none() != transform_counts_obj.is_none()) {
        throw std::invalid_argument(
            "i2sls transform_groups and transform_counts must be supplied together");
    }
    if (!transform_groups_obj.is_none() && !transform_counts_obj.is_none()) {
        py::sequence groups_seq = py::cast<py::sequence>(transform_groups_obj);
        if (py::len(groups_seq) > 0) {
            transform_bundle = group_bundle_from_objects(
                transform_groups_obj,
                transform_counts_obj,
                endog_vec.size(),
                "i2sls transform groups may be empty only for identity topology");
        } else if (py::len(py::cast<py::sequence>(transform_counts_obj)) > 0) {
            throw std::invalid_argument(
                "i2sls transform groups/counts length mismatch");
        }
    }
    I2SLSNoFEPathResultCore result;
    {
        py::gil_scoped_release release;
        result = solve_i2sls_nofe_tau_path_core(
            endog_vec,
            exog_mat,
            z_full_mat,
            beta_anchor_vec,
            rho_anchor,
            tau_target,
            transform_bundle.has_value() ? &(*transform_bundle) : nullptr,
            transform_tol,
            transform_maxiter,
            control);
    }
    return i2sls_nofe_tau_path_to_dict(result);
}

py::dict backend_info() {
    py::dict out;
    out["kernel_backend"] = "cpp";
    out["python_reference_backend"] = "python_reference";
    out["i2sls_status"] = "transformed_tau2_and_tau_path_available";
#ifdef PYIOLS_HAVE_SCIPY_OPENBLAS64
    out["linear_solver"] =
        "NumPy OpenBLAS64 LAPACK DGELSD minimum-norm least squares";
#elif defined(PYIOLS_HAVE_LAPACK)
    out["linear_solver"] = "LAPACK DGELSD minimum-norm least squares";
#else
    out["linear_solver"] = "Eigen::JacobiSVD minimum-norm least squares";
#endif
    out["input_ownership"] = "snapshot_copy";
    out["gil_policy"] = "release_around_core_solve";
    return out;
}

struct FullrecoverControlCore {
    int anderson_m = 5;
    double eps = 1e-6;
    int stopcrit = 10000;
    int tau_path_steps = 20;
    bool has_min_dtau = false;
    double min_dtau = 0.0;
    int max_path_evaluations = 10000;
    int max_phase1_iterations = 10000;
    int max_phase2_evaluations = 50000;
    int fe_demean_maxiter = 10000;
    double fe_demean_tol = 1e-10;
    double delta_log_min = -5.0;
    double delta_log_max = 5.0;
    int delta_grid_size = 5;
    double rho_log_min = -7.0;
    double rho_log_max = 14.0;
    int rho_grid_size = 150;
    // runtime_v3: Phase 2 restart start policy, read ONLY by the direct no-FE
    // fixed-tau route ("phase1_start" = runtime_v2 behaviour,
    // "best_score_iterate" = restart from / return the best preconditioned
    // score iterate). Legacy callers never read it.
    std::string direct_restart_start = "phase1_start";
    bool research_tiered_fe_tol = false;
    double research_inner_fe_demean_tol = 1e-8;
    bool speed_fullrecover_joint_anderson = false;
    bool speed_fullrecover_accept_score_warning_path_steps = false;
    bool speed_small_manyway_exact_demean = false;
    bool speed_compact_fixed_tau_details = false;
    int speed_small_manyway_exact_demean_max_rows = 1200;
    int speed_small_manyway_exact_demean_max_levels = 512;
};

bool cpp_env_flag_enabled_core(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string raw(value);
    return raw == "1" || raw == "true" || raw == "TRUE" || raw == "yes" ||
           raw == "YES" || raw == "on" || raw == "ON" || raw == "cpp" ||
           raw == "CPP";
}

double cpp_env_double_or_core(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed) || parsed <= 0.0) {
        return fallback;
    }
    return parsed;
}

int cpp_env_positive_int_or_core(const char* name, int fallback) {
    return std::max(
        1,
        static_cast<int>(std::lround(
            cpp_env_double_or_core(name, static_cast<double>(fallback)))));
}

void apply_cpp_speed_research_env_core(FullrecoverControlCore& control) {
    control.speed_fullrecover_joint_anderson =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_FULLRECOVER_JOINT_ANDERSON");
    control.speed_fullrecover_accept_score_warning_path_steps =
        cpp_env_flag_enabled_core(
            "PYIOLS_CPP_SPEED_FULLRECOVER_ACCEPT_SCORE_WARNING_PATH_STEP");
    control.speed_small_manyway_exact_demean =
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_SMALL_MANYWAY_EXACT_DEMEAN");
    control.speed_compact_fixed_tau_details =
        cpp_env_flag_enabled_core("PYIOLS_SPEED_COMPACT_FIXED_TAU_DETAILS");
    control.speed_small_manyway_exact_demean_max_rows =
        cpp_env_positive_int_or_core(
            "PYIOLS_CPP_SPEED_SMALL_MANYWAY_EXACT_DEMEAN_MAX_ROWS",
            control.speed_small_manyway_exact_demean_max_rows);
    control.speed_small_manyway_exact_demean_max_levels =
        cpp_env_positive_int_or_core(
            "PYIOLS_CPP_SPEED_SMALL_MANYWAY_EXACT_DEMEAN_MAX_LEVELS",
            control.speed_small_manyway_exact_demean_max_levels);
    if (!cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_TIERED_FE_TOL") &&
        !cpp_env_flag_enabled_core("PYIOLS_CPP_TIERED_FE_DEMEAN")) {
        return;
    }
    control.research_tiered_fe_tol = true;
    const double requested = cpp_env_double_or_core(
        "PYIOLS_CPP_SPEED_TIERED_FE_INNER_TOL",
        control.research_inner_fe_demean_tol);
    control.research_inner_fe_demean_tol =
        std::max(requested, control.fe_demean_tol);
}

int phase2_weighted_demean_maxiter_core(const FullrecoverControlCore& control) {
    const int budget = std::max(control.max_phase2_evaluations, 1);
    const int bounded = std::max(budget / 500, 100);
    return std::max(1, std::min(control.fe_demean_maxiter, bounded));
}

// runtime_v3: one Phase 2 rho attempt of the direct no-FE fixed-tau route
// (twin of the NumPy phase2_attempt_trace entries).
struct Phase2AttemptTraceCore {
    int attempt = 0;
    double rho = 0.0;
    int iterations = 0;
    int evaluations = 0;
    std::string failure_reason;  // empty = converged
    bool has_final_criterion = false;
    double final_criterion = 0.0;
    double best_criterion = std::numeric_limits<double>::quiet_NaN();
    double kappa_last = std::numeric_limits<double>::quiet_NaN();
    std::string start_policy = "phase1_start";
    // runtime_v4 progress-monitor fields (twins of the NumPy trace keys).
    double final_progress_residual = std::numeric_limits<double>::quiet_NaN();
    double best_progress_residual = std::numeric_limits<double>::quiet_NaN();
    int anderson_m_start = 0;
    int anderson_m_end = 0;
    int anderson_accepted = 0;
};

struct FullrecoverIterationCore {
    bool converged = false;
    std::string phase = "not_started";
    int iterations = 0;
    int restarts = 0;
    bool has_final_rho = false;
    double final_rho = 0.0;
    std::vector<double> rho_path;
    bool has_final_criterion = false;
    double final_criterion = 0.0;
    int anderson_m_initial = 5;
    int anderson_m_final = 5;
    std::string failure_reason;
    int phase1_attempts = 0;
    int phase1_accepted = 0;
    int phase1_rejected = 0;
    int phase1_evaluations = 0;  // runtime_v2: profiled core only
    int phase2_evaluations = 0;
    int max_phase2_evaluations = 0;
    double last_finite_criterion = std::numeric_limits<double>::quiet_NaN();
    double best_finite_criterion = std::numeric_limits<double>::quiet_NaN();
    std::string rho_policy;
    double fullcancel_initial_scale = std::numeric_limits<double>::quiet_NaN();
    int rho_candidate_count = 0;
    bool has_rho_hint = false;
    double rho_hint = std::numeric_limits<double>::quiet_NaN();
    double rho_log_min = std::numeric_limits<double>::quiet_NaN();
    double rho_log_max = std::numeric_limits<double>::quiet_NaN();
    int rho_grid_size = 0;
    bool has_rho_min = false;
    double rho_min = std::numeric_limits<double>::quiet_NaN();
    bool has_rho_max = false;
    double rho_max = std::numeric_limits<double>::quiet_NaN();
    std::string algorithm;
    std::string index_policy;
    std::string fe_policy;
    std::string fe_state_policy;
    std::string anderson_policy;
    double qhat_last = std::numeric_limits<double>::quiet_NaN();
    int qhat_invalid_count = 0;
    double kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
    double kappa_monitor_max = std::numeric_limits<double>::quiet_NaN();
    int kappa_monitor_checks = 0;
    double post_convergence_residual = std::numeric_limits<double>::quiet_NaN();
    double post_convergence_kappa_est = std::numeric_limits<double>::quiet_NaN();
    double post_convergence_score_residual = std::numeric_limits<double>::quiet_NaN();
    double post_convergence_fe_score_residual = std::numeric_limits<double>::quiet_NaN();
    double score_tolerance = std::numeric_limits<double>::quiet_NaN();
    std::string score_validation_policy = "diagnostic";
    std::string score_validation_status = "not_checked";
    std::vector<std::string> score_warning_codes;
    std::vector<std::string> score_warning_messages;
    bool post_convergence_score_exceeds_tolerance = false;
    bool post_convergence_fe_score_exceeds_tolerance = false;
    int joint_hdfe_phi_calls = 0;
    int joint_hdfe_phi_demean_sweeps = 0;
    bool research_tiered_fe_tol = false;
    double research_inner_fe_demean_tol =
        std::numeric_limits<double>::quiet_NaN();
    int phase2_rho_restart_events = 0;
    int phase2_rho_attempts = 0;
    int mixed_compressed_cached_psd_solves = 0;
    int mixed_compressed_fast_spd_solves = 0;
    int mixed_compressed_structural_basis_solves = 0;
    int mixed_compressed_solver_failures = 0;
    std::vector<double> fullcancel_rho_attempt_values;
    std::vector<int> fullcancel_rho_attempt_iterations;
    std::vector<int> fullcancel_rho_attempt_restarts;
    std::vector<int> fullcancel_rho_attempt_phase2_evaluations;
    std::vector<double> fullcancel_rho_attempt_final_criteria;
    std::vector<double> fullcancel_rho_attempt_post_residuals;
    std::vector<double> fullcancel_rho_attempt_post_score_residuals;
    std::vector<std::string> fullcancel_rho_attempt_failure_reasons;
    std::vector<bool> fullcancel_rho_attempt_converged;
    // runtime_v3 direct no-FE fixed-tau route only (profiled core with
    // direct_nofe_fixed_tau = true); defaults are untouched on legacy routes.
    std::string direct_restart_start;
    std::string nonconverged_iterate_policy;
    double kappa_coordinate_row_max = std::numeric_limits<double>::quiet_NaN();
    double best_score_norm = std::numeric_limits<double>::infinity();
    double best_score_rho = std::numeric_limits<double>::quiet_NaN();
    int phase2_warm_restarts = 0;
    int phase2_attempts = 0;
    std::vector<Phase2AttemptTraceCore> phase2_attempt_trace;
    int post_convergence_invalid_map_count = 0;  // runtime_v4, Phase 2
    int phase1_post_convergence_invalid_map_count = 0;  // runtime_v4, Phase 1
};

struct FullrecoverConcentratedCore {
    Eigen::VectorXd beta;
    Eigen::VectorXd linear;
    bool has_linear = false;
    FullrecoverIterationCore iteration;
};

struct TauPathRecordCore {
    double tau = 2.0;
    bool has_from_tau = false;
    double from_tau = 0.0;
    bool has_step = false;
    double step = 0.0;
    bool converged = false;
    Eigen::VectorXd beta;
    bool has_beta = false;
    bool has_rho = false;
    double rho = 0.0;
    int iterations = 0;
    int restarts = 0;
    int anderson_m_initial = 5;
    int anderson_m_final = 5;
    std::string failure_reason;
    bool has_beta_init_source_tau = false;
    double beta_init_source_tau = 0.0;
    bool has_rho_hint = false;
    double rho_hint = 0.0;
    Eigen::VectorXd linear;
    bool has_linear = false;
    FullrecoverIterationCore diagnostics;
};

struct FullrecoverTauPathCore {
    Eigen::VectorXd beta;
    Eigen::VectorXd linear;
    bool has_linear = false;
    double tau_fit = 2.0;
    bool converged = false;
    FullrecoverIterationCore iteration;
    std::string path_status = "not_started";
    double tau_anchor = 2.0;
    double tau_target = 2.0;
    double tau_attained = 2.0;
    bool has_tau_failed = false;
    double tau_failed = 0.0;
    bool has_tau_gap = false;
    double tau_gap = 0.0;
    std::string failure_reason;
    bool has_final_rho = false;
    double final_rho = 0.0;
    bool has_min_dtau = false;
    double min_dtau = 0.0;
    int n_path_evaluations = 0;
    int n_path_successes = 0;
    int n_path_failures = 0;
    bool speed_accept_score_warning_path_steps = false;
    std::vector<TauPathRecordCore> path;
};

bool all_finite_vec(const Eigen::Ref<const Eigen::VectorXd>& values) {
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        if (!std::isfinite(values(idx))) {
            return false;
        }
    }
    return true;
}

bool all_finite_mat(const Eigen::Ref<const Eigen::MatrixXd>& values) {
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        for (Eigen::Index col = 0; col < values.cols(); ++col) {
            if (!std::isfinite(values(row, col))) {
                return false;
            }
        }
    }
    return true;
}

double max_abs_mat_diff(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
    if (left.size() == 0 && right.size() == 0) {
        return 0.0;
    }
    return (left - right).cwiseAbs().maxCoeff();
}

double exp_guard_scalar(double value) {
    if (std::isinf(value) && value > 0.0) {
        return kFloat64Max;
    }
    if (value > kFloat64LogMax) {
        return kFloat64Max;
    }
    return std::exp(value);
}

Eigen::VectorXd exp_guard_vector(const Eigen::Ref<const Eigen::VectorXd>& values) {
    Eigen::VectorXd out(values.size());
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        out(idx) = exp_guard_scalar(values(idx));
    }
    return out;
}

double logsumexp_all_core(const Eigen::Ref<const Eigen::VectorXd>& values) {
    if (values.size() == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    bool has_posinf = false;
    bool has_finite = false;
    double max_value = -std::numeric_limits<double>::infinity();
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        const double value = values(idx);
        if (std::isinf(value) && value > 0.0) {
            has_posinf = true;
        }
        if (std::isfinite(value)) {
            has_finite = true;
            max_value = std::max(max_value, value);
        }
    }
    if (has_posinf) {
        return std::numeric_limits<double>::infinity();
    }
    if (!has_finite) {
        return -std::numeric_limits<double>::infinity();
    }
    double sum = 0.0;
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        const double value = values(idx);
        if (std::isfinite(value)) {
            sum += std::exp(value - max_value);
        }
    }
    return max_value + std::log(sum);
}

bool is_singleton_intercept_group(
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    if (n_groups != 1) {
        return false;
    }
    for (const std::int64_t group : groups) {
        if (group != 0) {
            return false;
        }
    }
    return true;
}

double profile_intercept_adjustment_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const Eigen::VectorXd>& linear,
    double tau) {
    Eigen::VectorXd numerator =
        Eigen::VectorXd::Constant(linear.size(), -std::numeric_limits<double>::infinity());
    Eigen::VectorXd denominator(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        if (endog(row) > 0.0) {
            numerator(row) = std::log(endog(row)) + (1.0 - tau) * linear(row);
        }
        denominator(row) = (2.0 - tau) * linear(row);
    }
    const double log_num = logsumexp_all_core(numerator);
    const double log_den = logsumexp_all_core(denominator);
    if (std::isfinite(log_num) && std::isfinite(log_den)) {
        return log_num - log_den;
    }
    return 0.0;
}

Eigen::VectorXd group_logsumexp_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    Eigen::VectorXd max_by_group =
        Eigen::VectorXd::Constant(n_groups, -std::numeric_limits<double>::infinity());
    std::vector<bool> has_nan(static_cast<std::size_t>(n_groups), false);
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        const auto group = static_cast<Eigen::Index>(groups[static_cast<std::size_t>(row)]);
        const double value = values(row);
        if (std::isnan(value)) {
            has_nan[static_cast<std::size_t>(group)] = true;
            continue;
        }
        if (value > max_by_group(group)) {
            max_by_group(group) = value;
        }
    }
    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_groups, -std::numeric_limits<double>::infinity());
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        const auto group = static_cast<Eigen::Index>(groups[static_cast<std::size_t>(row)]);
        const double max_value = max_by_group(group);
        const double value = values(row);
        if (has_nan[static_cast<std::size_t>(group)] || !std::isfinite(max_value)) {
            continue;
        }
        if (std::isfinite(value)) {
            sums(group) += std::exp(value - max_value);
        }
    }
    for (Eigen::Index group = 0; group < n_groups; ++group) {
        if (has_nan[static_cast<std::size_t>(group)]) {
            continue;
        }
        if (std::isinf(max_by_group(group)) && max_by_group(group) > 0.0) {
            out(group) = std::numeric_limits<double>::infinity();
        } else if (std::isfinite(max_by_group(group)) && sums(group) > 0.0) {
            out(group) = max_by_group(group) + std::log(sums(group));
        }
    }
    return out;
}

Eigen::VectorXd profile_one_effect_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const Eigen::VectorXd>& partial_linear,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau) {
    Eigen::VectorXd numerator =
        Eigen::VectorXd::Constant(partial_linear.size(), -std::numeric_limits<double>::infinity());
    Eigen::VectorXd denominator(partial_linear.size());
    for (Eigen::Index row = 0; row < partial_linear.size(); ++row) {
        if (endog(row) > 0.0) {
            numerator(row) =
                std::log(endog(row)) + (1.0 - tau) * partial_linear(row);
        }
        denominator(row) = (2.0 - tau) * partial_linear(row);
    }
    const Eigen::VectorXd log_num = group_logsumexp_core(numerator, groups, n_groups);
    const Eigen::VectorXd log_den = group_logsumexp_core(denominator, groups, n_groups);
    Eigen::VectorXd effect = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index group = 0; group < effect.size(); ++group) {
        if (std::isfinite(log_num(group)) && std::isfinite(log_den(group))) {
            effect(group) = log_num(group) - log_den(group);
        }
    }
    return effect;
}

Eigen::VectorXd profile_linear_profiled_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau) {
    Eigen::VectorXd base = exog * beta;
    if (is_singleton_intercept_group(groups, n_groups)) {
        const double adjustment = profile_intercept_adjustment_core(endog, base, tau);
        return Eigen::VectorXd(base.array() + adjustment);
    }
    const Eigen::VectorXd effect =
        profile_one_effect_core(endog, base, groups, n_groups, tau);
    Eigen::VectorXd linear(base.size());
    for (Eigen::Index row = 0; row < base.size(); ++row) {
        linear(row) = base(row) + effect(groups[static_cast<std::size_t>(row)]);
    }
    return linear;
}

Eigen::MatrixXd demean_singleton_intercept_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values) {
    if (values.size() == 0) {
        return Eigen::MatrixXd(values);
    }
    Eigen::RowVectorXd mean = values.colwise().mean();
    return values.rowwise() - mean;
}

Eigen::MatrixXd demean_profiled_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    if (is_singleton_intercept_group(groups, n_groups)) {
        return demean_singleton_intercept_core(values);
    }
    return demean_oneway_matrix(values, groups, n_groups, std::nullopt);
}

Eigen::MatrixXd demean_for_group_bundle_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const GroupBundleCore& bundle,
    const std::optional<Eigen::VectorXd>& weights,
    double tol,
    int maxiter);

struct MixedDesignCore {
    Eigen::MatrixXd design;
    std::vector<std::string> recover_column_names;
    bool gil_was_released = false;
    bool input_snapshot_owned = true;
};

struct DesignRankDiagnosticsCore {
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;
    Eigen::Index rank = 0;
    double condition_number = std::numeric_limits<double>::quiet_NaN();
    double min_singular_value = std::numeric_limits<double>::quiet_NaN();
    double max_singular_value = std::numeric_limits<double>::quiet_NaN();
    double rank_tolerance = std::numeric_limits<double>::quiet_NaN();
    std::string svd_status;
    std::string rank_method;
};

struct MixedRecoverCellDiagnosticsCore {
    int total_cells = 0;
    int min_cell_size = 0;
    int max_cell_size = 0;
    int min_positive_count = 0;
    int all_zero_cell_count = 0;
    int one_positive_cell_count = 0;
    std::vector<std::string> all_zero_cells;
    std::vector<std::string> one_positive_cells;
};

struct MixedSupportPreflightCore {
    MixedRecoverCellDiagnosticsCore recover_cells;
    DesignRankDiagnosticsCore raw_rank;
    Eigen::MatrixXd cancel_quotient;
    bool has_cancel_quotient = false;
    std::string cancel_quotient_status = "not_computed";
    std::string cancel_quotient_error;
    DesignRankDiagnosticsCore cancel_quotient_rank;
    bool gil_was_released = false;
    bool input_snapshot_owned = true;
};

template <typename Derived>
std::optional<DesignRankDiagnosticsCore> design_rank_diagnostics_gram_fast_path(
    const Eigen::MatrixBase<Derived>& design_base) {
    constexpr Eigen::Index kMinFastPathCols = 32;
    constexpr double kMaxFastPathConditionNumber = 1.0e4;
    constexpr double kMinSingularToleranceMultiple = 1.0e4;
    if (design_base.rows() < design_base.cols() ||
        design_base.cols() < kMinFastPathCols) {
        return std::nullopt;
    }
    const Eigen::MatrixXd gram = design_base.transpose() * design_base;
    if (!gram.allFinite()) {
        return std::nullopt;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        gram,
        Eigen::EigenvaluesOnly);
    if (spectrum.info() != Eigen::Success) {
        return std::nullopt;
    }
    const Eigen::VectorXd eigenvalues = spectrum.eigenvalues();
    if (eigenvalues.size() == 0) {
        return std::nullopt;
    }
    const double max_eigen = eigenvalues.maxCoeff();
    const double min_eigen = eigenvalues.minCoeff();
    if (!std::isfinite(max_eigen) || !std::isfinite(min_eigen) ||
        max_eigen <= 0.0 || min_eigen <= 0.0) {
        return std::nullopt;
    }
    const double max_sv = std::sqrt(max_eigen);
    const double min_sv = std::sqrt(min_eigen);
    if (!std::isfinite(max_sv) || !std::isfinite(min_sv) || min_sv <= 0.0) {
        return std::nullopt;
    }
    const double condition_number = max_sv / min_sv;
    const double tol =
        static_cast<double>(std::max(design_base.rows(), design_base.cols())) *
        std::numeric_limits<double>::epsilon() * std::max(max_sv, 1.0);
    if (!std::isfinite(condition_number) ||
        condition_number >= kMaxFastPathConditionNumber ||
        min_sv <= kMinSingularToleranceMultiple * tol) {
        return std::nullopt;
    }
    DesignRankDiagnosticsCore out;
    out.rows = design_base.rows();
    out.cols = design_base.cols();
    out.rank = design_base.cols();
    out.condition_number = condition_number;
    out.min_singular_value = min_sv;
    out.max_singular_value = max_sv;
    out.rank_tolerance = tol;
    out.svd_status = "computed";
    out.rank_method = "gram_well_conditioned_full_rank";
    return out;
}

template <typename Derived>
DesignRankDiagnosticsCore design_rank_diagnostics_direct_svd_core(
    const Eigen::MatrixBase<Derived>& design_base) {
    DesignRankDiagnosticsCore out;
    out.rows = design_base.rows();
    out.cols = design_base.cols();
    out.rank_method = "direct_svd";
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Eigen::MatrixXd(design_base), 0);
    if (svd.info() != Eigen::Success) {
        out.svd_status = "failed";
        return out;
    }
    const Eigen::VectorXd singular_values = svd.singularValues();
    if (singular_values.size() == 0) {
        return out;
    }
    const double max_sv = singular_values(0);
    const double min_sv = singular_values(singular_values.size() - 1);
    if (!std::isfinite(max_sv) || !std::isfinite(min_sv)) {
        return out;
    }
    const double tol =
        static_cast<double>(std::max(design_base.rows(), design_base.cols())) *
        std::numeric_limits<double>::epsilon() * std::max(max_sv, 1.0);
    Eigen::Index rank = 0;
    for (Eigen::Index idx = 0; idx < singular_values.size(); ++idx) {
        if (singular_values(idx) > tol) {
            ++rank;
        }
    }
    out.rank = rank;
    out.condition_number =
        min_sv <= 0.0 ? std::numeric_limits<double>::infinity() : max_sv / min_sv;
    out.min_singular_value = min_sv;
    out.max_singular_value = max_sv;
    out.rank_tolerance = tol;
    out.svd_status = "computed";
    return out;
}

template <typename Derived>
DesignRankDiagnosticsCore design_rank_diagnostics_core(
    const Eigen::MatrixBase<Derived>& design_base) {
    DesignRankDiagnosticsCore out;
    out.rows = design_base.rows();
    out.cols = design_base.cols();
    if (design_base.rows() == 0 || design_base.cols() == 0 ||
        !design_base.allFinite()) {
        return out;
    }
    std::optional<DesignRankDiagnosticsCore> fast =
        design_rank_diagnostics_gram_fast_path(design_base);
    if (fast.has_value()) {
        return fast.value();
    }
    return design_rank_diagnostics_direct_svd_core(design_base);
}

MixedRecoverCellDiagnosticsCore mixed_recover_cell_diagnostics_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const GroupBundleCore& recover_bundle,
    const std::vector<std::string>& recover_names) {
    if (recover_bundle.groups.size() != recover_names.size()) {
        throw std::invalid_argument(
            "recover_names must have the same length as recover_groups");
    }
    MixedRecoverCellDiagnosticsCore out;
    out.all_zero_cells.reserve(20);
    out.one_positive_cells.reserve(20);
    bool have_min_cell_size = false;
    bool have_min_positive_count = false;
    for (std::size_t block = 0; block < recover_bundle.groups.size(); ++block) {
        const std::vector<std::int64_t>& group = recover_bundle.groups[block];
        const std::int64_t count = recover_bundle.counts[block];
        if (static_cast<Eigen::Index>(group.size()) != endog.size()) {
            throw std::invalid_argument(
                "recover FE blocks must have the same number of rows as endog");
        }
        for (std::int64_t level = 0; level < count; ++level) {
            int cell_size = 0;
            int positives = 0;
            for (Eigen::Index row = 0; row < endog.size(); ++row) {
                if (group[static_cast<std::size_t>(row)] != level) {
                    continue;
                }
                ++cell_size;
                if (endog(row) > 0.0) {
                    ++positives;
                }
            }
            if (cell_size == 0) {
                continue;
            }
            ++out.total_cells;
            out.max_cell_size = std::max(out.max_cell_size, cell_size);
            out.min_cell_size = have_min_cell_size
                                    ? std::min(out.min_cell_size, cell_size)
                                    : cell_size;
            out.min_positive_count = have_min_positive_count
                                         ? std::min(out.min_positive_count, positives)
                                         : positives;
            have_min_cell_size = true;
            have_min_positive_count = true;
            const std::string cell_name =
                recover_names[block] + "[" + std::to_string(level) + "]";
            if (positives == 0) {
                ++out.all_zero_cell_count;
                if (out.all_zero_cells.size() < 20) {
                    out.all_zero_cells.push_back(cell_name);
                }
            } else if (positives == 1) {
                ++out.one_positive_cell_count;
                if (out.one_positive_cells.size() < 20) {
                    out.one_positive_cells.push_back(cell_name);
                }
            }
        }
    }
    return out;
}

MixedDesignCore mixed_design_matrix_core(
    const RowMatrixXd& exog,
    const GroupBundleCore& recover_bundle,
    const std::vector<std::string>& recover_names) {
    if (recover_bundle.groups.empty()) {
        throw std::invalid_argument(
            "mixed topology requires at least one recover FE block");
    }
    if (recover_bundle.groups.size() != recover_names.size()) {
        throw std::invalid_argument(
            "recover_names must have the same length as recover_groups");
    }
    Eigen::Index recover_cols = 0;
    for (const std::int64_t count : recover_bundle.counts) {
        recover_cols += static_cast<Eigen::Index>(std::max<std::int64_t>(count - 1, 0));
    }

    MixedDesignCore out;
    out.gil_was_released = PyGILState_Check() == 0;
    out.input_snapshot_owned = true;
    out.recover_column_names.reserve(static_cast<std::size_t>(recover_cols));
    out.design = Eigen::MatrixXd::Zero(exog.rows(), exog.cols() + recover_cols);
    if (exog.cols() > 0) {
        out.design.leftCols(exog.cols()) = Eigen::MatrixXd(exog);
    }

    Eigen::Index col = exog.cols();
    for (std::size_t block = 0; block < recover_bundle.groups.size(); ++block) {
        const std::vector<std::int64_t>& group = recover_bundle.groups[block];
        const std::int64_t count = recover_bundle.counts[block];
        if (static_cast<Eigen::Index>(group.size()) != exog.rows()) {
            throw std::invalid_argument(
                "recover FE blocks must have the same number of rows as exog");
        }
        for (std::int64_t level = 1; level < count; ++level) {
            out.recover_column_names.push_back(
                recover_names[block] + "[" + std::to_string(level) + "]");
            for (Eigen::Index row = 0; row < exog.rows(); ++row) {
                out.design(row, col) =
                    group[static_cast<std::size_t>(row)] == level ? 1.0 : 0.0;
            }
            ++col;
        }
    }
    return out;
}

MixedSupportPreflightCore mixed_support_conditioning_preflight_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const RowMatrixXd& mixed_design,
    const GroupBundleCore& cancel_bundle,
    const GroupBundleCore& recover_bundle,
    const std::vector<std::string>& recover_names,
    const FullrecoverControlCore& control) {
    if (endog.size() != mixed_design.rows()) {
        throw std::invalid_argument(
            "endog and mixed_design must have the same rows");
    }
    MixedSupportPreflightCore out;
    out.gil_was_released = PyGILState_Check() == 0;
    out.input_snapshot_owned = true;
    out.recover_cells = mixed_recover_cell_diagnostics_core(
        endog,
        recover_bundle,
        recover_names);
    out.raw_rank = design_rank_diagnostics_core(mixed_design);
    try {
        out.cancel_quotient = demean_for_group_bundle_core(
            mixed_design,
            cancel_bundle,
            std::nullopt,
            control.fe_demean_tol,
            control.fe_demean_maxiter);
        out.has_cancel_quotient = true;
        out.cancel_quotient_status = "computed";
        out.cancel_quotient_rank =
            design_rank_diagnostics_core(out.cancel_quotient);
    } catch (const std::invalid_argument& exc) {
        out.cancel_quotient = Eigen::MatrixXd(mixed_design.rows(), 0);
        out.cancel_quotient_status = "error";
        out.cancel_quotient_error = std::string("ValueError: ") + exc.what();
        out.cancel_quotient_rank = design_rank_diagnostics_core(out.cancel_quotient);
    } catch (const std::runtime_error& exc) {
        out.cancel_quotient = Eigen::MatrixXd(mixed_design.rows(), 0);
        out.cancel_quotient_status = "error";
        out.cancel_quotient_error = std::string("RuntimeError: ") + exc.what();
        out.cancel_quotient_rank = design_rank_diagnostics_core(out.cancel_quotient);
    } catch (const std::exception& exc) {
        out.cancel_quotient = Eigen::MatrixXd(mixed_design.rows(), 0);
        out.cancel_quotient_status = "error";
        out.cancel_quotient_error = std::string("Exception: ") + exc.what();
        out.cancel_quotient_rank = design_rank_diagnostics_core(out.cancel_quotient);
    }
    return out;
}

constexpr double kBreadConditionLimit = 1.0e12;
constexpr double kLeverageFloor = 1.0e-8;

struct SandwichCovarianceCore {
    bool available = false;
    std::string reason;
    Eigen::VectorXd se;
    Eigen::MatrixXd covariance;
    double bread_condition = std::numeric_limits<double>::quiet_NaN();
    int n_clusters = -1;
    bool gil_was_released = false;
    bool input_snapshot_owned = true;
};

SandwichCovarianceCore sandwich_covariance_core(
    const Eigen::Ref<const RowMatrixXd>& design,
    const Eigen::Ref<const Eigen::VectorXd>& bread_weight,
    const Eigen::Ref<const Eigen::VectorXd>& score,
    const std::string& se_type,
    int df_denom,
    const std::optional<std::vector<std::int64_t>>& cluster_codes,
    std::int64_t n_clusters);

SandwichCovarianceCore sandwich_unavailable_core(
    Eigen::Index size,
    const std::string& reason,
    double bread_condition = std::numeric_limits<double>::quiet_NaN(),
    int n_clusters = -1) {
    SandwichCovarianceCore out;
    out.available = false;
    out.reason = reason;
    out.se = Eigen::VectorXd::Constant(
        size,
        std::numeric_limits<double>::quiet_NaN());
    out.covariance = Eigen::MatrixXd::Constant(
        size,
        size,
        std::numeric_limits<double>::quiet_NaN());
    out.bread_condition = bread_condition;
    out.n_clusters = n_clusters;
    return out;
}

struct MixedCovarianceProjectionCore {
    SandwichCovarianceCore covariance;
    Eigen::Index recover_columns = 0;
    Eigen::Index slope_columns = 0;
    std::string projection_method;
    bool gil_was_released = false;
    bool input_snapshot_owned = true;
};

MixedCovarianceProjectionCore mixed_covariance_projection_unavailable_core(
    Eigen::Index size,
    const std::string& reason) {
    MixedCovarianceProjectionCore out;
    out.covariance = sandwich_unavailable_core(size, reason);
    out.slope_columns = size;
    out.projection_method = "dense_recovered_fe_weighted_projection";
    out.gil_was_released = PyGILState_Check() == 0;
    out.input_snapshot_owned = true;
    return out;
}

Eigen::MatrixXd recover_design_droplast_core(
    const GroupBundleCore& recover_bundle,
    Eigen::Index n_rows) {
    Eigen::Index recover_cols = 0;
    for (const std::int64_t count : recover_bundle.counts) {
        recover_cols +=
            static_cast<Eigen::Index>(std::max<std::int64_t>(count - 1, 0));
    }
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_rows, recover_cols);
    Eigen::Index col = 0;
    for (std::size_t block = 0; block < recover_bundle.groups.size(); ++block) {
        const std::vector<std::int64_t>& group = recover_bundle.groups[block];
        const std::int64_t count = recover_bundle.counts[block];
        if (static_cast<Eigen::Index>(group.size()) != n_rows) {
            throw std::invalid_argument(
                "recover FE blocks must have the same number of rows as slopes");
        }
        for (std::int64_t level = 0; level < count - 1; ++level) {
            for (Eigen::Index row = 0; row < n_rows; ++row) {
                out(row, col) =
                    group[static_cast<std::size_t>(row)] == level ? 1.0 : 0.0;
            }
            ++col;
        }
    }
    return out;
}

Eigen::MatrixXd min_norm_lstsq_with_tolerance_core(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b,
    double tol) {
    if (a.rows() != b.rows()) {
        throw std::invalid_argument("least-squares row dimensions do not match");
    }
    if (a.cols() == 0) {
        return Eigen::MatrixXd::Zero(0, b.cols());
    }
    if (a.rows() == 0) {
        throw std::invalid_argument("least-squares matrix must have rows");
    }
    Eigen::BDCSVD<Eigen::MatrixXd> svd(
        Eigen::MatrixXd(a),
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.info() != Eigen::Success) {
        throw std::runtime_error("least-squares SVD failed");
    }
    const Eigen::VectorXd singular = svd.singularValues();
    const double max_sv = singular.size() == 0 ? 0.0 : singular.maxCoeff();
    const double cutoff = std::max(tol, 0.0) * std::max(max_sv, 1.0);
    Eigen::MatrixXd projected = svd.matrixU().transpose() * b;
    for (Eigen::Index row = 0; row < projected.rows(); ++row) {
        const double sv = singular(row);
        if (std::isfinite(sv) && sv > cutoff) {
            projected.row(row) /= sv;
        } else {
            projected.row(row).setZero();
        }
    }
    return svd.matrixV() * projected;
}

MixedCovarianceProjectionCore mixed_covariance_projection_core(
    const RowMatrixXd& x_slopes,
    const GroupBundleCore& cancel_bundle,
    const GroupBundleCore& recover_bundle,
    const Eigen::Ref<const Eigen::VectorXd>& demean_weights,
    const Eigen::Ref<const Eigen::VectorXd>& bread_weight,
    const Eigen::Ref<const Eigen::VectorXd>& score,
    const std::string& se_type,
    int df_denom,
    const std::optional<std::vector<std::int64_t>>& cluster_codes,
    std::int64_t n_clusters,
    double tol,
    int maxiter) {
    const Eigen::Index n_obs = x_slopes.rows();
    const Eigen::Index k_slopes = x_slopes.cols();
    if (k_slopes == 0) {
        return mixed_covariance_projection_unavailable_core(
            k_slopes,
            "mixed_hc0_no_estimable_slopes");
    }
    if (demean_weights.size() != n_obs || bread_weight.size() != n_obs ||
        score.size() != n_obs) {
        return mixed_covariance_projection_unavailable_core(
            k_slopes,
            "shape_mismatch");
    }
    if (!all_finite_mat(x_slopes) || !all_finite_vec(demean_weights) ||
        !all_finite_vec(bread_weight) || !all_finite_vec(score)) {
        return mixed_covariance_projection_unavailable_core(
            k_slopes,
            "mixed_hc0_nonfinite_sandwich_inputs");
    }

    const Eigen::MatrixXd recover_design =
        recover_design_droplast_core(recover_bundle, n_obs);
    Eigen::MatrixXd design(n_obs, k_slopes + recover_design.cols());
    design.leftCols(k_slopes) = Eigen::MatrixXd(x_slopes);
    if (recover_design.cols() > 0) {
        design.rightCols(recover_design.cols()) = recover_design;
    }

    const std::optional<Eigen::VectorXd> weights = Eigen::VectorXd(demean_weights);
    const Eigen::MatrixXd design_absorbed =
        demean_for_group_bundle_core(design, cancel_bundle, weights, tol, maxiter);
    if (!all_finite_mat(design_absorbed)) {
        return mixed_covariance_projection_unavailable_core(
            k_slopes,
            "mixed_hc0_nonfinite_absorbed_design");
    }
    const Eigen::MatrixXd slope_design = design_absorbed.leftCols(k_slopes);
    const Eigen::MatrixXd recover_absorbed =
        design_absorbed.rightCols(recover_design.cols());

    Eigen::VectorXd root_w(n_obs);
    for (Eigen::Index row = 0; row < n_obs; ++row) {
        root_w(row) = std::sqrt(bread_weight(row));
    }
    if (!all_finite_vec(root_w)) {
        return mixed_covariance_projection_unavailable_core(
            k_slopes,
            "mixed_hc0_nonfinite_weight_root");
    }

    Eigen::MatrixXd slope_resid = slope_design;
    if (recover_absorbed.cols() > 0) {
        Eigen::MatrixXd weighted_recover = recover_absorbed;
        Eigen::MatrixXd weighted_slope = slope_design;
        for (Eigen::Index row = 0; row < n_obs; ++row) {
            weighted_recover.row(row) *= root_w(row);
            weighted_slope.row(row) *= root_w(row);
        }
        if (!all_finite_mat(weighted_recover)) {
            return mixed_covariance_projection_unavailable_core(
                k_slopes,
                "mixed_hc0_nonfinite_weighted_recover_design");
        }
        Eigen::MatrixXd gamma;
        try {
            gamma = min_norm_lstsq_with_tolerance_core(
                weighted_recover,
                weighted_slope,
                1.0e-10);
        } catch (const std::exception&) {
            return mixed_covariance_projection_unavailable_core(
                k_slopes,
                "mixed_hc0_recover_projection_failed");
        }
        slope_resid.noalias() = slope_design - recover_absorbed * gamma;
    }

    Eigen::MatrixXd weighted_slope_design = slope_resid;
    for (Eigen::Index row = 0; row < n_obs; ++row) {
        weighted_slope_design.row(row) *= root_w(row);
    }
    if (!all_finite_mat(weighted_slope_design)) {
        return mixed_covariance_projection_unavailable_core(
            k_slopes,
            "mixed_hc0_nonfinite_weighted_slope_design");
    }

    RowMatrixXd slope_resid_row = slope_resid;
    MixedCovarianceProjectionCore out;
    out.covariance = sandwich_covariance_core(
        slope_resid_row,
        bread_weight,
        score,
        se_type,
        df_denom,
        cluster_codes,
        n_clusters);
    out.recover_columns = recover_design.cols();
    out.slope_columns = k_slopes;
    out.projection_method = "dense_recovered_fe_weighted_projection";
    out.gil_was_released = PyGILState_Check() == 0;
    out.input_snapshot_owned = true;
    return out;
}

double matrix_condition_svd_core(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
    if (matrix.rows() != matrix.cols() || matrix.rows() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU);
    if (svd.info() != Eigen::Success) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::VectorXd singular_values = svd.singularValues();
    if (singular_values.size() == 0 || !all_finite_vec(singular_values)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double max_singular = singular_values.maxCoeff();
    const double min_singular = singular_values.minCoeff();
    if (!std::isfinite(max_singular) || !std::isfinite(min_singular)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (max_singular <= 0.0 || min_singular <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return max_singular / min_singular;
}

SandwichCovarianceCore sandwich_covariance_core(
    const Eigen::Ref<const RowMatrixXd>& design,
    const Eigen::Ref<const Eigen::VectorXd>& bread_weight,
    const Eigen::Ref<const Eigen::VectorXd>& score,
    const std::string& se_type,
    int df_denom,
    const std::optional<std::vector<std::int64_t>>& cluster_codes,
    std::int64_t n_clusters) {
    const Eigen::Index n_obs = design.rows();
    const Eigen::Index size = design.cols();
    if (bread_weight.size() != n_obs || score.size() != n_obs) {
        return sandwich_unavailable_core(size, "shape_mismatch");
    }
    if (df_denom < 1) {
        df_denom = 1;
    }
    if (size == 0) {
        return sandwich_unavailable_core(size, "no_estimable_parameters");
    }
    if (!all_finite_mat(design) || !all_finite_vec(bread_weight) ||
        !all_finite_vec(score)) {
        return sandwich_unavailable_core(size, "nonfinite_sandwich_inputs");
    }

    Eigen::MatrixXd bread = Eigen::MatrixXd::Zero(size, size);
    for (Eigen::Index row = 0; row < n_obs; ++row) {
        bread.noalias() +=
            bread_weight(row) * design.row(row).transpose() * design.row(row);
    }
    if (!all_finite_mat(bread)) {
        return sandwich_unavailable_core(size, "nonfinite_bread");
    }
    const double bread_condition = matrix_condition_svd_core(bread);
    if (!std::isfinite(bread_condition) ||
        bread_condition > kBreadConditionLimit) {
        return sandwich_unavailable_core(
            size,
            "ill_conditioned_bread",
            std::isfinite(bread_condition)
                ? bread_condition
                : std::numeric_limits<double>::quiet_NaN());
    }
    const Eigen::MatrixXd bread_inv = bread.inverse();
    if (!all_finite_mat(bread_inv)) {
        return sandwich_unavailable_core(
            size,
            "bread_inversion_failed",
            bread_condition);
    }

    Eigen::MatrixXd covariance(size, size);
    if (se_type == "iid") {
        double numerator = 0.0;
        bool any_positive_weight = false;
        for (Eigen::Index row = 0; row < n_obs; ++row) {
            if (bread_weight(row) > 0.0) {
                any_positive_weight = true;
                numerator += (score(row) * score(row)) / bread_weight(row);
            }
        }
        if (!any_positive_weight) {
            return sandwich_unavailable_core(
                size,
                "no_positive_bread_weight",
                bread_condition);
        }
        covariance.noalias() = (numerator / static_cast<double>(df_denom)) * bread_inv;
    } else {
        Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(size, size);
        if (se_type == "cluster") {
            if (!cluster_codes.has_value()) {
                return sandwich_unavailable_core(
                    size,
                    "cluster_variable_missing",
                    bread_condition);
            }
            if (static_cast<Eigen::Index>(cluster_codes->size()) != n_obs) {
                return sandwich_unavailable_core(
                    size,
                    "cluster_shape_mismatch",
                    bread_condition);
            }
            if (n_clusters < 2) {
                return sandwich_unavailable_core(
                    size,
                    "too_few_clusters",
                    bread_condition,
                    static_cast<int>(n_clusters));
            }
            Eigen::MatrixXd cluster_scores =
                Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_clusters), size);
            for (Eigen::Index row = 0; row < n_obs; ++row) {
                const Eigen::Index cluster =
                    static_cast<Eigen::Index>((*cluster_codes)[row]);
                cluster_scores.row(cluster) += score(row) * design.row(row);
            }
            meat.noalias() = cluster_scores.transpose() * cluster_scores;
            meat *= (static_cast<double>(n_clusters) /
                     static_cast<double>(std::max<std::int64_t>(n_clusters - 1, 1))) *
                    ((static_cast<double>(n_obs) - 1.0) /
                     static_cast<double>(df_denom));
        } else {
            for (Eigen::Index row = 0; row < n_obs; ++row) {
                double adjustment = 1.0;
                if (se_type == "hc2" || se_type == "hc3") {
                    double leverage =
                        bread_weight(row) *
                        (design.row(row) * bread_inv * design.row(row).transpose())(0, 0);
                    leverage = std::min(std::max(leverage, 0.0), 1.0 - kLeverageFloor);
                    adjustment = 1.0 / (1.0 - leverage);
                    if (se_type == "hc3") {
                        adjustment *= adjustment;
                    }
                }
                const double weighted_score = score(row) * score(row) * adjustment;
                meat.noalias() += weighted_score *
                                  design.row(row).transpose() * design.row(row);
            }
            if (se_type == "hc1") {
                meat *= static_cast<double>(n_obs) /
                        static_cast<double>(df_denom);
            }
        }
        if (!all_finite_mat(meat)) {
            return sandwich_unavailable_core(
                size,
                "nonfinite_meat",
                bread_condition,
                se_type == "cluster" ? static_cast<int>(n_clusters) : -1);
        }
        covariance.noalias() = bread_inv * meat * bread_inv;
    }

    if (!all_finite_mat(covariance)) {
        return sandwich_unavailable_core(
            size,
            "nonfinite_covariance",
            bread_condition,
            se_type == "cluster" ? static_cast<int>(n_clusters) : -1);
    }
    SandwichCovarianceCore out;
    out.available = true;
    out.se.resize(size);
    for (Eigen::Index idx = 0; idx < size; ++idx) {
        out.se(idx) = std::sqrt(std::max(covariance(idx, idx), 0.0));
    }
    out.covariance = covariance;
    out.bread_condition = bread_condition;
    out.n_clusters = se_type == "cluster" ? static_cast<int>(n_clusters) : -1;
    out.gil_was_released = PyGILState_Check() == 0;
    out.input_snapshot_owned = true;
    return out;
}

class CompressedMixedFallbackRequired final : public std::runtime_error {
  public:
    explicit CompressedMixedFallbackRequired(const std::string& reason)
        : std::runtime_error(reason) {}
};

struct MixedCompressedDesignCore {
    RowMatrixXd slopes;
    GroupBundleCore recover_bundle;
    std::vector<Eigen::Index> recover_offsets;
    Eigen::Index recover_cols = 0;
    Eigen::Index total_cols = 0;
};

struct MixedCompressedGramSolverCore {
    Eigen::MatrixXd gram;
    Eigen::MatrixXd eigenvectors;
    Eigen::VectorXd inverse_eigenvalues;
    double min_retained_eigenvalue = std::numeric_limits<double>::infinity();
    double max_retained_eigenvalue = 0.0;
    double max_discarded_eigenvalue = 0.0;
};

struct MixedCompressedSolveStatsCore {
    int cached_psd_solves = 0;
    int fast_spd_solves = 0;
    int structural_basis_solves = 0;
    int solver_failures = 0;
};

MixedCompressedDesignCore mixed_compressed_design_core(
    const RowMatrixXd& exog,
    const GroupBundleCore& recover_bundle) {
    MixedCompressedDesignCore out;
    out.slopes = exog;
    out.recover_bundle = recover_bundle;
    out.recover_offsets.reserve(recover_bundle.counts.size());
    Eigen::Index offset = exog.cols();
    for (const std::int64_t count : recover_bundle.counts) {
        out.recover_offsets.push_back(offset);
        const Eigen::Index block_cols =
            static_cast<Eigen::Index>(std::max<std::int64_t>(count - 1, 0));
        out.recover_cols += block_cols;
        offset += block_cols;
    }
    out.total_cols = exog.cols() + out.recover_cols;
    return out;
}

Eigen::Index mixed_recover_col_core(
    const MixedCompressedDesignCore& design,
    std::size_t block,
    std::int64_t level) {
    if (level <= 0) {
        return -1;
    }
    return design.recover_offsets[block] + static_cast<Eigen::Index>(level - 1);
}

Eigen::VectorXd mixed_compressed_linear_state_core(
    const MixedCompressedDesignCore& design,
    const Eigen::Ref<const Eigen::VectorXd>& beta) {
    if (beta.size() != design.total_cols) {
        throw std::invalid_argument("mixed compressed beta has wrong length");
    }
    Eigen::VectorXd out(design.slopes.rows());
    if (design.slopes.cols() > 0) {
        out.noalias() = design.slopes * beta.head(design.slopes.cols());
    } else {
        out.setZero();
    }
    for (std::size_t block = 0; block < design.recover_bundle.groups.size(); ++block) {
        const std::vector<std::int64_t>& group = design.recover_bundle.groups[block];
        for (Eigen::Index row = 0; row < out.size(); ++row) {
            const Eigen::Index col = mixed_recover_col_core(
                design,
                block,
                group[static_cast<std::size_t>(row)]);
            if (col >= 0) {
                out(row) += beta(col);
            }
        }
    }
    return out;
}

void mixed_compressed_active_recover_cols_core(
    const MixedCompressedDesignCore& design,
    Eigen::Index row,
    std::vector<Eigen::Index>& active_cols) {
    active_cols.clear();
    for (std::size_t block = 0; block < design.recover_bundle.groups.size(); ++block) {
        const Eigen::Index col = mixed_recover_col_core(
            design,
            block,
            design.recover_bundle.groups[block][static_cast<std::size_t>(row)]);
        if (col >= 0) {
            active_cols.push_back(col);
        }
    }
}

bool mixed_compressed_oneway_cross_core(
    const MixedCompressedDesignCore& design,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    const Eigen::VectorXd* weights,
    const Eigen::Ref<const Eigen::VectorXd>& response,
    Eigen::MatrixXd& gram,
    Eigen::VectorXd& rhs) {
    if (design.slopes.rows() != response.size() ||
        static_cast<Eigen::Index>(cancel_groups.size()) != response.size()) {
        throw std::invalid_argument("mixed compressed cross dimensions do not match");
    }
    if (!design.slopes.allFinite() || !response.allFinite()) {
        return false;
    }
    if (weights != nullptr &&
        (weights->size() != response.size() || !weights->allFinite())) {
        return false;
    }

    const Eigen::Index k_slopes = design.slopes.cols();
    const Eigen::Index p = design.total_cols;
    gram = Eigen::MatrixXd::Zero(p, p);
    rhs = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd cancel_sums =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_cancel_groups), p);
    Eigen::VectorXd cancel_response_sums =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_cancel_groups));
    Eigen::VectorXd cancel_denoms =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_cancel_groups));
    std::vector<std::vector<Eigen::Index>> cancel_supports(
        static_cast<std::size_t>(n_cancel_groups));
    std::vector<Eigen::Index> active_cols;
    active_cols.reserve(design.recover_bundle.groups.size());

    for (Eigen::Index row = 0; row < response.size(); ++row) {
        const Eigen::Index cancel_group =
            static_cast<Eigen::Index>(cancel_groups[static_cast<std::size_t>(row)]);
        const double weight = weights != nullptr ? (*weights)(row) : 1.0;
        const double y = response(row);
        if (!std::isfinite(weight) || !std::isfinite(y)) {
            return false;
        }
        std::vector<Eigen::Index>& support =
            cancel_supports[static_cast<std::size_t>(cancel_group)];
        if (cancel_denoms(cancel_group) <= 0.0) {
            support.reserve(
                static_cast<std::size_t>(
                    k_slopes +
                    static_cast<Eigen::Index>(design.recover_bundle.groups.size())));
            for (Eigen::Index col = 0; col < k_slopes; ++col) {
                support.push_back(col);
            }
        }
        cancel_denoms(cancel_group) += weight;
        cancel_response_sums(cancel_group) += weight * y;

        for (Eigen::Index col = 0; col < k_slopes; ++col) {
            const double x_col = design.slopes(row, col);
            const double wx_col = weight * x_col;
            rhs(col) += wx_col * y;
            cancel_sums(cancel_group, col) += wx_col;
            for (Eigen::Index other = 0; other < k_slopes; ++other) {
                gram(col, other) += wx_col * design.slopes(row, other);
            }
        }

        mixed_compressed_active_recover_cols_core(design, row, active_cols);
        for (const Eigen::Index dummy_col : active_cols) {
            support.push_back(dummy_col);
            rhs(dummy_col) += weight * y;
            cancel_sums(cancel_group, dummy_col) += weight;
            for (Eigen::Index slope_col = 0; slope_col < k_slopes; ++slope_col) {
                const double value = weight * design.slopes(row, slope_col);
                gram(slope_col, dummy_col) += value;
                gram(dummy_col, slope_col) += value;
            }
        }
        for (std::size_t left = 0; left < active_cols.size(); ++left) {
            const Eigen::Index left_col = active_cols[left];
            gram(left_col, left_col) += weight;
            for (std::size_t right = left + 1; right < active_cols.size(); ++right) {
                const Eigen::Index right_col = active_cols[right];
                gram(left_col, right_col) += weight;
                gram(right_col, left_col) += weight;
            }
        }
    }

    for (Eigen::Index group = 0; group < cancel_denoms.size(); ++group) {
        const double denom = cancel_denoms(group);
        if (denom > 1.0e-15) {
            std::vector<Eigen::Index>& support =
                cancel_supports[static_cast<std::size_t>(group)];
            std::sort(support.begin(), support.end());
            support.erase(std::unique(support.begin(), support.end()), support.end());
            const double response_mean = cancel_response_sums(group) / denom;
            for (const Eigen::Index left_col : support) {
                const double left_sum = cancel_sums(group, left_col);
                if (left_sum == 0.0) {
                    continue;
                }
                rhs(left_col) -= left_sum * response_mean;
                const double scaled_left_sum = left_sum / denom;
                for (const Eigen::Index right_col : support) {
                    const double right_sum = cancel_sums(group, right_col);
                    if (right_sum != 0.0) {
                        gram(left_col, right_col) -= scaled_left_sum * right_sum;
                    }
                }
            }
        }
    }
    gram = 0.5 * (gram + gram.transpose());
    return gram.allFinite() && rhs.allFinite();
}

bool mixed_compressed_oneway_rhs_core(
    const MixedCompressedDesignCore& design,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    const Eigen::VectorXd* weights,
    const Eigen::Ref<const Eigen::VectorXd>& response,
    Eigen::VectorXd& rhs) {
    if (design.slopes.rows() != response.size() ||
        static_cast<Eigen::Index>(cancel_groups.size()) != response.size()) {
        throw std::invalid_argument("mixed compressed rhs dimensions do not match");
    }
    if (!design.slopes.allFinite() || !response.allFinite()) {
        return false;
    }
    if (weights != nullptr &&
        (weights->size() != response.size() || !weights->allFinite())) {
        return false;
    }

    const Eigen::Index k_slopes = design.slopes.cols();
    const Eigen::Index p = design.total_cols;
    rhs = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd cancel_sums =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_cancel_groups), p);
    Eigen::VectorXd cancel_response_sums =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_cancel_groups));
    Eigen::VectorXd cancel_denoms =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_cancel_groups));
    std::vector<std::vector<Eigen::Index>> cancel_supports(
        static_cast<std::size_t>(n_cancel_groups));
    std::vector<Eigen::Index> active_cols;
    active_cols.reserve(design.recover_bundle.groups.size());

    for (Eigen::Index row = 0; row < response.size(); ++row) {
        const Eigen::Index cancel_group =
            static_cast<Eigen::Index>(cancel_groups[static_cast<std::size_t>(row)]);
        const double weight = weights != nullptr ? (*weights)(row) : 1.0;
        const double y = response(row);
        if (!std::isfinite(weight) || !std::isfinite(y)) {
            return false;
        }
        std::vector<Eigen::Index>& support =
            cancel_supports[static_cast<std::size_t>(cancel_group)];
        if (cancel_denoms(cancel_group) <= 0.0) {
            support.reserve(
                static_cast<std::size_t>(
                    k_slopes +
                    static_cast<Eigen::Index>(design.recover_bundle.groups.size())));
            for (Eigen::Index col = 0; col < k_slopes; ++col) {
                support.push_back(col);
            }
        }
        cancel_denoms(cancel_group) += weight;
        cancel_response_sums(cancel_group) += weight * y;

        for (Eigen::Index col = 0; col < k_slopes; ++col) {
            const double wx_col = weight * design.slopes(row, col);
            rhs(col) += wx_col * y;
            cancel_sums(cancel_group, col) += wx_col;
        }

        mixed_compressed_active_recover_cols_core(design, row, active_cols);
        for (const Eigen::Index dummy_col : active_cols) {
            support.push_back(dummy_col);
            rhs(dummy_col) += weight * y;
            cancel_sums(cancel_group, dummy_col) += weight;
        }
    }

    for (Eigen::Index group = 0; group < cancel_denoms.size(); ++group) {
        const double denom = cancel_denoms(group);
        if (denom > 1.0e-15) {
            std::vector<Eigen::Index>& support =
                cancel_supports[static_cast<std::size_t>(group)];
            std::sort(support.begin(), support.end());
            support.erase(std::unique(support.begin(), support.end()), support.end());
            const double response_mean = cancel_response_sums(group) / denom;
            for (const Eigen::Index col : support) {
                const double sum = cancel_sums(group, col);
                if (sum != 0.0) {
                    rhs(col) -= sum * response_mean;
                }
            }
        }
    }
    return rhs.allFinite();
}

std::optional<MixedCompressedGramSolverCore>
mixed_compressed_factor_psd_gram_core(
    const Eigen::Ref<const Eigen::MatrixXd>& gram,
    Eigen::Index observation_rows) {
    if (gram.rows() != gram.cols() || gram.cols() < 32 || observation_rows <= 0 ||
        !gram.allFinite()) {
        return std::nullopt;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        gram,
        Eigen::ComputeEigenvectors);
    if (spectrum.info() != Eigen::Success) {
        return std::nullopt;
    }
    const Eigen::VectorXd eigenvalues = spectrum.eigenvalues();
    if (eigenvalues.size() == 0) {
        return std::nullopt;
    }
    const double max_eigen = eigenvalues.maxCoeff();
    if (!std::isfinite(max_eigen) || max_eigen <= 0.0) {
        return std::nullopt;
    }
    const double cutoff =
        max_eigen *
        std::max(
            std::sqrt(std::numeric_limits<double>::epsilon()),
            std::numeric_limits<double>::epsilon() *
                static_cast<double>(std::max(observation_rows, gram.cols())));
    MixedCompressedGramSolverCore out;
    out.gram = gram;
    out.eigenvectors = spectrum.eigenvectors();
    out.inverse_eigenvalues = Eigen::VectorXd::Zero(eigenvalues.size());
    int retained = 0;
    for (Eigen::Index idx = 0; idx < eigenvalues.size(); ++idx) {
        const double lambda = eigenvalues(idx);
        if (std::isfinite(lambda) && lambda > cutoff) {
            out.inverse_eigenvalues(idx) = 1.0 / lambda;
            out.min_retained_eigenvalue =
                std::min(out.min_retained_eigenvalue, lambda);
            out.max_retained_eigenvalue =
                std::max(out.max_retained_eigenvalue, lambda);
            retained += 1;
        } else if (std::isfinite(lambda) && lambda > 0.0) {
            out.max_discarded_eigenvalue =
                std::max(out.max_discarded_eigenvalue, lambda);
        }
    }
    if (retained == 0 || !out.inverse_eigenvalues.allFinite() ||
        !std::isfinite(out.min_retained_eigenvalue) ||
        out.min_retained_eigenvalue <= 0.0 ||
        out.max_retained_eigenvalue <= 0.0) {
        return std::nullopt;
    }
    return out;
}

std::optional<MixedCompressedGramSolverCore>
mixed_compressed_unweighted_solver_core(
    const MixedCompressedDesignCore& design,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    Eigen::Index observation_rows) {
    Eigen::MatrixXd gram;
    Eigen::VectorXd rhs;
    const Eigen::VectorXd zero_response = Eigen::VectorXd::Zero(observation_rows);
    if (!mixed_compressed_oneway_cross_core(
            design,
            cancel_groups,
            n_cancel_groups,
            nullptr,
            zero_response,
            gram,
            rhs)) {
        return std::nullopt;
    }
    return mixed_compressed_factor_psd_gram_core(gram, observation_rows);
}

std::optional<Eigen::VectorXd> mixed_compressed_apply_solver_core(
    const MixedCompressedGramSolverCore& solver,
    const Eigen::Ref<const Eigen::VectorXd>& rhs) {
    if (solver.gram.rows() != rhs.size() ||
        solver.eigenvectors.rows() != rhs.size() ||
        solver.inverse_eigenvalues.size() != rhs.size() || !rhs.allFinite()) {
        return std::nullopt;
    }
    const Eigen::VectorXd rhs_projected = solver.eigenvectors.transpose() * rhs;
    Eigen::VectorXd beta =
        solver.eigenvectors *
        (solver.inverse_eigenvalues.array() * rhs_projected.array()).matrix();
    if (!beta.allFinite()) {
        return std::nullopt;
    }
    const double rhs_scale =
        std::max(pyiols::kernel::stable_l2_norm(rhs), 1.0);
    const Eigen::VectorXd normal_residual_vec = solver.gram * beta - rhs;
    const double normal_residual =
        pyiols::kernel::stable_l2_norm(normal_residual_vec) / rhs_scale;
    if (!std::isfinite(normal_residual) || normal_residual > 1.0e-6) {
        return std::nullopt;
    }
    return beta;
}

bool mixed_compressed_solver_well_conditioned_core(
    const MixedCompressedGramSolverCore& solver) {
    if (!std::isfinite(solver.min_retained_eigenvalue) ||
        !std::isfinite(solver.max_retained_eigenvalue) ||
        !std::isfinite(solver.max_discarded_eigenvalue) ||
        solver.min_retained_eigenvalue <= 0.0 ||
        solver.max_retained_eigenvalue <= 0.0) {
        return false;
    }
    const double retained_condition =
        solver.max_retained_eigenvalue / solver.min_retained_eigenvalue;
    if (!std::isfinite(retained_condition) || retained_condition > 1.0e8) {
        return false;
    }
    return solver.max_discarded_eigenvalue <=
           solver.min_retained_eigenvalue * 1.0e-12;
}

std::optional<Eigen::VectorXd> mixed_compressed_apply_structural_basis_solver_core(
    const MixedCompressedGramSolverCore& structural_solver,
    const Eigen::Ref<const Eigen::MatrixXd>& gram,
    const Eigen::Ref<const Eigen::VectorXd>& rhs) {
    if (gram.rows() != gram.cols() || gram.rows() != rhs.size() ||
        structural_solver.eigenvectors.rows() != rhs.size() ||
        structural_solver.inverse_eigenvalues.size() != rhs.size() ||
        !gram.allFinite() || !rhs.allFinite()) {
        return std::nullopt;
    }
    std::vector<Eigen::Index> retained;
    retained.reserve(static_cast<std::size_t>(rhs.size()));
    for (Eigen::Index idx = 0; idx < structural_solver.inverse_eigenvalues.size(); ++idx) {
        const double inverse = structural_solver.inverse_eigenvalues(idx);
        if (std::isfinite(inverse) && inverse > 0.0) {
            retained.push_back(idx);
        }
    }
    if (retained.empty()) {
        return std::nullopt;
    }
    Eigen::MatrixXd basis(rhs.size(), static_cast<Eigen::Index>(retained.size()));
    for (std::size_t col = 0; col < retained.size(); ++col) {
        basis.col(static_cast<Eigen::Index>(col)) =
            structural_solver.eigenvectors.col(retained[col]);
    }
    Eigen::MatrixXd restricted_gram = basis.transpose() * gram * basis;
    restricted_gram = 0.5 * (restricted_gram + restricted_gram.transpose());
    const Eigen::VectorXd restricted_rhs = basis.transpose() * rhs;
    const std::optional<Eigen::VectorXd> restricted_beta =
        pyiols::kernel::solve_well_conditioned_spd_gram_fast_path(
            restricted_gram,
            restricted_rhs,
            1);
    if (!restricted_beta.has_value()) {
        return std::nullopt;
    }
    Eigen::VectorXd beta = basis * (*restricted_beta);
    if (!beta.allFinite()) {
        return std::nullopt;
    }
    const double rhs_scale =
        std::max(pyiols::kernel::stable_l2_norm(rhs), 1.0);
    const Eigen::VectorXd normal_residual_vec = gram * beta - rhs;
    const double normal_residual =
        pyiols::kernel::stable_l2_norm(normal_residual_vec) / rhs_scale;
    if (!std::isfinite(normal_residual) || normal_residual > 1.0e-6) {
        return std::nullopt;
    }
    return beta;
}

Eigen::VectorXd mixed_compressed_solve_oneway_cancel_core(
    const MixedCompressedDesignCore& design,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    const Eigen::VectorXd* weights,
    const Eigen::Ref<const Eigen::VectorXd>& response,
    const MixedCompressedGramSolverCore* cached_unweighted_solver = nullptr) {
    if (weights == nullptr && cached_unweighted_solver != nullptr) {
        Eigen::VectorXd rhs;
        if (!mixed_compressed_oneway_rhs_core(
                design,
                cancel_groups,
                n_cancel_groups,
                nullptr,
                response,
                rhs)) {
            throw CompressedMixedFallbackRequired(
                "mixed_compressed_rhs_nonfinite");
        }
        const std::optional<Eigen::VectorXd> beta =
            mixed_compressed_apply_solver_core(*cached_unweighted_solver, rhs);
        if (beta.has_value()) {
            return *beta;
        }
    }
    Eigen::MatrixXd gram;
    Eigen::VectorXd rhs;
    if (!mixed_compressed_oneway_cross_core(
            design,
            cancel_groups,
            n_cancel_groups,
            weights,
            response,
            gram,
            rhs)) {
        throw CompressedMixedFallbackRequired(
            "mixed_compressed_cross_nonfinite");
    }
    const std::optional<Eigen::VectorXd> beta =
        pyiols::kernel::solve_well_conditioned_spd_gram_fast_path(
            gram,
            rhs,
            32);
    if (beta.has_value()) {
        return *beta;
    }
    if (weights != nullptr && cached_unweighted_solver != nullptr) {
        const std::optional<Eigen::VectorXd> structural_beta =
            mixed_compressed_apply_structural_basis_solver_core(
                *cached_unweighted_solver,
                gram,
                rhs);
        if (structural_beta.has_value()) {
            return *structural_beta;
        }
    }
    throw CompressedMixedFallbackRequired(
        "mixed_compressed_gram_not_well_conditioned");
}

Eigen::VectorXd demean_vector_oneway_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const Eigen::VectorXd* weights) {
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(n_groups);
    Eigen::VectorXd denoms = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        const Eigen::Index group =
            static_cast<Eigen::Index>(groups[static_cast<std::size_t>(row)]);
        const double weight = weights != nullptr ? (*weights)(row) : 1.0;
        sums(group) += weight * values(row);
        denoms(group) += weight;
    }
    for (Eigen::Index group = 0; group < n_groups; ++group) {
        const double divisor =
            weights != nullptr ? denoms(group) : std::max(denoms(group), 1.0);
        if (divisor > 1.0e-15) {
            sums(group) /= divisor;
        } else {
            sums(group) = 0.0;
        }
    }
    Eigen::VectorXd out(values.size());
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        out(row) = values(row) -
                   sums(static_cast<Eigen::Index>(
                       groups[static_cast<std::size_t>(row)]));
    }
    return out;
}

double mixed_compressed_score_residual_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const MixedCompressedDesignCore& design,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    double tau) {
    const Eigen::VectorXd q = mixed_compressed_linear_state_core(design, beta);
    if (!all_finite_vec(q)) {
        return std::numeric_limits<double>::infinity();
    }
    Eigen::VectorXd weights(q.size());
    Eigen::VectorXd ratio_values(q.size());
    for (Eigen::Index row = 0; row < q.size(); ++row) {
        weights(row) = exp_guard_scalar((2.0 - tau) * q(row));
        ratio_values(row) = endog(row) * exp_guard_scalar(-q(row));
    }
    if (!all_finite_vec(weights) || !all_finite_vec(ratio_values)) {
        return std::numeric_limits<double>::infinity();
    }
    const bool tau2_fixed_anchor = std::abs(tau - 2.0) <= 1e-10;
    const Eigen::VectorXd ratio_demeaned =
        demean_vector_oneway_core(
            ratio_values,
            cancel_groups,
            n_cancel_groups,
            tau2_fixed_anchor ? nullptr : &weights);
    if (!all_finite_vec(ratio_demeaned)) {
        return std::numeric_limits<double>::infinity();
    }
    Eigen::VectorXd score = Eigen::VectorXd::Zero(design.total_cols);
    std::vector<Eigen::Index> active_cols;
    active_cols.reserve(design.recover_bundle.groups.size());
    for (Eigen::Index row = 0; row < q.size(); ++row) {
        const double value =
            tau2_fixed_anchor ? ratio_demeaned(row) : weights(row) * ratio_demeaned(row);
        for (Eigen::Index col = 0; col < design.slopes.cols(); ++col) {
            score(col) += design.slopes(row, col) * value;
        }
        mixed_compressed_active_recover_cols_core(design, row, active_cols);
        for (const Eigen::Index dummy_col : active_cols) {
            score(dummy_col) += value;
        }
    }
    if (!all_finite_vec(score)) {
        return std::numeric_limits<double>::infinity();
    }
    if (score.size() == 0) {
        return 0.0;
    }
    return score.cwiseAbs().maxCoeff() / std::max<double>(endog.size(), 1.0);
}

void demean_for_group_bundle_into_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const GroupBundleCore& bundle,
    const Eigen::VectorXd* weights,
    double tol,
    int maxiter,
    Eigen::MatrixXd& out,
    DemeanWorkspaceCore& workspace,
    bool* converged) {
    if (bundle.groups.empty()) {
        out = values;
        if (converged != nullptr) {
            *converged = true;
        }
        return;
    }
    if (values.cols() == 0) {
        out = values;
        if (converged != nullptr) {
            *converged = true;
        }
        return;
    }
    if (bundle.groups.size() == 1) {
        demean_oneway_matrix_into_core(
            values,
            bundle.groups.front(),
            bundle.counts.front(),
            weights,
            out,
            workspace);
        if (converged != nullptr) {
            *converged = true;
        }
        return;
    }

    out = values;
    bool did_converge = false;
    for (int iter = 0; iter < std::max(maxiter, 0); ++iter) {
        workspace.previous = out;
        Eigen::MatrixXd* current = &out;
        Eigen::MatrixXd* target = &workspace.scratch;
        for (std::size_t index = 0; index < bundle.groups.size(); ++index) {
            demean_oneway_matrix_into_core(
                *current,
                bundle.groups[index],
                bundle.counts[index],
                weights,
                *target,
                workspace);
            std::swap(current, target);
        }
        if (current != &out) {
            out = *current;
        }
        if (max_abs_mat_diff(out, workspace.previous) <= tol) {
            did_converge = true;
            break;
        }
    }
    if (converged != nullptr) {
        *converged = did_converge;
    }
}

void demean_vector_for_group_bundle_unweighted_into_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const GroupBundleCore& bundle,
    double tol,
    int maxiter,
    Eigen::VectorXd& out,
    DemeanVectorWorkspaceCore& workspace,
    bool* converged) {
    workspace.last_sweep_iterations = 0;
    if (bundle.groups.empty()) {
        out = values;
        if (converged != nullptr) {
            *converged = true;
        }
        return;
    }
    if (values.size() == 0) {
        out = values;
        if (converged != nullptr) {
            *converged = true;
        }
        return;
    }
    if (bundle.groups.size() == 1) {
        demean_oneway_vector_unweighted_into_core(
            values,
            bundle.groups.front(),
            bundle.counts.front(),
            out,
            workspace,
            bundle.unweighted_inverse_counts.empty()
                ? nullptr
                : &bundle.unweighted_inverse_counts.front());
        if (converged != nullptr) {
            *converged = true;
        }
        return;
    }

    if (maxiter <= 0) {
        out = values;
        if (converged != nullptr) {
            *converged = false;
        }
        return;
    }
    workspace.bundle_sums.resize(bundle.groups.size());
    workspace.bundle_denom.resize(bundle.groups.size());
    bool did_converge = false;
    for (int iter = 0; iter < maxiter; ++iter) {
        workspace.last_sweep_iterations += 1;
        if (iter == 0) {
            workspace.previous = values;
        } else {
            workspace.previous = out;
        }
        Eigen::VectorXd* current = &workspace.previous;
        Eigen::VectorXd* target =
            bundle.groups.size() % 2 == 0 ? &workspace.scratch : &out;
        for (std::size_t index = 0; index < bundle.groups.size(); ++index) {
            demean_oneway_vector_unweighted_with_buffers_core(
                *current,
                bundle.groups[index],
                bundle.counts[index],
                *target,
                workspace.bundle_sums[index],
                workspace.bundle_denom[index],
                index < bundle.unweighted_inverse_counts.size()
                    ? &bundle.unweighted_inverse_counts[index]
                    : nullptr);
            current = target;
            target = current == &out ? &workspace.scratch : &out;
        }
        if (current != &out) {
            out = *current;
        }
        if ((out - workspace.previous).cwiseAbs().maxCoeff() <= tol) {
            did_converge = true;
            break;
        }
    }
    if (converged != nullptr) {
        *converged = did_converge;
    }
}

Eigen::MatrixXd demean_for_group_bundle_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const GroupBundleCore& bundle,
    const std::optional<Eigen::VectorXd>& weights,
    double tol,
    int maxiter) {
    Eigen::MatrixXd out;
    DemeanWorkspaceCore workspace;
    const Eigen::VectorXd* weights_ptr = weights.has_value() ? &(*weights) : nullptr;
    demean_for_group_bundle_into_core(
        values,
        bundle,
        weights_ptr,
        tol,
        maxiter,
        out,
        workspace);
    return out;
}

Eigen::VectorXd demean_vector_for_group_bundle_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const GroupBundleCore& bundle,
    const std::optional<Eigen::VectorXd>& weights,
    double tol,
    int maxiter) {
    const Eigen::MatrixXd mat = values;
    return demean_for_group_bundle_core(mat, bundle, weights, tol, maxiter).col(0);
}

struct FullrecoverInvariantCore {
    Eigen::MatrixXd x_demeaned;
    Eigen::MatrixXd projection;
    Eigen::MatrixXd manyway_q_basis;
};

FullrecoverInvariantCore fullrecover_invariant_core(
    const Eigen::Ref<const RowMatrixXd>& exog,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    FullrecoverInvariantCore state;
    state.x_demeaned = demean_profiled_core(exog, groups, n_groups);
    if (state.x_demeaned.cols() == 0) {
        state.projection = Eigen::MatrixXd(0, state.x_demeaned.rows());
        return state;
    }
    state.projection = pyiols::kernel::min_norm_lstsq(
        state.x_demeaned.transpose() * state.x_demeaned,
        state.x_demeaned.transpose());
    return state;
}

Eigen::VectorXd initial_linear_profiled_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau) {
    return profile_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
}

bool phi_step_profiled_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    int phase,
    double param,
    Eigen::VectorXd& beta_new) {
    const Eigen::VectorXd linear_current =
        profile_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    if (!all_finite_vec(linear_current)) {
        beta_new = beta;
        return false;
    }
    const double phase1_log_denom = phase == 1 ? std::log1p(param) : 0.0;
    const double phase1_inv_denom =
        phase == 1 ? 1.0 / (1.0 + param) : std::numeric_limits<double>::quiet_NaN();
    const bool tau_is_one = std::abs(tau - 1.0) <= 1e-12;
    const bool tau_is_two = std::abs(tau - 2.0) <= 1e-12;
    Eigen::VectorXd pseudo(linear_current.size());
    for (Eigen::Index row = 0; row < linear_current.size(); ++row) {
        const double linear = linear_current(row);
        if (phase == 0) {
            const double mu = exp_guard_scalar(linear);
            const double u_ratio = endog(row) / mu;
            const double weight = exp_guard_scalar((2.0 - tau) * linear);
            const double delta = param;
            const double delta_tilde = delta / std::max(weight, 1e-300);
            pseudo(row) =
                linear + std::log1p((u_ratio - 1.0) / (1.0 + delta_tilde));
        } else {
            double positive_component = 0.0;
            if (endog(row) > 0.0) {
                if (tau_is_one) {
                    positive_component = endog(row) * phase1_inv_denom;
                } else {
                    const double positive_log =
                        std::log(endog(row)) + (1.0 - tau) * linear -
                        phase1_log_denom;
                    positive_component = exp_guard_scalar(positive_log);
                }
            }
            const double negative_component =
                tau_is_two ? phase1_inv_denom
                           : exp_guard_scalar((2.0 - tau) * linear -
                                              phase1_log_denom);
            pseudo(row) = linear + positive_component - negative_component;
        }
    }
    if (!all_finite_vec(pseudo)) {
        beta_new = beta;
        return false;
    }
    const Eigen::MatrixXd pseudo_mat = pseudo;
    const Eigen::VectorXd demeaned =
        demean_profiled_core(pseudo_mat, groups, n_groups).col(0);
    beta_new = projection * demeaned;
    if (!all_finite_vec(beta_new)) {
        beta_new = beta;
        return false;
    }
    return true;
}

double profiled_slope_score_residual_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const Eigen::Ref<const Eigen::MatrixXd>& x_demeaned,
    double tau) {
    const Eigen::VectorXd linear =
        profile_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    Eigen::VectorXd score_obs(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        score_obs(row) = endog(row) * exp_guard_scalar((1.0 - tau) * linear(row)) -
                         exp_guard_scalar((2.0 - tau) * linear(row));
    }
    if (!all_finite_vec(score_obs)) {
        return std::numeric_limits<double>::infinity();
    }
    const Eigen::VectorXd score = x_demeaned.transpose() * score_obs;
    if (!all_finite_vec(score)) {
        return std::numeric_limits<double>::infinity();
    }
    if (score.size() == 0) {
        return 0.0;
    }
    return score.cwiseAbs().maxCoeff() / std::max<double>(endog.size(), 1.0);
}

double profiled_fe_score_residual_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau) {
    const Eigen::VectorXd linear =
        profile_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    Eigen::VectorXd score_obs(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        score_obs(row) = endog(row) * exp_guard_scalar((1.0 - tau) * linear(row)) -
                         exp_guard_scalar((2.0 - tau) * linear(row));
    }
    if (!all_finite_vec(score_obs)) {
        return std::numeric_limits<double>::infinity();
    }
    if (is_singleton_intercept_group(groups, n_groups)) {
        return std::abs(score_obs.sum()) / std::max<double>(endog.size(), 1.0);
    }
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < score_obs.size(); ++row) {
        sums(groups[static_cast<std::size_t>(row)]) += score_obs(row);
    }
    return sums.cwiseAbs().maxCoeff() / std::max<double>(endog.size(), 1.0);
}

std::vector<double> exp_linspace(double log_min, double log_max, int size);
std::vector<double> unique_positive_finite(std::vector<double> values);
std::vector<double> focus_rho_path_core(
    const std::vector<double>& path,
    std::optional<double> rho_hint);
std::vector<double> rho_path_core(
    const FullrecoverControlCore& control,
    std::optional<double> rho_hint);
std::vector<double> delta_grid_core(const FullrecoverControlCore& control);
double max_abs_change_core(
    const Eigen::Ref<const Eigen::VectorXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& right);
bool criterion_stagnated_core(
    const std::vector<double>& history,
    double eps,
    int window,
    int min_iterations,
    double min_relative_drop);
bool record_criterion_progress_core(
    std::vector<double>& history,
    double criterion,
    int phase,
    double tau,
    double eps,
    int max_phase2_evaluations,
    double& last_finite_criterion,
    double& best_finite_criterion,
    std::string& failure_reason,
    const std::vector<double>* stagnation_history = nullptr);
int adapt_anderson_m_core(int aa_m, double kappa_val);
FullrecoverIterationCore not_started_iteration_core(
    const FullrecoverControlCore& control);

bool any_nan_vec(const Eigen::Ref<const Eigen::VectorXd>& values) {
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        if (std::isnan(values(idx))) {
            return true;
        }
    }
    return false;
}

Eigen::VectorXd min_norm_lstsq_vector_allow_empty_cols(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::VectorXd>& b) {
    if (a.rows() != b.rows()) {
        throw std::invalid_argument("least-squares row dimensions do not match");
    }
    if (a.cols() == 0) {
        return Eigen::VectorXd(0);
    }
    const Eigen::MatrixXd b_mat = b;
    return pyiols::kernel::min_norm_lstsq(a, b_mat).col(0);
}

std::int64_t group_bundle_total_levels_core(const GroupBundleCore& bundle) {
    std::int64_t total = 0;
    for (const std::int64_t count : bundle.counts) {
        total += std::max<std::int64_t>(count, 0);
    }
    return total;
}

bool small_manyway_exact_demean_enabled_core(
    const GroupBundleCore& bundle,
    Eigen::Index n_obs,
    const FullrecoverControlCore& control) {
    if (!control.speed_small_manyway_exact_demean) {
        return false;
    }
    if (bundle.groups.size() < 2 || n_obs <= 0) {
        return false;
    }
    const std::int64_t total_levels = group_bundle_total_levels_core(bundle);
    if (total_levels <= 0) {
        return false;
    }
    const int max_rows = control.speed_small_manyway_exact_demean_max_rows;
    const int max_levels = control.speed_small_manyway_exact_demean_max_levels;
    return n_obs <= max_rows && total_levels <= max_levels;
}

Eigen::MatrixXd small_manyway_dummy_q_basis_core(
    const GroupBundleCore& bundle,
    Eigen::Index n_obs) {
    const std::int64_t total_levels = group_bundle_total_levels_core(bundle);
    if (n_obs <= 0 || total_levels <= 0) {
        return Eigen::MatrixXd(n_obs, 0);
    }
    Eigen::MatrixXd dummies =
        Eigen::MatrixXd::Zero(n_obs, static_cast<Eigen::Index>(total_levels));
    Eigen::Index offset = 0;
    for (std::size_t dim = 0; dim < bundle.groups.size(); ++dim) {
        const auto& groups = bundle.groups[dim];
        const Eigen::Index count = static_cast<Eigen::Index>(bundle.counts[dim]);
        for (Eigen::Index row = 0; row < n_obs; ++row) {
            const auto group = static_cast<Eigen::Index>(
                groups[static_cast<std::size_t>(row)]);
            if (group >= 0 && group < count) {
                dummies(row, offset + group) = 1.0;
            }
        }
        offset += count;
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(dummies);
    qr.setThreshold(
        Eigen::NumTraits<double>::epsilon() *
        static_cast<double>(std::max(dummies.rows(), dummies.cols())));
    const Eigen::Index rank = static_cast<Eigen::Index>(qr.rank());
    if (rank <= 0) {
        return Eigen::MatrixXd(n_obs, 0);
    }
    Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(n_obs, rank);
    basis = qr.householderQ() * basis;
    return basis;
}

Eigen::VectorXd residualize_vector_with_q_basis_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& q_basis) {
    if (q_basis.cols() == 0) {
        return values;
    }
    Eigen::VectorXd projected(values.size());
    projected.noalias() = q_basis * (q_basis.transpose() * values);
    return values - projected;
}

Eigen::MatrixXd residualize_matrix_with_q_basis_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& q_basis) {
    if (q_basis.cols() == 0) {
        return values;
    }
    Eigen::MatrixXd projected(values.rows(), values.cols());
    projected.noalias() = q_basis * (q_basis.transpose() * values);
    return values - projected;
}

FullrecoverInvariantCore fullrecover_invariant_bundle_core(
    const Eigen::Ref<const RowMatrixXd>& exog,
    const GroupBundleCore& bundle,
    double tau_target,
    const FullrecoverControlCore& control) {
    FullrecoverInvariantCore state;
    if (std::abs(tau_target - 2.0) > 1e-10 &&
        small_manyway_exact_demean_enabled_core(bundle, exog.rows(), control)) {
        state.manyway_q_basis =
            small_manyway_dummy_q_basis_core(bundle, exog.rows());
        state.x_demeaned =
            residualize_matrix_with_q_basis_core(exog, state.manyway_q_basis);
    } else {
        state.x_demeaned = demean_for_group_bundle_core(
            exog,
            bundle,
            std::nullopt,
            control.fe_demean_tol,
            control.fe_demean_maxiter);
    }
    if (state.x_demeaned.cols() == 0) {
        state.projection = Eigen::MatrixXd(0, state.x_demeaned.rows());
        return state;
    }
    state.projection = pyiols::kernel::min_norm_lstsq(
        state.x_demeaned.transpose() * state.x_demeaned,
        state.x_demeaned.transpose());
    return state;
}

Eigen::VectorXd recover_fe_linear_bundle_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const GroupBundleCore& bundle,
    const FullrecoverControlCore& control,
    const Eigen::MatrixXd* manyway_q_basis = nullptr) {
    const Eigen::VectorXd demeaned =
        manyway_q_basis != nullptr && manyway_q_basis->cols() > 0
            ? residualize_vector_with_q_basis_core(values, *manyway_q_basis)
            : demean_vector_for_group_bundle_core(
                  values,
                  bundle,
                  std::nullopt,
                  control.fe_demean_tol,
                  control.fe_demean_maxiter);
    return values - demeaned;
}

Eigen::VectorXd initial_linear_joint_fullrecover_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const GroupBundleCore& bundle,
    const FullrecoverControlCore& control,
    const Eigen::MatrixXd* manyway_q_basis = nullptr) {
    const Eigen::VectorXd base = exog * beta;
    Eigen::VectorXd target(endog.size());
    for (Eigen::Index row = 0; row < endog.size(); ++row) {
        target(row) = std::log(endog(row) + 1.0);
    }
    const Eigen::VectorXd recovered = recover_fe_linear_bundle_core(
        target - base,
        bundle,
        control,
        manyway_q_basis);
    return base + recovered;
}

struct FullrecoverJointWorkspaceCore {
    Eigen::VectorXd pseudo;
    Eigen::VectorXd pseudo_demeaned;
    Eigen::VectorXd fitted_demeaned;
    DemeanVectorWorkspaceCore pseudo_demean;
};

bool phi_step_joint_fullrecover_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const Eigen::Ref<const Eigen::VectorXd>& linear,
    const GroupBundleCore& bundle,
    double tau,
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    const Eigen::Ref<const Eigen::MatrixXd>& x_demeaned,
    const Eigen::MatrixXd* manyway_q_basis,
    int phase,
    double param,
    double fe_demean_tol,
    const FullrecoverControlCore& control,
    Eigen::VectorXd& beta_new,
    Eigen::VectorXd& linear_new,
    FullrecoverJointWorkspaceCore* workspace = nullptr,
    FullrecoverIterationCore* diagnostics = nullptr) {
    if (!all_finite_vec(linear)) {
        beta_new = beta;
        linear_new = linear;
        return false;
    }
    if (diagnostics != nullptr) {
        diagnostics->joint_hdfe_phi_calls += 1;
    }
    Eigen::VectorXd pseudo_local;
    Eigen::VectorXd pseudo_demeaned_local;
    Eigen::VectorXd fitted_demeaned_local;
    DemeanVectorWorkspaceCore pseudo_demean_local;
    Eigen::VectorXd& pseudo =
        workspace != nullptr ? workspace->pseudo : pseudo_local;
    Eigen::VectorXd& pseudo_demeaned =
        workspace != nullptr ? workspace->pseudo_demeaned : pseudo_demeaned_local;
    Eigen::VectorXd& fitted_demeaned =
        workspace != nullptr ? workspace->fitted_demeaned : fitted_demeaned_local;
    DemeanVectorWorkspaceCore& pseudo_demean =
        workspace != nullptr ? workspace->pseudo_demean : pseudo_demean_local;

    const double phase1_log_denom = phase == 1 ? std::log1p(param) : 0.0;
    const double phase1_inv_denom =
        phase == 1 ? 1.0 / (1.0 + param) : std::numeric_limits<double>::quiet_NaN();
    const bool tau_is_one = std::abs(tau - 1.0) <= 1e-12;
    const bool tau_is_two = std::abs(tau - 2.0) <= 1e-12;
    pseudo.resize(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        if (phase == 0) {
            const double mu = exp_guard_scalar(linear(row));
            const double u_ratio = endog(row) / mu;
            const double weight = exp_guard_scalar((2.0 - tau) * linear(row));
            const double delta_tilde = param / std::max(weight, 1e-300);
            pseudo(row) =
                linear(row) + std::log1p((u_ratio - 1.0) / (1.0 + delta_tilde));
        } else {
            double positive_component = 0.0;
            if (endog(row) > 0.0) {
                if (tau_is_one) {
                    positive_component = endog(row) * phase1_inv_denom;
                } else {
                    const double positive_log =
                        std::log(endog(row)) + (1.0 - tau) * linear(row) -
                        phase1_log_denom;
                    positive_component = exp_guard_scalar(positive_log);
                }
            }
            const double negative_component =
                tau_is_two ? phase1_inv_denom
                           : exp_guard_scalar((2.0 - tau) * linear(row) -
                                              phase1_log_denom);
            pseudo(row) = linear(row) + positive_component - negative_component;
        }
    }
    if (!all_finite_vec(pseudo)) {
        beta_new = beta;
        linear_new = linear;
        return false;
    }
    if (manyway_q_basis != nullptr && manyway_q_basis->cols() > 0) {
        pseudo_demeaned =
            residualize_vector_with_q_basis_core(pseudo, *manyway_q_basis);
        if (diagnostics != nullptr) {
            diagnostics->joint_hdfe_phi_demean_sweeps += 1;
        }
    } else {
        demean_vector_for_group_bundle_unweighted_into_core(
            pseudo,
            bundle,
            fe_demean_tol,
            control.fe_demean_maxiter,
            pseudo_demeaned,
            pseudo_demean);
        if (diagnostics != nullptr) {
            diagnostics->joint_hdfe_phi_demean_sweeps +=
                pseudo_demean.last_sweep_iterations;
        }
    }
    beta_new = projection * pseudo_demeaned;
    if (!all_finite_vec(beta_new)) {
        beta_new = beta;
        linear_new = linear;
        return false;
    }
    fitted_demeaned.resize(pseudo.size());
    fitted_demeaned.noalias() = x_demeaned * beta_new;
    if (!all_finite_vec(fitted_demeaned)) {
        beta_new = beta;
        linear_new = linear;
        return false;
    }
    linear_new.resize(linear.size());
    for (Eigen::Index row = 0; row < linear_new.size(); ++row) {
        linear_new(row) =
            pseudo(row) - pseudo_demeaned(row) + fitted_demeaned(row);
    }
    if (!all_finite_vec(linear_new)) {
        beta_new = beta;
        linear_new = linear;
        return false;
    }
    return true;
}

double slope_score_residual_with_linear_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const Eigen::VectorXd>& linear,
    const Eigen::Ref<const Eigen::MatrixXd>& x_demeaned,
    double tau) {
    Eigen::VectorXd score_obs(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        score_obs(row) = endog(row) * exp_guard_scalar((1.0 - tau) * linear(row)) -
                         exp_guard_scalar((2.0 - tau) * linear(row));
    }
    if (!all_finite_vec(score_obs)) {
        return std::numeric_limits<double>::infinity();
    }
    const Eigen::VectorXd score = x_demeaned.transpose() * score_obs;
    if (!all_finite_vec(score)) {
        return std::numeric_limits<double>::infinity();
    }
    if (score.size() == 0) {
        return 0.0;
    }
    return score.cwiseAbs().maxCoeff() / std::max<double>(endog.size(), 1.0);
}

double fe_score_residual_bundle_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const Eigen::VectorXd>& linear,
    const GroupBundleCore& bundle,
    double tau) {
    Eigen::VectorXd score_obs(linear.size());
    for (Eigen::Index row = 0; row < linear.size(); ++row) {
        score_obs(row) = endog(row) * exp_guard_scalar((1.0 - tau) * linear(row)) -
                         exp_guard_scalar((2.0 - tau) * linear(row));
    }
    if (!all_finite_vec(score_obs)) {
        return std::numeric_limits<double>::infinity();
    }
    double residual = 0.0;
    for (std::size_t dim = 0; dim < bundle.groups.size(); ++dim) {
        Eigen::VectorXd sums = Eigen::VectorXd::Zero(bundle.counts[dim]);
        const std::vector<std::int64_t>& groups = bundle.groups[dim];
        for (Eigen::Index row = 0; row < score_obs.size(); ++row) {
            sums(groups[static_cast<std::size_t>(row)]) += score_obs(row);
        }
        if (sums.size() > 0) {
            residual = std::max(residual, sums.cwiseAbs().maxCoeff());
        }
    }
    return residual / std::max<double>(endog.size(), 1.0);
}

bool phase2_initialization_usable_joint_fullrecover_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const Eigen::Ref<const Eigen::VectorXd>& linear,
    const GroupBundleCore& bundle,
    double tau,
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    const Eigen::Ref<const Eigen::MatrixXd>& x_demeaned,
    const Eigen::MatrixXd* manyway_q_basis,
    const FullrecoverControlCore& control,
    FullrecoverIterationCore* diagnostics = nullptr) {
    if (!all_finite_vec(beta) || !all_finite_vec(linear)) {
        return false;
    }
    const std::vector<double> rho_path = rho_path_core(control, std::nullopt);
    for (const double rho : rho_path) {
        Eigen::VectorXd beta_test(beta.size());
        Eigen::VectorXd linear_test(linear.size());
        if (phi_step_joint_fullrecover_core(
                endog,
                beta,
                linear,
                bundle,
                tau,
                projection,
                x_demeaned,
                manyway_q_basis,
                1,
                rho,
                control.fe_demean_tol,
                control,
                beta_test,
                linear_test,
                nullptr,
                diagnostics)) {
            return true;
        }
    }
    return false;
}

Eigen::VectorXd fullcancel_linear_state_core(
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta) {
    return exog * beta;
}

struct FullcancelPhiWorkspaceCore {
    DemeanWorkspaceCore x_demean;
    DemeanWorkspaceCore y_demean;
    Eigen::VectorXd q;
    Eigen::VectorXd weights;
    Eigen::MatrixXd x_demeaned;
    Eigen::MatrixXd y_values;
    Eigen::MatrixXd y_demeaned;
};

double fullcancel_score_residual_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const GroupBundleCore& bundle,
    double tau,
    const FullrecoverControlCore& control,
    const Eigen::MatrixXd* unweighted_x_demeaned = nullptr,
    FullcancelPhiWorkspaceCore* workspace = nullptr) {
    Eigen::VectorXd q_local;
    Eigen::VectorXd weights_local;
    Eigen::MatrixXd ratio_values_local;
    Eigen::MatrixXd x_demeaned_local;
    Eigen::MatrixXd ratio_demeaned_local;
    DemeanWorkspaceCore x_demean_local;
    DemeanWorkspaceCore ratio_demean_local;
    Eigen::VectorXd& q = workspace != nullptr ? workspace->q : q_local;
    Eigen::VectorXd& weights =
        workspace != nullptr ? workspace->weights : weights_local;
    Eigen::MatrixXd& ratio_values =
        workspace != nullptr ? workspace->y_values : ratio_values_local;
    Eigen::MatrixXd& x_demeaned =
        workspace != nullptr ? workspace->x_demeaned : x_demeaned_local;
    Eigen::MatrixXd& ratio_demeaned =
        workspace != nullptr ? workspace->y_demeaned : ratio_demeaned_local;
    DemeanWorkspaceCore& x_demean =
        workspace != nullptr ? workspace->x_demean : x_demean_local;
    DemeanWorkspaceCore& ratio_demean =
        workspace != nullptr ? workspace->y_demean : ratio_demean_local;
    q.resize(exog.rows());
    q.noalias() = exog * beta;
    if (!all_finite_vec(q)) {
        return std::numeric_limits<double>::infinity();
    }
    weights.resize(q.size());
    ratio_values.resize(q.size(), 1);
    for (Eigen::Index row = 0; row < q.size(); ++row) {
        weights(row) = exp_guard_scalar((2.0 - tau) * q(row));
        ratio_values(row, 0) = endog(row) * exp_guard_scalar(-q(row));
    }
    if (!all_finite_vec(weights) || !all_finite_mat(ratio_values)) {
        return std::numeric_limits<double>::infinity();
    }
    if (std::abs(tau - 2.0) <= 1e-10) {
        Eigen::MatrixXd x_demeaned_owned;
        const Eigen::MatrixXd* x_demeaned_fixed = unweighted_x_demeaned;
        if (x_demeaned_fixed == nullptr) {
            x_demeaned_owned = demean_for_group_bundle_core(
                exog,
                bundle,
                std::nullopt,
                control.fe_demean_tol,
                control.fe_demean_maxiter);
            x_demeaned_fixed = &x_demeaned_owned;
        }
        demean_for_group_bundle_into_core(
            ratio_values,
            bundle,
            nullptr,
            control.fe_demean_tol,
            control.fe_demean_maxiter,
            ratio_demeaned,
            ratio_demean);
        if (!all_finite_mat(*x_demeaned_fixed) || !all_finite_mat(ratio_demeaned)) {
            return std::numeric_limits<double>::infinity();
        }
        const Eigen::VectorXd score =
            x_demeaned_fixed->transpose() * ratio_demeaned.col(0);
        if (!all_finite_vec(score)) {
            return std::numeric_limits<double>::infinity();
        }
        if (score.size() == 0) {
            return 0.0;
        }
        return score.cwiseAbs().maxCoeff() / std::max<double>(endog.size(), 1.0);
    }
    demean_for_group_bundle_into_core(
        exog,
        bundle,
        &weights,
        control.fe_demean_tol,
        control.fe_demean_maxiter,
        x_demeaned,
        x_demean);
    demean_for_group_bundle_into_core(
        ratio_values,
        bundle,
        &weights,
        control.fe_demean_tol,
        control.fe_demean_maxiter,
        ratio_demeaned,
        ratio_demean);
    if (!all_finite_mat(x_demeaned) || !all_finite_mat(ratio_demeaned)) {
        return std::numeric_limits<double>::infinity();
    }
    for (Eigen::Index row = 0; row < ratio_demeaned.rows(); ++row) {
        ratio_demeaned(row, 0) *= weights(row);
    }
    const Eigen::VectorXd score = x_demeaned.transpose() * ratio_demeaned.col(0);
    if (!all_finite_vec(score)) {
        return std::numeric_limits<double>::infinity();
    }
    if (score.size() == 0) {
        return 0.0;
    }
    return score.cwiseAbs().maxCoeff() / std::max<double>(endog.size(), 1.0);
}

double fullcancel_initial_scale_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const GroupBundleCore& bundle,
    double tau,
    const FullrecoverControlCore& control,
    FullcancelPhiWorkspaceCore* workspace = nullptr) {
    Eigen::VectorXd q_local;
    Eigen::VectorXd weights_local;
    Eigen::MatrixXd ratio_values_local;
    Eigen::MatrixXd ratio_demeaned_local;
    DemeanWorkspaceCore ratio_demean_local;
    Eigen::VectorXd& q = workspace != nullptr ? workspace->q : q_local;
    Eigen::VectorXd& weights =
        workspace != nullptr ? workspace->weights : weights_local;
    Eigen::MatrixXd& ratio_values =
        workspace != nullptr ? workspace->y_values : ratio_values_local;
    Eigen::MatrixXd& ratio_demeaned =
        workspace != nullptr ? workspace->y_demeaned : ratio_demeaned_local;
    DemeanWorkspaceCore& ratio_demean =
        workspace != nullptr ? workspace->y_demean : ratio_demean_local;
    q.resize(exog.rows());
    q.noalias() = exog * beta;
    weights.resize(q.size());
    ratio_values.resize(q.size(), 1);
    for (Eigen::Index row = 0; row < q.size(); ++row) {
        weights(row) = exp_guard_scalar((2.0 - tau) * q(row));
        ratio_values(row, 0) = endog(row) * exp_guard_scalar(-q(row));
    }
    if (!all_finite_vec(weights) || !all_finite_mat(ratio_values)) {
        return 1.0;
    }
    bool ratio_demean_converged = false;
    if (std::abs(tau - 2.0) <= 1e-10) {
        demean_for_group_bundle_into_core(
            ratio_values,
            bundle,
            nullptr,
            control.fe_demean_tol,
            control.fe_demean_maxiter,
            ratio_demeaned,
            ratio_demean);
        ratio_demean_converged = true;
    } else {
        demean_for_group_bundle_into_core(
            ratio_values,
            bundle,
            &weights,
            control.fe_demean_tol,
            phase2_weighted_demean_maxiter_core(control),
            ratio_demeaned,
            ratio_demean,
            &ratio_demean_converged);
    }
    if (!ratio_demean_converged || ratio_demeaned.size() == 0 ||
        !all_finite_mat(ratio_demeaned)) {
        return 1.0;
    }
    const double scale = ratio_demeaned.col(0).cwiseAbs().maxCoeff();
    if (!std::isfinite(scale)) {
        return 1.0;
    }
    return std::max(scale, 1.0);
}

struct FullcancelRhoPathCore {
    std::vector<double> values;
    double scale = 1.0;
};

FullcancelRhoPathCore rho_path_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const GroupBundleCore& bundle,
    double tau,
    std::optional<double> rho_hint,
    const FullrecoverControlCore& control,
    FullcancelPhiWorkspaceCore* workspace = nullptr) {
    FullcancelRhoPathCore out;
    out.scale =
        fullcancel_initial_scale_core(endog, exog, beta, bundle, tau, control, workspace);
    std::vector<double> scaled;
    const std::vector<double> base = exp_linspace(
        control.rho_log_min,
        control.rho_log_max,
        std::max(control.rho_grid_size, 2));
    scaled.reserve(base.size());
    for (const double value : base) {
        scaled.push_back(value * out.scale);
    }
    out.values = focus_rho_path_core(unique_positive_finite(scaled), rho_hint);
    return out;
}

double fullcancel_initial_scale_mixed_compressed_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const MixedCompressedDesignCore& design,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    double tau) {
    const Eigen::VectorXd q = mixed_compressed_linear_state_core(design, beta);
    if (!all_finite_vec(q)) {
        return 1.0;
    }
    Eigen::VectorXd weights(q.size());
    Eigen::VectorXd ratio_values(q.size());
    for (Eigen::Index row = 0; row < q.size(); ++row) {
        weights(row) = exp_guard_scalar((2.0 - tau) * q(row));
        ratio_values(row) = endog(row) * exp_guard_scalar(-q(row));
    }
    if (!all_finite_vec(weights) || !all_finite_vec(ratio_values)) {
        return 1.0;
    }
    if (std::abs(tau - 2.0) <= 1e-10) {
        const Eigen::VectorXd ratio_demeaned =
            demean_vector_oneway_core(ratio_values, cancel_groups, n_cancel_groups, nullptr);
        if (ratio_demeaned.size() == 0 || !all_finite_vec(ratio_demeaned)) {
            return 1.0;
        }
        const double scale = ratio_demeaned.cwiseAbs().maxCoeff();
        if (!std::isfinite(scale)) {
            return 1.0;
        }
        return std::max(scale, 1.0);
    }
    const Eigen::VectorXd ratio_demeaned =
        demean_vector_oneway_core(ratio_values, cancel_groups, n_cancel_groups, &weights);
    if (ratio_demeaned.size() == 0 || !all_finite_vec(ratio_demeaned)) {
        return 1.0;
    }
    const double scale = ratio_demeaned.cwiseAbs().maxCoeff();
    if (!std::isfinite(scale)) {
        return 1.0;
    }
    return std::max(scale, 1.0);
}

FullcancelRhoPathCore rho_path_fullcancel_mixed_compressed_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const MixedCompressedDesignCore& design,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    double tau,
    std::optional<double> rho_hint,
    const FullrecoverControlCore& control) {
    FullcancelRhoPathCore out;
    out.scale = fullcancel_initial_scale_mixed_compressed_core(
        endog,
        design,
        beta,
        cancel_groups,
        n_cancel_groups,
        tau);
    std::vector<double> scaled;
    const std::vector<double> base = exp_linspace(
        control.rho_log_min,
        control.rho_log_max,
        std::max(control.rho_grid_size, 2));
    scaled.reserve(base.size());
    for (const double value : base) {
        scaled.push_back(value * out.scale);
    }
    out.values = focus_rho_path_core(unique_positive_finite(scaled), rho_hint);
    return out;
}

void attach_fullcancel_rho_details_core(
    FullrecoverIterationCore& iteration,
    const FullcancelRhoPathCore& rho_bundle,
    std::optional<double> rho_hint,
    const FullrecoverControlCore& control) {
    iteration.rho_policy = "scaled_fixed_log_grid";
    iteration.fullcancel_initial_scale = rho_bundle.scale;
    iteration.rho_candidate_count = static_cast<int>(rho_bundle.values.size());
    iteration.has_rho_hint =
        rho_hint.has_value() && std::isfinite(*rho_hint);
    iteration.rho_hint = iteration.has_rho_hint
                             ? *rho_hint
                             : std::numeric_limits<double>::quiet_NaN();
    iteration.rho_log_min = control.rho_log_min;
    iteration.rho_log_max = control.rho_log_max;
    iteration.rho_grid_size = control.rho_grid_size;
    iteration.has_rho_min = !rho_bundle.values.empty();
    iteration.has_rho_max = !rho_bundle.values.empty();
    if (!rho_bundle.values.empty()) {
        iteration.rho_min = rho_bundle.values.front();
        iteration.rho_max = rho_bundle.values.back();
    } else {
        iteration.rho_min = std::numeric_limits<double>::quiet_NaN();
        iteration.rho_max = std::numeric_limits<double>::quiet_NaN();
    }
}

bool phi_step_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const GroupBundleCore& bundle,
    double tau,
    int phase,
    double param,
    const FullrecoverControlCore& control,
    Eigen::VectorXd& beta_new,
    const Eigen::MatrixXd* unweighted_x_demeaned = nullptr,
    FullcancelPhiWorkspaceCore* workspace = nullptr) {
    Eigen::VectorXd q_local;
    Eigen::VectorXd& q = workspace != nullptr ? workspace->q : q_local;
    q.resize(exog.rows());
    q.noalias() = exog * beta;
    if (!all_finite_vec(q)) {
        beta_new = beta;
        return false;
    }

    if (phase == 0) {
        Eigen::MatrixXd y_values_local;
        Eigen::MatrixXd y_demeaned_local;
        DemeanWorkspaceCore y_demean_local;
        Eigen::MatrixXd& y_values =
            workspace != nullptr ? workspace->y_values : y_values_local;
        Eigen::MatrixXd& y_demeaned =
            workspace != nullptr ? workspace->y_demeaned : y_demeaned_local;
        DemeanWorkspaceCore& y_demean =
            workspace != nullptr ? workspace->y_demean : y_demean_local;
        y_values.resize(q.size(), 1);
        for (Eigen::Index row = 0; row < q.size(); ++row) {
            const double ratio = endog(row) * exp_guard_scalar(-q(row));
            const double log_arg = (ratio - 1.0) / (1.0 + param);
            y_values(row, 0) = q(row) + std::log1p(log_arg);
        }
        if (!all_finite_mat(y_values)) {
            beta_new = beta;
            return false;
        }
        Eigen::MatrixXd x_demeaned_owned;
        const Eigen::MatrixXd* x_demeaned = unweighted_x_demeaned;
        if (x_demeaned == nullptr) {
            x_demeaned_owned = demean_for_group_bundle_core(
                exog,
                bundle,
                std::nullopt,
                control.fe_demean_tol,
                control.fe_demean_maxiter);
            x_demeaned = &x_demeaned_owned;
        }
        demean_for_group_bundle_into_core(
            y_values,
            bundle,
            nullptr,
            control.fe_demean_tol,
            control.fe_demean_maxiter,
            y_demeaned,
            y_demean);
        beta_new = min_norm_lstsq_vector_allow_empty_cols(
            *x_demeaned,
            y_demeaned.col(0));
        if (!all_finite_vec(beta_new)) {
            beta_new = beta;
            return false;
        }
        return true;
    }

    Eigen::VectorXd weights_local;
    Eigen::MatrixXd x_demeaned_local;
    Eigen::MatrixXd ratio_values_local;
    Eigen::MatrixXd ratio_demeaned_local;
    DemeanWorkspaceCore x_demean_local;
    DemeanWorkspaceCore ratio_demean_local;
    Eigen::VectorXd& weights =
        workspace != nullptr ? workspace->weights : weights_local;
    Eigen::MatrixXd& x_demeaned =
        workspace != nullptr ? workspace->x_demeaned : x_demeaned_local;
    Eigen::MatrixXd& ratio_values =
        workspace != nullptr ? workspace->y_values : ratio_values_local;
    Eigen::MatrixXd& ratio_demeaned =
        workspace != nullptr ? workspace->y_demeaned : ratio_demeaned_local;
    DemeanWorkspaceCore& x_demean =
        workspace != nullptr ? workspace->x_demean : x_demean_local;
    DemeanWorkspaceCore& ratio_demean =
        workspace != nullptr ? workspace->y_demean : ratio_demean_local;
    const bool tau2_fixed_anchor = std::abs(tau - 2.0) <= 1e-10;
    weights.resize(q.size());
    ratio_values.resize(q.size(), 1);
    for (Eigen::Index row = 0; row < q.size(); ++row) {
        weights(row) = tau2_fixed_anchor ? 1.0
                                         : exp_guard_scalar((2.0 - tau) * q(row));
        ratio_values(row, 0) = endog(row) * exp_guard_scalar(-q(row));
    }
    if (!all_finite_vec(weights) || !all_finite_mat(ratio_values)) {
        beta_new = beta;
        return false;
    }
    if (tau2_fixed_anchor) {
        Eigen::MatrixXd x_demeaned_owned;
        const Eigen::MatrixXd* x_demeaned_fixed = unweighted_x_demeaned;
        if (x_demeaned_fixed == nullptr) {
            x_demeaned_owned = demean_for_group_bundle_core(
                exog,
                bundle,
                std::nullopt,
                control.fe_demean_tol,
                control.fe_demean_maxiter);
            x_demeaned_fixed = &x_demeaned_owned;
        }
        demean_for_group_bundle_into_core(
            ratio_values,
            bundle,
            nullptr,
            control.fe_demean_tol,
            control.fe_demean_maxiter,
            ratio_demeaned,
            ratio_demean);
        if (!all_finite_mat(*x_demeaned_fixed) || !all_finite_mat(ratio_demeaned)) {
            beta_new = beta;
            return false;
        }
        const double eta = 1.0 / (1.0 + param);
        for (Eigen::Index row = 0; row < ratio_demeaned.rows(); ++row) {
            const double fitted = x_demeaned_fixed->row(row).dot(beta);
            ratio_demeaned(row, 0) = fitted + eta * ratio_demeaned(row, 0);
        }
        beta_new = min_norm_lstsq_vector_allow_empty_cols(
            *x_demeaned_fixed,
            ratio_demeaned.col(0));
        if (!all_finite_vec(beta_new)) {
            beta_new = beta;
            return false;
        }
        return true;
    }
    const int phase2_demean_maxiter = phase2_weighted_demean_maxiter_core(control);
    bool x_demean_converged = false;
    bool ratio_demean_converged = false;
    demean_for_group_bundle_into_core(
        exog,
        bundle,
        &weights,
        control.fe_demean_tol,
        phase2_demean_maxiter,
        x_demeaned,
        x_demean,
        &x_demean_converged);
    demean_for_group_bundle_into_core(
        ratio_values,
        bundle,
        &weights,
        control.fe_demean_tol,
        phase2_demean_maxiter,
        ratio_demeaned,
        ratio_demean,
        &ratio_demean_converged);
    if (!x_demean_converged || !ratio_demean_converged ||
        !all_finite_mat(x_demeaned) || !all_finite_mat(ratio_demeaned)) {
        beta_new = beta;
        return false;
    }
    const double eta = 1.0 / (1.0 + param);
    for (Eigen::Index row = 0; row < x_demeaned.rows(); ++row) {
        const double sqrt_weight = std::sqrt(weights(row));
        const double fitted = x_demeaned.row(row).dot(beta);
        ratio_demeaned(row, 0) =
            sqrt_weight * (fitted + eta * ratio_demeaned(row, 0));
        x_demeaned.row(row) *= sqrt_weight;
    }
    beta_new = min_norm_lstsq_vector_allow_empty_cols(
        x_demeaned,
        ratio_demeaned.col(0));
    if (!all_finite_vec(beta_new)) {
        beta_new = beta;
        return false;
    }
    return true;
}

bool phi_step_fullcancel_mixed_compressed_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const MixedCompressedDesignCore& design,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    double tau,
    int phase,
    double param,
    Eigen::VectorXd& beta_new,
    const MixedCompressedGramSolverCore* tau2_unweighted_solver = nullptr) {
    const Eigen::VectorXd q = mixed_compressed_linear_state_core(design, beta);
    if (!all_finite_vec(q)) {
        beta_new = beta;
        return false;
    }

    if (phase == 0) {
        Eigen::VectorXd y_values(q.size());
        for (Eigen::Index row = 0; row < q.size(); ++row) {
            const double ratio = endog(row) * exp_guard_scalar(-q(row));
            const double log_arg = (ratio - 1.0) / (1.0 + param);
            y_values(row) = q(row) + std::log1p(log_arg);
        }
        if (!all_finite_vec(y_values)) {
            beta_new = beta;
            return false;
        }
        beta_new = mixed_compressed_solve_oneway_cancel_core(
            design,
            cancel_groups,
            n_cancel_groups,
            nullptr,
            y_values,
            tau2_unweighted_solver);
        if (!all_finite_vec(beta_new)) {
            beta_new = beta;
            return false;
        }
        return true;
    }

    const bool use_tau2_unweighted_solver =
        tau2_unweighted_solver != nullptr && std::abs(tau - 2.0) <= 1e-10;
    Eigen::VectorXd weights(q.size());
    Eigen::VectorXd ratio_values(q.size());
    Eigen::VectorXd q_sums = Eigen::VectorXd::Zero(n_cancel_groups);
    Eigen::VectorXd denoms = Eigen::VectorXd::Zero(n_cancel_groups);
    for (Eigen::Index row = 0; row < q.size(); ++row) {
        weights(row) =
            use_tau2_unweighted_solver ? 1.0
                                       : exp_guard_scalar((2.0 - tau) * q(row));
        ratio_values(row) = endog(row) * exp_guard_scalar(-q(row));
        const Eigen::Index group =
            static_cast<Eigen::Index>(cancel_groups[static_cast<std::size_t>(row)]);
        q_sums(group) += weights(row) * q(row);
        denoms(group) += weights(row);
    }
    if (!all_finite_vec(weights) || !all_finite_vec(ratio_values)) {
        beta_new = beta;
        return false;
    }
    const Eigen::VectorXd ratio_demeaned =
        demean_vector_oneway_core(
            ratio_values,
            cancel_groups,
            n_cancel_groups,
            use_tau2_unweighted_solver ? nullptr : &weights);
    if (!all_finite_vec(ratio_demeaned)) {
        beta_new = beta;
        return false;
    }
    for (Eigen::Index group = 0; group < denoms.size(); ++group) {
        if (denoms(group) > 1.0e-15) {
            q_sums(group) /= denoms(group);
        } else {
            q_sums(group) = 0.0;
        }
    }
    const double eta = 1.0 / (1.0 + param);
    Eigen::VectorXd response(q.size());
    for (Eigen::Index row = 0; row < q.size(); ++row) {
        const Eigen::Index group =
            static_cast<Eigen::Index>(cancel_groups[static_cast<std::size_t>(row)]);
        const double fitted = q(row) - q_sums(group);
        response(row) = fitted + eta * ratio_demeaned(row);
    }
    if (!all_finite_vec(response)) {
        beta_new = beta;
        return false;
    }
    beta_new = mixed_compressed_solve_oneway_cancel_core(
        design,
        cancel_groups,
        n_cancel_groups,
        use_tau2_unweighted_solver ? nullptr : &weights,
        response,
        tau2_unweighted_solver);
    if (!all_finite_vec(beta_new)) {
        beta_new = beta;
        return false;
    }
    return true;
}

bool phase2_initialization_usable_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const GroupBundleCore& bundle,
    double tau,
    const FullrecoverControlCore& control,
    const Eigen::MatrixXd* unweighted_x_demeaned = nullptr,
    FullcancelPhiWorkspaceCore* workspace = nullptr) {
    if (!all_finite_vec(beta)) {
        return false;
    }
    FullcancelPhiWorkspaceCore local_workspace;
    FullcancelPhiWorkspaceCore& phi_workspace =
        workspace != nullptr ? *workspace : local_workspace;
    const FullcancelRhoPathCore rho_path =
        rho_path_fullcancel_core(
            endog,
            exog,
            beta,
            bundle,
            tau,
            std::nullopt,
            control,
            &phi_workspace);
    for (const double rho : rho_path.values) {
        Eigen::VectorXd beta_test(beta.size());
        if (phi_step_fullcancel_core(
                endog,
                exog,
                beta,
                bundle,
                tau,
                1,
                rho,
                control,
                beta_test,
                unweighted_x_demeaned,
                &phi_workspace)) {
            return true;
        }
    }
    return false;
}

bool phase2_initialization_usable_fullcancel_mixed_compressed_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const MixedCompressedDesignCore& design,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    double tau,
    const FullrecoverControlCore& control,
    const MixedCompressedGramSolverCore* tau2_unweighted_solver = nullptr) {
    if (!all_finite_vec(beta)) {
        return false;
    }
    const FullcancelRhoPathCore rho_path =
        rho_path_fullcancel_mixed_compressed_core(
            endog,
            design,
            beta,
            cancel_groups,
            n_cancel_groups,
            tau,
            std::nullopt,
            control);
    for (const double rho : rho_path.values) {
        Eigen::VectorXd beta_test(beta.size());
        if (phi_step_fullcancel_mixed_compressed_core(
                endog,
                design,
                beta,
                cancel_groups,
                n_cancel_groups,
                tau,
                1,
                rho,
                beta_test,
                tau2_unweighted_solver)) {
            return true;
        }
    }
    return false;
}

double qhat_from_beta_path_core(const std::vector<Eigen::VectorXd>& beta_path) {
    if (beta_path.size() < 4) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::VectorXd d1 =
        (beta_path[beta_path.size() - 1] - beta_path[beta_path.size() - 2]).cwiseAbs();
    const Eigen::VectorXd d2 =
        (beta_path[beta_path.size() - 2] - beta_path[beta_path.size() - 3]).cwiseAbs();
    const Eigen::VectorXd d3 =
        (beta_path[beta_path.size() - 3] - beta_path[beta_path.size() - 4]).cwiseAbs();
    double sum = 0.0;
    int count = 0;
    for (Eigen::Index idx = 0; idx < d1.size(); ++idx) {
        const double value = std::log(d1(idx) / d2(idx)) /
                             std::log(d2(idx) / d3(idx));
        if (!std::isnan(value)) {
            sum += value;
            count += 1;
        }
    }
    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(count);
}

double kappa_monitor_from_beta_path_core(
    const std::vector<Eigen::VectorXd>& beta_path) {
    if (beta_path.size() < 4) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double kappa = -std::numeric_limits<double>::infinity();
    bool has_value = false;
    for (std::size_t row = 3; row < beta_path.size(); ++row) {
        const Eigen::VectorXd d_now = (beta_path[row] - beta_path[row - 1]).cwiseAbs();
        const Eigen::VectorXd d_prev =
            (beta_path[row - 1] - beta_path[row - 2]).cwiseAbs();
        double mean_ratio = 0.0;
        int count = 0;
        for (Eigen::Index idx = 0; idx < d_now.size(); ++idx) {
            const double ratio = d_now(idx) / d_prev(idx);
            if (!std::isnan(ratio)) {
                mean_ratio += ratio;
                count += 1;
            }
        }
        if (count > 0) {
            kappa = std::max(kappa, mean_ratio / static_cast<double>(count));
            has_value = true;
        }
    }
    return has_value ? kappa : std::numeric_limits<double>::quiet_NaN();
}

FullrecoverConcentratedCore concentrated_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const GroupBundleCore& bundle,
    double tau,
    const FullrecoverControlCore& control,
    std::optional<Eigen::VectorXd> beta_init,
    std::optional<double> rho_hint,
    FullcancelPhiWorkspaceCore* workspace = nullptr) {
    if (!beta_init.has_value() && std::abs(tau - 2.0) > 1e-10) {
        throw std::invalid_argument(
            "Non-anchor fullcancel path steps require beta_init from tau=2");
    }
    const std::vector<double> delta_grid =
        beta_init.has_value() ? std::vector<double>{} : delta_grid_core(control);
    Eigen::MatrixXd x_demeaned;
    Eigen::VectorXd beta;
    int n_outer = 0;
    if (beta_init.has_value()) {
        beta = *beta_init;
        n_outer = 1;
    } else {
        x_demeaned = demean_for_group_bundle_core(
            exog,
            bundle,
            std::nullopt,
            control.fe_demean_tol,
            control.fe_demean_maxiter);
        Eigen::VectorXd log_y(endog.size());
        for (Eigen::Index row = 0; row < endog.size(); ++row) {
            log_y(row) = std::log(endog(row) + 1.0);
        }
        const Eigen::VectorXd y_init = demean_vector_for_group_bundle_core(
            log_y,
            bundle,
            std::nullopt,
            control.fe_demean_tol,
            control.fe_demean_maxiter);
        try {
            beta = min_norm_lstsq_vector_allow_empty_cols(x_demeaned, y_init);
        } catch (const std::exception&) {
            beta = Eigen::VectorXd::Constant(
                exog.cols(),
                std::numeric_limits<double>::quiet_NaN());
        }
        n_outer = static_cast<int>(delta_grid.size()) + 1;
    }
    if (std::abs(tau - 2.0) <= 1e-10 && x_demeaned.size() == 0) {
        x_demeaned = demean_for_group_bundle_core(
            exog,
            bundle,
            std::nullopt,
            control.fe_demean_tol,
            control.fe_demean_maxiter);
    }

    const Eigen::VectorXd beta_initial = beta;
    Eigen::VectorXd phase1_accepted_beta = beta;
    int phase1_attempts = 0;
    int phase1_accepted = 0;
    int phase1_rejected = 0;
    FullrecoverIterationCore final_diag = not_started_iteration_core(control);
    FullcancelPhiWorkspaceCore local_phi_workspace;
    FullcancelPhiWorkspaceCore& phi_workspace =
        workspace != nullptr ? *workspace : local_phi_workspace;
    std::vector<double> fullcancel_rho_attempt_values;
    std::vector<int> fullcancel_rho_attempt_iterations;
    std::vector<int> fullcancel_rho_attempt_restarts;
    std::vector<int> fullcancel_rho_attempt_phase2_evaluations;
    std::vector<double> fullcancel_rho_attempt_final_criteria;
    std::vector<double> fullcancel_rho_attempt_post_residuals;
    std::vector<double> fullcancel_rho_attempt_post_score_residuals;
    std::vector<std::string> fullcancel_rho_attempt_failure_reasons;
    std::vector<bool> fullcancel_rho_attempt_converged;

    for (int outer_index = 0; outer_index < n_outer; ++outer_index) {
        if (any_nan_vec(beta)) {
            beta = beta_initial;
        }
        Eigen::VectorXd beta_outer_initial = beta;
        int phase = 0;
        double param = 0.0;
        double eps_current = control.eps;
        int stopcrit_current = control.stopcrit;
        std::vector<double> rho_path;
        FullcancelRhoPathCore rho_bundle;
        if (outer_index < static_cast<int>(delta_grid.size())) {
            phase = 0;
            param = delta_grid[static_cast<std::size_t>(outer_index)];
            eps_current = std::max(control.eps * 10.0, 1e-5);
            stopcrit_current =
                std::min(control.stopcrit, control.max_phase1_iterations);
        } else {
            phase = 1;
            if (!beta_init.has_value()) {
                beta = phase1_accepted_beta;
            }
            beta_outer_initial = beta;
            rho_bundle = rho_path_fullcancel_core(
                endog,
                exog,
                beta,
                bundle,
                tau,
                rho_hint,
                control,
                &phi_workspace);
            rho_path = rho_bundle.values;
            if (rho_path.empty()) {
                FullrecoverConcentratedCore out;
                out.beta = beta;
                out.has_linear = false;
                out.iteration = not_started_iteration_core(control);
                out.iteration.phase = "P2";
                out.iteration.failure_reason = "empty_finite_rho_path";
                out.iteration.algorithm = "weighted_within_fullcancel";
                out.iteration.index_policy = "slope_only_raw_exog";
                out.iteration.fe_policy = "weighted_within_cancel_all";
                out.iteration.phase1_attempts = phase1_attempts;
                out.iteration.phase1_accepted = phase1_accepted;
                out.iteration.phase1_rejected = phase1_rejected;
                attach_fullcancel_rho_details_core(
                    out.iteration, rho_bundle, rho_hint, control);
                return out;
            }
            param = rho_path.front();
            eps_current = control.eps;
            stopcrit_current = control.stopcrit;
        }

        int restart_index = 0;
        bool verified = false;
        int phase2_evaluations = 0;
        bool phase2_budget_exhausted = false;
        int iteration_count = 0;
        bool has_final_criterion = false;
        double final_criterion = 0.0;
        int aa_m_final = control.anderson_m;
        std::string failure_reason;
        double qhat_last = std::numeric_limits<double>::quiet_NaN();
        int qhat_invalid_count = 0;
        double kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
        double kappa_monitor_max = std::numeric_limits<double>::quiet_NaN();
        int kappa_monitor_checks = 0;
        double post_convergence_residual = std::numeric_limits<double>::quiet_NaN();
        double post_convergence_kappa_est = std::numeric_limits<double>::quiet_NaN();
        double post_convergence_score_residual =
            std::numeric_limits<double>::quiet_NaN();
        double score_tol = std::numeric_limits<double>::quiet_NaN();
        std::string score_validation_status = "not_checked";
        std::vector<std::string> score_warning_codes;
        std::vector<std::string> score_warning_messages;
        bool score_exceeds_tolerance = false;
        double last_finite_criterion = std::numeric_limits<double>::quiet_NaN();
        double best_finite_criterion = std::numeric_limits<double>::quiet_NaN();

        while (!verified) {
            int k = 1;
            double criterion = 100.0;
            std::vector<double> criterion_history;
            int kappa_bad_count = 0;
            std::vector<Eigen::VectorXd> beta_path;
            EigenVectorPairRingCore aa_hist;
            beta_path.push_back(beta);
            int aa_m = control.anderson_m;
            criterion_history.reserve(
                static_cast<std::size_t>(std::max(stopcrit_current, 0)));
            beta_path.reserve(
                static_cast<std::size_t>(std::max(stopcrit_current + 1, 1)));
            if (aa_m > 0) {
                aa_hist.reset(static_cast<std::size_t>(aa_m + 1));
            }

            while (criterion > eps_current && k < stopcrit_current &&
                   !std::isnan(criterion)) {
                const Eigen::VectorXd beta0 = beta;
                if (phase == 1) {
                    phase2_evaluations += 1;
                    if (phase2_evaluations > control.max_phase2_evaluations) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "phase2_evaluation_budget_exhausted";
                        phase2_budget_exhausted = true;
                        break;
                    }
                }
                Eigen::VectorXd beta_phi(beta.size());
                const Eigen::MatrixXd* fixed_x_demeaned =
                    (phase == 0 || std::abs(tau - 2.0) <= 1e-10)
                        ? &x_demeaned
                        : nullptr;
                const bool valid = phi_step_fullcancel_core(
                    endog,
                    exog,
                    beta,
                    bundle,
                    tau,
                    phase,
                    param,
                    control,
                    beta_phi,
                    fixed_x_demeaned,
                    &phi_workspace);
                if (!valid) {
                    criterion = std::numeric_limits<double>::quiet_NaN();
                    failure_reason = "invalid_phi_step";
                    break;
                }

                bool accepted_aa_step = false;
                if (aa_m > 0 && k > aa_m + 1) {
                    const Eigen::VectorXd f_k = beta_phi - beta;
                    aa_hist.push(beta, f_k);
                    const int m_cur = static_cast<int>(aa_hist.size()) - 1;
                    if (m_cur >= 1) {
                        Eigen::MatrixXd f_mat(beta.size(), m_cur);
                        for (int j = 1; j <= m_cur; ++j) {
                            f_mat.col(j - 1) =
                                aa_hist.f_back() -
                                aa_hist.f_at(aa_hist.size() - static_cast<std::size_t>(j + 1));
                        }
                        Eigen::VectorXd gamma =
                            pyiols::kernel::min_norm_lstsq_anderson(
                                f_mat,
                                aa_hist.f_back())
                                .col(0);
                        Eigen::VectorXd beta_aa = beta_phi;
                        for (int j = 0; j < m_cur; ++j) {
                            const std::size_t idx =
                                aa_hist.size() - static_cast<std::size_t>(j + 2);
                            const Eigen::VectorXd phi_prev =
                                aa_hist.x_at(idx) + aa_hist.f_at(idx);
                            beta_aa += gamma(j) * (phi_prev - beta_phi);
                        }
                        if (all_finite_vec(beta_aa)) {
                            if (phase == 1) {
                                phase2_evaluations += 1;
                                if (phase2_evaluations >
                                    control.max_phase2_evaluations) {
                                    criterion =
                                        std::numeric_limits<double>::quiet_NaN();
                                    failure_reason =
                                        "phase2_evaluation_budget_exhausted";
                                    phase2_budget_exhausted = true;
                                    break;
                                }
                            }
                            Eigen::VectorXd beta_aa_phi(beta.size());
                            const bool aa_valid = phi_step_fullcancel_core(
                                endog,
                                exog,
                                beta_aa,
                                bundle,
                                tau,
                                phase,
                                param,
                                control,
                                beta_aa_phi,
                                fixed_x_demeaned,
                                &phi_workspace);
                            if (aa_valid) {
                                const Eigen::VectorXd plain_residual = beta_phi - beta;
                                const Eigen::VectorXd aa_residual =
                                    beta_aa_phi - beta_aa;
                                const double res_plain =
                                    pyiols::kernel::stable_l2_norm(plain_residual);
                                const double res_aa =
                                    pyiols::kernel::stable_l2_norm(aa_residual);
                                if (res_aa < res_plain * 1.1) {
                                    beta = beta_aa;
                                    beta_path.push_back(beta);
                                    if (k >= 2) {
                                        criterion = max_abs_change_core(beta, beta0);
                                    }
                                    accepted_aa_step = true;
                                }
                            }
                        }
                    }
                } else if (aa_m > 0) {
                    aa_hist.push(beta, beta_phi - beta);
                }

                if (!accepted_aa_step) {
                    beta = beta_phi;
                    beta_path.push_back(beta);
                }
                if (k >= 2) {
                    if (!accepted_aa_step) {
                        criterion = max_abs_change_core(beta, beta0);
                    }
                    if (record_criterion_progress_core(
                            criterion_history,
                            criterion,
                            phase,
                            tau,
                            eps_current,
                            control.max_phase2_evaluations,
                            last_finite_criterion,
                            best_finite_criterion,
                            failure_reason)) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                    }
                }

                if (k > 4 && beta_path.size() >= 4 &&
                    !(phase == 1 && std::abs(tau - 2.0) <= 1e-10)) {
                    const double qhat = qhat_from_beta_path_core(beta_path);
                    qhat_last = qhat;
                    if (std::isinf(std::abs(qhat)) && criterion > eps_current * 10.0) {
                        qhat_invalid_count += 1;
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "q_convergence_invalid";
                    }
                }

                if (phase == 1 && std::abs(tau - 2.0) > 1e-10 && k >= 30 &&
                    k % 10 == 0 && beta_path.size() >= 3) {
                    const double kappa_val = kappa_monitor_from_beta_path_core(beta_path);
                    kappa_monitor_last = kappa_val;
                    kappa_monitor_checks += 1;
                    if (std::isfinite(kappa_val)) {
                        kappa_monitor_max = std::isfinite(kappa_monitor_max)
                                                ? std::max(kappa_monitor_max, kappa_val)
                                                : kappa_val;
                    }
                    const int aa_m_new = adapt_anderson_m_core(aa_m, kappa_val);
                    if (aa_m_new > aa_m) {
                        aa_m = aa_m_new;
                        kappa_bad_count = 0;
                    } else if (std::isfinite(kappa_val) && kappa_val >= 1.0) {
                        kappa_bad_count += 1;
                    } else {
                        kappa_bad_count = 0;
                    }
                    if (kappa_bad_count >= 3) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "noncontractive";
                    }
                }

                k += 1;
                iteration_count = k;

                if (std::isnan(criterion)) {
                    if (phase == 1) {
                        fullcancel_rho_attempt_values.push_back(param);
                        fullcancel_rho_attempt_iterations.push_back(iteration_count);
                        fullcancel_rho_attempt_restarts.push_back(restart_index);
                        fullcancel_rho_attempt_phase2_evaluations.push_back(
                            phase2_evaluations);
                        fullcancel_rho_attempt_final_criteria.push_back(
                            std::numeric_limits<double>::quiet_NaN());
                        fullcancel_rho_attempt_post_residuals.push_back(
                            post_convergence_residual);
                        fullcancel_rho_attempt_post_score_residuals.push_back(
                            post_convergence_score_residual);
                        fullcancel_rho_attempt_failure_reasons.push_back(
                            failure_reason);
                        fullcancel_rho_attempt_converged.push_back(false);
                    }
                    restart_index += 1;
                    if (phase == 1 && !phase2_budget_exhausted &&
                        restart_index < static_cast<int>(rho_path.size())) {
                        param = rho_path[static_cast<std::size_t>(restart_index)];
                        criterion = 100.0;
                        beta = beta_outer_initial;
                        k = 1;
                        criterion_history.clear();
                        kappa_bad_count = 0;
                        beta_path.clear();
                        beta_path.push_back(beta);
                        aa_hist.clear();
                    } else if (phase == 0) {
                        break;
                    }
                }
            }

            bool converged = !std::isnan(criterion) && criterion <= eps_current;
            has_final_criterion = !std::isnan(criterion);
            final_criterion = criterion;
            if (converged) {
                Eigen::VectorXd beta_check(beta.size());
                const Eigen::MatrixXd* check_x_demeaned =
                    (phase == 0 || std::abs(tau - 2.0) <= 1e-10)
                        ? &x_demeaned
                        : nullptr;
                const bool valid = phi_step_fullcancel_core(
                    endog,
                    exog,
                    beta,
                    bundle,
                    tau,
                    phase,
                    param,
                    control,
                    beta_check,
                    check_x_demeaned,
                    &phi_workspace);
                if (valid) {
                    post_convergence_residual =
                        max_abs_change_core(beta_check, beta);
                    if (post_convergence_residual > eps_current * 10.0) {
                        converged = false;
                        failure_reason = "post_convergence_residual";
                    }
                }
                if (converged && phase == 1) {
                    const double score_residual = fullcancel_score_residual_core(
                        endog,
                        exog,
                        beta,
                        bundle,
                        tau,
                        control,
                        std::abs(tau - 2.0) <= 1e-10 ? &x_demeaned : nullptr,
                        &phi_workspace);
                    score_tol = std::max(eps_current * 10.0, 1e-8);
                    post_convergence_score_residual = score_residual;
                    score_exceeds_tolerance =
                        !std::isfinite(score_residual) || score_residual > score_tol;
                    if (score_exceeds_tolerance) {
                        score_warning_codes.push_back("post_convergence_score_residual");
                        score_warning_messages.push_back(
                            "Fullcancel weighted-within score residual exceeds "
                            "tolerance at this tau-path state.");
                        score_validation_status = "diagnostic_warning";
                    } else if (score_validation_status == "not_checked") {
                        score_validation_status = "passed";
                    }
                }
            if (converged && beta_path.size() >= 4 &&
                !(phase == 1 && std::abs(tau - 2.0) <= 1e-10)) {
                    const double d0 =
                        max_abs_change_core(beta_path[beta_path.size() - 1],
                                            beta_path[beta_path.size() - 2]);
                    const double d1 =
                        max_abs_change_core(beta_path[beta_path.size() - 2],
                                            beta_path[beta_path.size() - 3]);
                    const double d2 =
                        max_abs_change_core(beta_path[beta_path.size() - 3],
                                            beta_path[beta_path.size() - 4]);
                    if (d1 > 1e-15 && d2 > 1e-15) {
                        const double kappa_est = d0 / d1;
                        post_convergence_kappa_est = kappa_est;
                        if (kappa_est >= 1.0) {
                            converged = false;
                            failure_reason = "post_convergence_noncontractive";
                        }
                    }
                }
            }

            if (phase == 1 &&
                (fullcancel_rho_attempt_values.empty() ||
                 fullcancel_rho_attempt_values.back() != param ||
                 fullcancel_rho_attempt_phase2_evaluations.back() !=
                     phase2_evaluations ||
                 fullcancel_rho_attempt_converged.back() != converged)) {
                fullcancel_rho_attempt_values.push_back(param);
                fullcancel_rho_attempt_iterations.push_back(iteration_count);
                fullcancel_rho_attempt_restarts.push_back(restart_index);
                fullcancel_rho_attempt_phase2_evaluations.push_back(
                    phase2_evaluations);
                fullcancel_rho_attempt_final_criteria.push_back(
                    has_final_criterion ? final_criterion
                                        : std::numeric_limits<double>::quiet_NaN());
                fullcancel_rho_attempt_post_residuals.push_back(
                    post_convergence_residual);
                fullcancel_rho_attempt_post_score_residuals.push_back(
                    post_convergence_score_residual);
                fullcancel_rho_attempt_failure_reasons.push_back(
                    converged ? std::string() : failure_reason);
                fullcancel_rho_attempt_converged.push_back(converged);
            }
            if (!converged && phase == 1 && !phase2_budget_exhausted &&
                restart_index < static_cast<int>(rho_path.size()) - 1) {
                restart_index = std::min(
                    restart_index + std::max(restart_index, 5),
                    static_cast<int>(rho_path.size()) - 1);
                param = rho_path[static_cast<std::size_t>(restart_index)];
                beta = beta_outer_initial;
            } else {
                verified = true;
            }
            aa_m_final = aa_m;

            if (phase == 0) {
                phase1_attempts += 1;
                const bool candidate_usable =
                    converged &&
                    phase2_initialization_usable_fullcancel_core(
                        endog,
                        exog,
                        beta,
                        bundle,
                        tau,
                        control,
                        std::abs(tau - 2.0) <= 1e-10 ? &x_demeaned : nullptr,
                        &phi_workspace);
                if (candidate_usable) {
                    phase1_accepted += 1;
                    phase1_accepted_beta = beta;
                } else {
                    phase1_rejected += 1;
                    beta = phase1_accepted_beta;
                    if (converged && failure_reason.empty()) {
                        failure_reason = "phase1_candidate_unusable";
                    }
                }
                final_diag.converged = candidate_usable;
            } else {
                if (!converged) {
                    beta = beta_outer_initial;
                }
                final_diag.converged = converged;
            }

            final_diag.phase = phase == 0 ? "P1" : "P2";
            final_diag.iterations = iteration_count;
            final_diag.restarts = restart_index;
            final_diag.has_final_rho = phase != 0;
            final_diag.final_rho = param;
            final_diag.rho_path.clear();
            if (phase != 0) {
                const int upto =
                    std::min(restart_index + 1, static_cast<int>(rho_path.size()));
                for (int idx = 0; idx < upto; ++idx) {
                    final_diag.rho_path.push_back(rho_path[static_cast<std::size_t>(idx)]);
                }
            }
            final_diag.has_final_criterion = has_final_criterion;
            final_diag.final_criterion = final_criterion;
            final_diag.anderson_m_initial = control.anderson_m;
            final_diag.anderson_m_final = aa_m_final;
            final_diag.failure_reason =
                final_diag.converged
                    ? ""
                    : (!failure_reason.empty() ? failure_reason : "not_converged");
            final_diag.phase1_attempts = phase1_attempts;
            final_diag.phase1_accepted = phase1_accepted;
            final_diag.phase1_rejected = phase1_rejected;
            final_diag.phase2_evaluations = phase2_evaluations;
            final_diag.max_phase2_evaluations = control.max_phase2_evaluations;
            final_diag.last_finite_criterion = last_finite_criterion;
            final_diag.best_finite_criterion = best_finite_criterion;
            if (phase == 1) {
                attach_fullcancel_rho_details_core(
                    final_diag, rho_bundle, rho_hint, control);
            }
            final_diag.algorithm = "weighted_within_fullcancel";
            final_diag.index_policy = "slope_only_raw_exog";
            final_diag.fe_policy = "weighted_within_cancel_all";
            final_diag.anderson_policy = "beta_anderson";
            final_diag.qhat_last = qhat_last;
            final_diag.qhat_invalid_count = qhat_invalid_count;
            final_diag.kappa_monitor_last = kappa_monitor_last;
            final_diag.kappa_monitor_max = kappa_monitor_max;
            final_diag.kappa_monitor_checks = kappa_monitor_checks;
            final_diag.post_convergence_residual = post_convergence_residual;
            final_diag.post_convergence_kappa_est = post_convergence_kappa_est;
            final_diag.post_convergence_score_residual =
                post_convergence_score_residual;
            final_diag.score_tolerance = score_tol;
            final_diag.score_validation_policy = "diagnostic";
            final_diag.score_validation_status = score_validation_status;
            final_diag.score_warning_codes = score_warning_codes;
            final_diag.score_warning_messages = score_warning_messages;
            final_diag.post_convergence_score_exceeds_tolerance =
                score_exceeds_tolerance;
            final_diag.fullcancel_rho_attempt_values =
                fullcancel_rho_attempt_values;
            final_diag.fullcancel_rho_attempt_iterations =
                fullcancel_rho_attempt_iterations;
            final_diag.fullcancel_rho_attempt_restarts =
                fullcancel_rho_attempt_restarts;
            final_diag.fullcancel_rho_attempt_phase2_evaluations =
                fullcancel_rho_attempt_phase2_evaluations;
            final_diag.fullcancel_rho_attempt_final_criteria =
                fullcancel_rho_attempt_final_criteria;
            final_diag.fullcancel_rho_attempt_post_residuals =
                fullcancel_rho_attempt_post_residuals;
            final_diag.fullcancel_rho_attempt_post_score_residuals =
                fullcancel_rho_attempt_post_score_residuals;
            final_diag.fullcancel_rho_attempt_failure_reasons =
                fullcancel_rho_attempt_failure_reasons;
            final_diag.fullcancel_rho_attempt_converged =
                fullcancel_rho_attempt_converged;
        }
        if (phase == 1) {
            break;
        }
    }

    FullrecoverConcentratedCore out;
    out.beta = beta;
    if (all_finite_vec(beta)) {
        out.linear = fullcancel_linear_state_core(exog, beta);
        out.has_linear = all_finite_vec(out.linear);
    }
    out.iteration = final_diag;
    return out;
}

FullrecoverConcentratedCore concentrated_fullcancel_mixed_compressed_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const MixedCompressedDesignCore& design,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    double tau,
    const FullrecoverControlCore& control,
    std::optional<Eigen::VectorXd> beta_init,
    std::optional<double> rho_hint,
    bool require_tau2_coordinate_safe = false) {
    if (!beta_init.has_value() && std::abs(tau - 2.0) > 1e-10) {
        throw std::invalid_argument(
            "Non-anchor fullcancel path steps require beta_init from tau=2");
    }
    const std::vector<double> delta_grid =
        beta_init.has_value() ? std::vector<double>{} : delta_grid_core(control);
    const std::optional<MixedCompressedGramSolverCore> tau2_unweighted_solver =
        mixed_compressed_unweighted_solver_core(
            design,
            cancel_groups,
            n_cancel_groups,
            endog.size());
    const MixedCompressedGramSolverCore* tau2_solver_candidate =
        tau2_unweighted_solver.has_value() ? &(*tau2_unweighted_solver) : nullptr;
    const bool tau2_solver_direct_safe =
        tau2_solver_candidate != nullptr &&
        mixed_compressed_solver_well_conditioned_core(*tau2_solver_candidate);
    const MixedCompressedGramSolverCore* tau2_solver_ptr =
        !require_tau2_coordinate_safe || tau2_solver_direct_safe
            ? tau2_solver_candidate
            : nullptr;
    Eigen::VectorXd beta;
    int n_outer = 0;
    if (beta_init.has_value()) {
        beta = *beta_init;
        n_outer = 1;
    } else {
        Eigen::VectorXd log_y(endog.size());
        for (Eigen::Index row = 0; row < endog.size(); ++row) {
            log_y(row) = std::log(endog(row) + 1.0);
        }
        beta = mixed_compressed_solve_oneway_cancel_core(
            design,
            cancel_groups,
            n_cancel_groups,
            nullptr,
            log_y,
            tau2_solver_ptr);
        n_outer = static_cast<int>(delta_grid.size()) + 1;
    }

    const Eigen::VectorXd beta_initial = beta;
    Eigen::VectorXd phase1_accepted_beta = beta;
    int phase1_attempts = 0;
    int phase1_accepted = 0;
    int phase1_rejected = 0;
    FullrecoverIterationCore final_diag = not_started_iteration_core(control);

    for (int outer_index = 0; outer_index < n_outer; ++outer_index) {
        if (any_nan_vec(beta)) {
            beta = beta_initial;
        }
        Eigen::VectorXd beta_outer_initial = beta;
        int phase = 0;
        double param = 0.0;
        double eps_current = control.eps;
        int stopcrit_current = control.stopcrit;
        std::vector<double> rho_path;
        FullcancelRhoPathCore rho_bundle;
        if (outer_index < static_cast<int>(delta_grid.size())) {
            phase = 0;
            param = delta_grid[static_cast<std::size_t>(outer_index)];
            eps_current = std::max(control.eps * 10.0, 1e-5);
            stopcrit_current =
                std::min(control.stopcrit, control.max_phase1_iterations);
        } else {
            phase = 1;
            if (!beta_init.has_value()) {
                beta = phase1_accepted_beta;
            }
            beta_outer_initial = beta;
            rho_bundle = rho_path_fullcancel_mixed_compressed_core(
                endog,
                design,
                beta,
                cancel_groups,
                n_cancel_groups,
                tau,
                rho_hint,
                control);
            rho_path = rho_bundle.values;
            if (rho_path.empty()) {
                FullrecoverConcentratedCore out;
                out.beta = beta;
                out.has_linear = false;
                out.iteration = not_started_iteration_core(control);
                out.iteration.phase = "P2";
                out.iteration.failure_reason = "empty_finite_rho_path";
                out.iteration.algorithm = "weighted_within_fullcancel";
                out.iteration.index_policy = "slope_only_raw_exog";
                out.iteration.fe_policy = "weighted_within_cancel_all";
                out.iteration.phase1_attempts = phase1_attempts;
                out.iteration.phase1_accepted = phase1_accepted;
                out.iteration.phase1_rejected = phase1_rejected;
                attach_fullcancel_rho_details_core(
                    out.iteration, rho_bundle, rho_hint, control);
                return out;
            }
            param = rho_path.front();
            eps_current = control.eps;
            stopcrit_current = control.stopcrit;
        }

        int restart_index = 0;
        bool verified = false;
        int phase2_evaluations = 0;
        bool phase2_budget_exhausted = false;
        int iteration_count = 0;
        bool has_final_criterion = false;
        double final_criterion = 0.0;
        int aa_m_final = control.anderson_m;
        std::string failure_reason;
        double qhat_last = std::numeric_limits<double>::quiet_NaN();
        int qhat_invalid_count = 0;
        double kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
        double kappa_monitor_max = std::numeric_limits<double>::quiet_NaN();
        int kappa_monitor_checks = 0;
        double post_convergence_residual = std::numeric_limits<double>::quiet_NaN();
        double post_convergence_kappa_est = std::numeric_limits<double>::quiet_NaN();
        double post_convergence_score_residual =
            std::numeric_limits<double>::quiet_NaN();
        double score_tol = std::numeric_limits<double>::quiet_NaN();
        std::string score_validation_status = "not_checked";
        std::vector<std::string> score_warning_codes;
        std::vector<std::string> score_warning_messages;
        bool score_exceeds_tolerance = false;
        double last_finite_criterion = std::numeric_limits<double>::quiet_NaN();
        double best_finite_criterion = std::numeric_limits<double>::quiet_NaN();

        while (!verified) {
            int k = 1;
            double criterion = 100.0;
            std::vector<double> criterion_history;
            int kappa_bad_count = 0;
            std::vector<Eigen::VectorXd> beta_path;
            EigenVectorPairRingCore aa_hist;
            beta_path.push_back(beta);
            int aa_m = control.anderson_m;
            criterion_history.reserve(
                static_cast<std::size_t>(std::max(stopcrit_current, 0)));
            beta_path.reserve(
                static_cast<std::size_t>(std::max(stopcrit_current + 1, 1)));
            if (aa_m > 0) {
                aa_hist.reset(static_cast<std::size_t>(aa_m + 1));
            }

            while (criterion > eps_current && k < stopcrit_current &&
                   !std::isnan(criterion)) {
                const Eigen::VectorXd beta0 = beta;
                if (phase == 1) {
                    phase2_evaluations += 1;
                    if (phase2_evaluations > control.max_phase2_evaluations) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "phase2_evaluation_budget_exhausted";
                        phase2_budget_exhausted = true;
                        break;
                    }
                }
                Eigen::VectorXd beta_phi(beta.size());
                const bool valid = phi_step_fullcancel_mixed_compressed_core(
                    endog,
                    design,
                    beta,
                    cancel_groups,
                    n_cancel_groups,
                    tau,
                    phase,
                    param,
                    beta_phi,
                    tau2_solver_ptr);
                if (!valid) {
                    criterion = std::numeric_limits<double>::quiet_NaN();
                    failure_reason = "invalid_phi_step";
                    break;
                }

                bool accepted_aa_step = false;
                if (aa_m > 0 && k > aa_m + 1) {
                    const Eigen::VectorXd f_k = beta_phi - beta;
                    aa_hist.push(beta, f_k);
                    const int m_cur = static_cast<int>(aa_hist.size()) - 1;
                    if (m_cur >= 1) {
                        Eigen::MatrixXd f_mat(beta.size(), m_cur);
                        for (int j = 1; j <= m_cur; ++j) {
                            f_mat.col(j - 1) =
                                aa_hist.f_back() -
                                aa_hist.f_at(aa_hist.size() - static_cast<std::size_t>(j + 1));
                        }
                        Eigen::VectorXd gamma =
                            pyiols::kernel::min_norm_lstsq_anderson(
                                f_mat,
                                aa_hist.f_back())
                                .col(0);
                        Eigen::VectorXd beta_aa = beta_phi;
                        for (int j = 0; j < m_cur; ++j) {
                            const std::size_t idx =
                                aa_hist.size() - static_cast<std::size_t>(j + 2);
                            const Eigen::VectorXd phi_prev =
                                aa_hist.x_at(idx) + aa_hist.f_at(idx);
                            beta_aa += gamma(j) * (phi_prev - beta_phi);
                        }
                        if (all_finite_vec(beta_aa)) {
                            if (phase == 1) {
                                phase2_evaluations += 1;
                                if (phase2_evaluations >
                                    control.max_phase2_evaluations) {
                                    criterion =
                                        std::numeric_limits<double>::quiet_NaN();
                                    failure_reason =
                                        "phase2_evaluation_budget_exhausted";
                                    phase2_budget_exhausted = true;
                                    break;
                                }
                            }
                            Eigen::VectorXd beta_aa_phi(beta.size());
                            const bool aa_valid =
                                phi_step_fullcancel_mixed_compressed_core(
                                    endog,
                                    design,
                                    beta_aa,
                                    cancel_groups,
                                    n_cancel_groups,
                                    tau,
                                    phase,
                                    param,
                                    beta_aa_phi,
                                    tau2_solver_ptr);
                            if (aa_valid) {
                                const Eigen::VectorXd plain_residual = beta_phi - beta;
                                const Eigen::VectorXd aa_residual =
                                    beta_aa_phi - beta_aa;
                                const double res_plain =
                                    pyiols::kernel::stable_l2_norm(plain_residual);
                                const double res_aa =
                                    pyiols::kernel::stable_l2_norm(aa_residual);
                                if (res_aa < res_plain * 1.1) {
                                    beta = beta_aa;
                                    beta_path.push_back(beta);
                                    if (k >= 2) {
                                        criterion = max_abs_change_core(beta, beta0);
                                    }
                                    accepted_aa_step = true;
                                }
                            }
                        }
                    }
                } else if (aa_m > 0) {
                    aa_hist.push(beta, beta_phi - beta);
                }

                if (!accepted_aa_step) {
                    beta = beta_phi;
                    beta_path.push_back(beta);
                }
                if (k >= 2) {
                    if (!accepted_aa_step) {
                        criterion = max_abs_change_core(beta, beta0);
                    }
                    if (record_criterion_progress_core(
                            criterion_history,
                            criterion,
                            phase,
                            tau,
                            eps_current,
                            control.max_phase2_evaluations,
                            last_finite_criterion,
                            best_finite_criterion,
                            failure_reason)) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                    }
                }

                if (k > 4 && beta_path.size() >= 4 &&
                    !(phase == 1 && std::abs(tau - 2.0) <= 1e-10)) {
                    const double qhat = qhat_from_beta_path_core(beta_path);
                    qhat_last = qhat;
                    if (std::isinf(std::abs(qhat)) && criterion > eps_current * 10.0) {
                        qhat_invalid_count += 1;
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "q_convergence_invalid";
                    }
                }

                if (phase == 1 && std::abs(tau - 2.0) > 1e-10 && k >= 30 &&
                    k % 10 == 0 && beta_path.size() >= 3) {
                    const double kappa_val = kappa_monitor_from_beta_path_core(beta_path);
                    kappa_monitor_last = kappa_val;
                    kappa_monitor_checks += 1;
                    if (std::isfinite(kappa_val)) {
                        kappa_monitor_max = std::isfinite(kappa_monitor_max)
                                                ? std::max(kappa_monitor_max, kappa_val)
                                                : kappa_val;
                    }
                    const int aa_m_new = adapt_anderson_m_core(aa_m, kappa_val);
                    if (aa_m_new > aa_m) {
                        aa_m = aa_m_new;
                        kappa_bad_count = 0;
                    } else if (std::isfinite(kappa_val) && kappa_val >= 1.0) {
                        kappa_bad_count += 1;
                    } else {
                        kappa_bad_count = 0;
                    }
                    if (kappa_bad_count >= 3) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "noncontractive";
                    }
                }

                k += 1;
                iteration_count = k;

                if (std::isnan(criterion)) {
                    restart_index += 1;
                    if (phase == 1 && !phase2_budget_exhausted &&
                        restart_index < static_cast<int>(rho_path.size())) {
                        param = rho_path[static_cast<std::size_t>(restart_index)];
                        criterion = 100.0;
                        beta = beta_outer_initial;
                        k = 1;
                        criterion_history.clear();
                        kappa_bad_count = 0;
                        beta_path.clear();
                        beta_path.push_back(beta);
                        aa_hist.clear();
                    } else if (phase == 0) {
                        break;
                    }
                }
            }

            bool converged = !std::isnan(criterion) && criterion <= eps_current;
            has_final_criterion = !std::isnan(criterion);
            final_criterion = criterion;
            if (converged) {
                Eigen::VectorXd beta_check(beta.size());
                const bool valid = phi_step_fullcancel_mixed_compressed_core(
                    endog,
                    design,
                    beta,
                    cancel_groups,
                    n_cancel_groups,
                    tau,
                    phase,
                    param,
                    beta_check,
                    tau2_solver_ptr);
                if (valid) {
                    post_convergence_residual =
                        max_abs_change_core(beta_check, beta);
                    if (post_convergence_residual > eps_current * 10.0) {
                        converged = false;
                        failure_reason = "post_convergence_residual";
                    }
                }
                if (converged && phase == 1) {
                    const double score_residual =
                        mixed_compressed_score_residual_core(
                            endog,
                            design,
                            beta,
                            cancel_groups,
                            n_cancel_groups,
                            tau);
                    score_tol = std::max(eps_current * 10.0, 1e-8);
                    post_convergence_score_residual = score_residual;
                    score_exceeds_tolerance =
                        !std::isfinite(score_residual) || score_residual > score_tol;
                    if (score_exceeds_tolerance) {
                        score_warning_codes.push_back("post_convergence_score_residual");
                        score_warning_messages.push_back(
                            "Fullcancel weighted-within score residual exceeds "
                            "tolerance at this tau-path state.");
                        score_validation_status = "diagnostic_warning";
                    } else if (score_validation_status == "not_checked") {
                        score_validation_status = "passed";
                    }
                }
                if (converged && beta_path.size() >= 4 &&
                    !(phase == 1 && std::abs(tau - 2.0) <= 1e-10)) {
                    const double d0 =
                        max_abs_change_core(beta_path[beta_path.size() - 1],
                                            beta_path[beta_path.size() - 2]);
                    const double d1 =
                        max_abs_change_core(beta_path[beta_path.size() - 2],
                                            beta_path[beta_path.size() - 3]);
                    const double d2 =
                        max_abs_change_core(beta_path[beta_path.size() - 3],
                                            beta_path[beta_path.size() - 4]);
                    if (d1 > 1e-15 && d2 > 1e-15) {
                        const double kappa_est = d0 / d1;
                        post_convergence_kappa_est = kappa_est;
                        if (kappa_est >= 1.0) {
                            converged = false;
                            failure_reason = "post_convergence_noncontractive";
                        }
                    }
                }
            }

            if (!converged && phase == 1 && !phase2_budget_exhausted &&
                restart_index < static_cast<int>(rho_path.size()) - 1) {
                restart_index = std::min(
                    restart_index + std::max(restart_index, 5),
                    static_cast<int>(rho_path.size()) - 1);
                param = rho_path[static_cast<std::size_t>(restart_index)];
                beta = beta_outer_initial;
            } else {
                verified = true;
            }
            aa_m_final = aa_m;

            if (phase == 0) {
                phase1_attempts += 1;
                const bool candidate_usable =
                    converged &&
                    phase2_initialization_usable_fullcancel_mixed_compressed_core(
                        endog,
                        design,
                        beta,
                        cancel_groups,
                        n_cancel_groups,
                        tau,
                        control,
                        tau2_solver_ptr);
                if (candidate_usable) {
                    phase1_accepted += 1;
                    phase1_accepted_beta = beta;
                } else {
                    phase1_rejected += 1;
                    beta = phase1_accepted_beta;
                    if (converged && failure_reason.empty()) {
                        failure_reason = "phase1_candidate_unusable";
                    }
                }
                final_diag.converged = candidate_usable;
            } else {
                if (!converged) {
                    beta = beta_outer_initial;
                }
                final_diag.converged = converged;
            }

            final_diag.phase = phase == 0 ? "P1" : "P2";
            final_diag.iterations = iteration_count;
            final_diag.restarts = restart_index;
            final_diag.has_final_rho = phase != 0;
            final_diag.final_rho = param;
            final_diag.rho_path.clear();
            if (phase != 0) {
                const int upto =
                    std::min(restart_index + 1, static_cast<int>(rho_path.size()));
                for (int idx = 0; idx < upto; ++idx) {
                    final_diag.rho_path.push_back(rho_path[static_cast<std::size_t>(idx)]);
                }
            }
            final_diag.has_final_criterion = has_final_criterion;
            final_diag.final_criterion = final_criterion;
            final_diag.anderson_m_initial = control.anderson_m;
            final_diag.anderson_m_final = aa_m_final;
            final_diag.failure_reason =
                final_diag.converged
                    ? ""
                    : (!failure_reason.empty() ? failure_reason : "not_converged");
            final_diag.phase1_attempts = phase1_attempts;
            final_diag.phase1_accepted = phase1_accepted;
            final_diag.phase1_rejected = phase1_rejected;
            final_diag.phase2_evaluations = phase2_evaluations;
            final_diag.max_phase2_evaluations = control.max_phase2_evaluations;
            final_diag.last_finite_criterion = last_finite_criterion;
            final_diag.best_finite_criterion = best_finite_criterion;
            if (phase == 1) {
                attach_fullcancel_rho_details_core(
                    final_diag, rho_bundle, rho_hint, control);
            }
            final_diag.algorithm = "weighted_within_fullcancel";
            final_diag.index_policy = "slope_only_raw_exog";
            final_diag.fe_policy = "weighted_within_cancel_all";
            final_diag.anderson_policy = "beta_anderson";
            final_diag.qhat_last = qhat_last;
            final_diag.qhat_invalid_count = qhat_invalid_count;
            final_diag.kappa_monitor_last = kappa_monitor_last;
            final_diag.kappa_monitor_max = kappa_monitor_max;
            final_diag.kappa_monitor_checks = kappa_monitor_checks;
            final_diag.post_convergence_residual = post_convergence_residual;
            final_diag.post_convergence_kappa_est = post_convergence_kappa_est;
            final_diag.post_convergence_score_residual =
                post_convergence_score_residual;
            final_diag.score_tolerance = score_tol;
            final_diag.score_validation_policy = "diagnostic";
            final_diag.score_validation_status = score_validation_status;
            final_diag.score_warning_codes = score_warning_codes;
            final_diag.score_warning_messages = score_warning_messages;
            final_diag.post_convergence_score_exceeds_tolerance =
                score_exceeds_tolerance;
        }
        if (phase == 1) {
            break;
        }
    }

    FullrecoverConcentratedCore out;
    out.beta = beta;
    if (all_finite_vec(beta)) {
        out.linear = mixed_compressed_linear_state_core(design, beta);
        out.has_linear = all_finite_vec(out.linear);
    }
    out.iteration = final_diag;
    return out;
}

std::vector<double> exp_linspace(double log_min, double log_max, int size) {
    std::vector<double> out;
    const int n = std::max(size, 0);
    if (n == 0) {
        return out;
    }
    if (n == 1) {
        out.push_back(std::exp(log_min));
        return out;
    }
    out.reserve(static_cast<std::size_t>(n));
    for (int idx = 0; idx < n; ++idx) {
        const double frac = static_cast<double>(idx) / static_cast<double>(n - 1);
        out.push_back(std::exp(log_min + frac * (log_max - log_min)));
    }
    return out;
}

std::vector<double> unique_positive_finite(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    std::vector<double> out;
    for (const double value : values) {
        if (!std::isfinite(value) || value <= 0.0) {
            continue;
        }
        if (out.empty() || value != out.back()) {
            out.push_back(value);
        }
    }
    return out;
}

std::vector<double> focus_rho_path_core(
    const std::vector<double>& path,
    std::optional<double> rho_hint) {
    if (path.empty() || !rho_hint.has_value() || !std::isfinite(*rho_hint) ||
        *rho_hint <= 0.0) {
        return path;
    }
    const double low = path.front();
    const double high = path.back();
    const double hint = std::min(std::max(*rho_hint, low), high);
    const double mults[] = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0};
    std::vector<double> ordered;
    auto push_unique = [&ordered](double value) {
        for (const double seen : ordered) {
            if (seen == value) {
                return;
            }
        }
        ordered.push_back(value);
    };
    for (const double mult : mults) {
        const double value = std::min(std::max(hint * mult, low), high);
        if (std::isfinite(value) && value > 0.0) {
            push_unique(value);
        }
    }
    for (const double value : path) {
        push_unique(value);
    }
    return ordered;
}

std::vector<double> rho_path_core(
    const FullrecoverControlCore& control,
    std::optional<double> rho_hint) {
    if (!(std::isfinite(control.rho_log_min) &&
          std::isfinite(control.rho_log_max)) ||
        control.rho_log_max <= control.rho_log_min) {
        throw std::invalid_argument("rho_log_max must be greater than rho_log_min");
    }
    std::vector<double> path = unique_positive_finite(exp_linspace(
        control.rho_log_min,
        control.rho_log_max,
        std::max(control.rho_grid_size, 2)));
    return focus_rho_path_core(path, rho_hint);
}

std::vector<double> delta_grid_core(const FullrecoverControlCore& control) {
    const int grid_size = std::max(control.delta_grid_size, 0);
    if (grid_size == 0) {
        return {};
    }
    if (!(std::isfinite(control.delta_log_min) &&
          std::isfinite(control.delta_log_max)) ||
        control.delta_log_max <= control.delta_log_min) {
        throw std::invalid_argument("delta_log_max must be greater than delta_log_min");
    }
    return exp_linspace(control.delta_log_min, control.delta_log_max, grid_size);
}

bool phase2_initialization_usable_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    const FullrecoverControlCore& control) {
    if (!all_finite_vec(beta)) {
        return false;
    }
    const std::vector<double> rho_path = rho_path_core(control, std::nullopt);
    for (const double rho : rho_path) {
        Eigen::VectorXd beta_test(beta.size());
        if (phi_step_profiled_core(
                endog,
                exog,
                beta,
                groups,
                n_groups,
                tau,
                projection,
                1,
                rho,
                beta_test)) {
            return true;
        }
    }
    return false;
}

double max_abs_change_core(
    const Eigen::Ref<const Eigen::VectorXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& right) {
    if (left.size() == 0 && right.size() == 0) {
        return 0.0;
    }
    return (left - right).cwiseAbs().maxCoeff();
}

bool criterion_stagnated_core(
    const std::vector<double>& history,
    double eps,
    int window = 50,
    int min_iterations = 100,
    double min_relative_drop = 0.01) {
    if (history.size() < static_cast<std::size_t>(std::max(window * 2, min_iterations))) {
        return false;
    }
    const double current = history.back();
    const double previous = history[history.size() - static_cast<std::size_t>(window)];
    if (!(std::isfinite(current) && std::isfinite(previous))) {
        return false;
    }
    if (current <= eps * 10.0) {
        return false;
    }
    const double relative_drop = (previous - current) / std::max(std::abs(previous), eps);
    return relative_drop < min_relative_drop;
}

bool record_criterion_progress_core(
    std::vector<double>& history,
    double criterion,
    int phase,
    double tau,
    double eps,
    int max_phase2_evaluations,
    double& last_finite_criterion,
    double& best_finite_criterion,
    std::string& failure_reason,
    const std::vector<double>* stagnation_history) {
    // runtime_v4: the direct route passes its plain-map residual history as
    // stagnation_history; every other caller keeps the criterion history.
    if (!std::isfinite(criterion)) {
        return false;
    }
    history.push_back(criterion);
    last_finite_criterion = criterion;
    best_finite_criterion =
        std::isfinite(best_finite_criterion)
            ? std::min(best_finite_criterion, criterion)
            : criterion;
    if (phase == 1 && tau > 2.0 && history.size() >= 20) {
        const double criterion_ref = history[history.size() - 20];
        if (std::isfinite(criterion_ref) && criterion_ref > eps &&
            criterion > std::max(criterion_ref * 100.0, 1e6)) {
            failure_reason = "criterion_explosion";
            return true;
        }
    }
    const int no_progress_min_iterations =
        std::min(std::max(max_phase2_evaluations / 2500, 20), 200);
    const int no_progress_window =
        std::min(std::max(no_progress_min_iterations / 2, 10), 50);
    if (phase == 1 &&
        criterion_stagnated_core(
            stagnation_history == nullptr ? history : *stagnation_history,
            eps,
            no_progress_window,
            no_progress_min_iterations,
            1e-4)) {
        failure_reason = "criterion_stagnation";
        return true;
    }
    return false;
}

int phase2_expensive_failed_step_threshold_core(
    const FullrecoverControlCore& control) {
    const int budget = std::max(control.max_phase2_evaluations, 1);
    return std::min(budget, std::max(budget / 20, 1000));
}

bool tau_path_expensive_failed_step_core(
    const FullrecoverIterationCore& iteration,
    const FullrecoverControlCore& control) {
    if (iteration.converged) {
        return false;
    }
    if (iteration.failure_reason == "phase2_evaluation_budget_exhausted") {
        return true;
    }
    if (iteration.failure_reason != "criterion_stagnation") {
        return false;
    }
    return iteration.phase2_evaluations >=
           phase2_expensive_failed_step_threshold_core(control);
}

bool tau_path_score_warning_refinement_step_core(
    const FullrecoverIterationCore& iteration) {
    return iteration.converged &&
           iteration.score_validation_status == "diagnostic_warning";
}

int adapt_anderson_m_core(int aa_m, double kappa_val) {
    if (!std::isfinite(kappa_val)) {
        return aa_m;
    }
    if (kappa_val >= 0.95) {
        return std::max(aa_m, 20);
    }
    if (kappa_val >= 0.85) {
        return std::max(aa_m, 12);
    }
    if (kappa_val >= 0.70) {
        return std::max(aa_m, 8);
    }
    return aa_m;
}

// runtime_v3 direct-route kappa statistic (twin of the NumPy
// _windowed_contraction_rate): geometric-mean per-iteration step contraction
// over the last `window` recorded finite criteria,
// (c_last / c_{last-window})^(1/window). NaN when not computable.
constexpr int kDirectKappaWindow = 10;

double windowed_contraction_rate_core(
    const std::vector<double>& history,
    int window) {
    if (window < 1 ||
        history.size() < static_cast<std::size_t>(window) + 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double current = history.back();
    const double previous =
        history[history.size() - 1 - static_cast<std::size_t>(window)];
    if (!(std::isfinite(current) && std::isfinite(previous))) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (previous <= 0.0 || current < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::pow(current / previous, 1.0 / static_cast<double>(window));
}

// runtime_v4 per-attempt progress-monitor fields (twin of the NumPy
// _direct_progress_trace_fields); the v3 criterion fields keep their meaning.
void set_direct_progress_trace_fields(
    Phase2AttemptTraceCore& record,
    const std::vector<double>& progress_history,
    int aa_m_start,
    int aa_m_end,
    int aa_accepted) {
    if (!progress_history.empty()) {
        record.final_progress_residual = progress_history.back();
        record.best_progress_residual =
            *std::min_element(progress_history.begin(), progress_history.end());
    }
    record.anderson_m_start = aa_m_start;
    record.anderson_m_end = aa_m_end;
    record.anderson_accepted = aa_accepted;
}

#if defined(PYIOLS_DIRECT_TEST_HOOKS)
// runtime_v4 TEST-ONLY fault injection, compiled only with
// -DPYIOLS_DIRECT_TEST_HOOKS=1 (never in the pinned campaign extension).
// While armed, the direct-route final verification map of the given phase
// (0 = Phase 1, 1 = Phase 2) is reported invalid. Single-threaded tests only.
int g_direct_test_fault_phase = -1;
int g_direct_test_fault_remaining = 0;  // -1 = every matching post-check
int g_direct_test_fault_fired = 0;

bool direct_test_hook_invalidate_post_check(int phase) {
    if (phase != g_direct_test_fault_phase || g_direct_test_fault_remaining == 0) {
        return false;
    }
    if (g_direct_test_fault_remaining > 0) {
        g_direct_test_fault_remaining -= 1;
    }
    g_direct_test_fault_fired += 1;
    return true;
}
#endif

constexpr const char* kDirectRestartStartPhase1 = "phase1_start";
constexpr const char* kDirectRestartStartBestScore = "best_score_iterate";
// runtime_v4: renamed because the windowed rate now reads the plain-map
// residual history instead of accepted moves.
constexpr const char* kDirectKappaMonitorPolicy = "windowed_plain_residual_ratio_10";
constexpr const char* kDirectProgressMonitorPolicy =
    "plain_map_residual_maxabs_star_at_current_iterate";
constexpr const char* kDirectPostCheckPolicy =
    "invalid_verification_map_clears_convergence";
constexpr const char* kDirectNofeRuntimeVersion = "direct_nofe_runtime_v4_2026_10_03";

FullrecoverIterationCore not_started_iteration_core(
    const FullrecoverControlCore& control) {
    FullrecoverIterationCore iteration;
    iteration.anderson_m_initial = control.anderson_m;
    iteration.anderson_m_final = control.anderson_m;
    iteration.failure_reason = "not_started";
    return iteration;
}

FullrecoverConcentratedCore concentrated_fullrecover_joint_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const GroupBundleCore& bundle,
    double tau,
    const FullrecoverControlCore& control,
    const FullrecoverInvariantCore& state,
    std::optional<Eigen::VectorXd> beta_init,
    std::optional<Eigen::VectorXd> linear_init,
    std::optional<double> rho_hint) {
    if (!beta_init.has_value() && std::abs(tau - 2.0) > 1e-10) {
        throw std::invalid_argument(
            "Non-anchor iOLS path steps require beta_init from tau=2");
    }
    const std::vector<double> delta_grid =
        beta_init.has_value() ? std::vector<double>{} : delta_grid_core(control);
    const Eigen::MatrixXd* manyway_q_basis =
        state.manyway_q_basis.cols() > 0 ? &state.manyway_q_basis : nullptr;
    Eigen::VectorXd beta;
    Eigen::VectorXd linear;
    int n_outer = 0;
    if (beta_init.has_value()) {
        beta = *beta_init;
        linear = linear_init.has_value()
                     ? *linear_init
                     : initial_linear_joint_fullrecover_core(
                           endog, exog, beta, bundle, control, manyway_q_basis);
        n_outer = 1;
    } else {
        Eigen::VectorXd log_y(endog.size());
        for (Eigen::Index row = 0; row < endog.size(); ++row) {
            log_y(row) = std::log(endog(row) + 1.0);
        }
        const Eigen::VectorXd y_init =
            manyway_q_basis != nullptr
                ? residualize_vector_with_q_basis_core(log_y, *manyway_q_basis)
                : demean_vector_for_group_bundle_core(
                      log_y,
                      bundle,
                      std::nullopt,
                      control.fe_demean_tol,
                      control.fe_demean_maxiter);
        beta = state.projection * y_init;
        linear = initial_linear_joint_fullrecover_core(
            endog, exog, beta, bundle, control, manyway_q_basis);
        n_outer = static_cast<int>(delta_grid.size()) + 1;
    }

    const Eigen::VectorXd beta_initial = beta;
    const Eigen::VectorXd linear_initial = linear;
    Eigen::VectorXd phase1_accepted_beta = beta;
    Eigen::VectorXd phase1_accepted_linear = linear;
    int phase1_attempts = 0;
    int phase1_accepted = 0;
    int phase1_rejected = 0;
    FullrecoverIterationCore final_diag = not_started_iteration_core(control);
    FullrecoverJointWorkspaceCore joint_phi_workspace;
    const bool joint_state_anderson_enabled =
        control.speed_fullrecover_joint_anderson && control.anderson_m > 0;
    const std::string joint_anderson_policy =
        joint_state_anderson_enabled ? "joint_state_anderson_speed"
                                     : "disabled_for_joint_hdfe";

    for (int outer_index = 0; outer_index < n_outer; ++outer_index) {
        FullrecoverIterationCore iteration = not_started_iteration_core(control);
        if (any_nan_vec(beta)) {
            beta = beta_initial;
            linear = linear_initial;
        }
        Eigen::VectorXd beta_outer_initial = beta;
        Eigen::VectorXd linear_outer_initial = linear;
        int phase = 0;
        double param = 0.0;
        double eps_current = control.eps;
        int stopcrit_current = control.stopcrit;
        std::vector<double> rho_path;
        if (outer_index < static_cast<int>(delta_grid.size())) {
            phase = 0;
            param = delta_grid[static_cast<std::size_t>(outer_index)];
            eps_current = std::max(control.eps * 10.0, 1e-5);
            stopcrit_current =
                std::min(control.stopcrit, control.max_phase1_iterations);
        } else {
            phase = 1;
            if (!beta_init.has_value()) {
                beta = phase1_accepted_beta;
                linear = phase1_accepted_linear;
            }
            rho_path = rho_path_core(control, rho_hint);
            if (rho_path.empty()) {
                FullrecoverConcentratedCore out;
                out.beta = beta;
                out.linear = linear;
                out.has_linear = all_finite_vec(linear);
                out.iteration = not_started_iteration_core(control);
                out.iteration.phase = "P2";
                out.iteration.failure_reason = "empty_finite_rho_path";
                out.iteration.phase1_attempts = phase1_attempts;
                out.iteration.phase1_accepted = phase1_accepted;
                out.iteration.phase1_rejected = phase1_rejected;
                out.iteration.fe_state_policy = "pure_hdfe_recover_all";
                out.iteration.anderson_policy = joint_anderson_policy;
                return out;
            }
            param = rho_path.front();
            beta_outer_initial = beta;
            linear_outer_initial = linear;
            eps_current = control.eps;
            stopcrit_current = control.stopcrit;
        }

        int restart_index = 0;
        bool verified = false;
        int phase2_evaluations = 0;
        bool phase2_budget_exhausted = false;
        int iteration_count = 0;
        bool has_final_criterion = false;
        double final_criterion = 0.0;
        int aa_m_final = 0;
        std::string failure_reason;
        double qhat_last = std::numeric_limits<double>::quiet_NaN();
        int qhat_invalid_count = 0;
        double kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
        double kappa_monitor_max = std::numeric_limits<double>::quiet_NaN();
        int kappa_monitor_checks = 0;
        double post_convergence_residual = std::numeric_limits<double>::quiet_NaN();
        double post_convergence_kappa_est = std::numeric_limits<double>::quiet_NaN();
        double post_convergence_score_residual =
            std::numeric_limits<double>::quiet_NaN();
        double post_convergence_fe_score_residual =
            std::numeric_limits<double>::quiet_NaN();
        double score_tol = std::numeric_limits<double>::quiet_NaN();
        std::string score_validation_status = "not_checked";
        std::vector<std::string> score_warning_codes;
        std::vector<std::string> score_warning_messages;
        bool score_exceeds_tolerance = false;
        bool fe_score_exceeds_tolerance = false;
        double last_finite_criterion = std::numeric_limits<double>::quiet_NaN();
        double best_finite_criterion = std::numeric_limits<double>::quiet_NaN();
        int phase2_rho_restart_events = 0;
        int phase2_rho_attempts = phase == 1 ? 1 : 0;

        while (!verified) {
            int k = 1;
            double criterion = 100.0;
            std::vector<double> criterion_history;
            int kappa_bad_count = 0;
            std::vector<Eigen::VectorXd> beta_path;
            EigenVectorPairRingCore aa_hist;
            beta_path.push_back(beta);
            int aa_m = joint_state_anderson_enabled ? control.anderson_m : 0;
            criterion_history.reserve(
                static_cast<std::size_t>(std::max(stopcrit_current, 0)));
            beta_path.reserve(
                static_cast<std::size_t>(std::max(stopcrit_current + 1, 1)));
            if (aa_m > 0) {
                aa_hist.reset(static_cast<std::size_t>(aa_m + 1));
            }

            while (criterion > eps_current && k < stopcrit_current &&
                   !std::isnan(criterion)) {
                const Eigen::VectorXd beta0 = beta;
                const Eigen::VectorXd linear0 = linear;
                if (phase == 1) {
                    phase2_evaluations += 1;
                    if (phase2_evaluations > control.max_phase2_evaluations) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "phase2_evaluation_budget_exhausted";
                        phase2_budget_exhausted = true;
                        break;
                    }
                }
                Eigen::VectorXd beta_phi(beta.size());
                Eigen::VectorXd linear_phi(linear.size());
                const double active_fe_demean_tol =
                    control.research_tiered_fe_tol && phase == 0
                        ? control.research_inner_fe_demean_tol
                        : control.fe_demean_tol;
                const bool valid = phi_step_joint_fullrecover_core(
                    endog,
                    beta,
                    linear,
                    bundle,
                    tau,
                    state.projection,
                    state.x_demeaned,
                    manyway_q_basis,
                    phase,
                    param,
                    active_fe_demean_tol,
                    control,
                    beta_phi,
                    linear_phi,
                    &joint_phi_workspace,
                    &iteration);
                if (!valid) {
                    criterion = std::numeric_limits<double>::quiet_NaN();
                    failure_reason = "invalid_phi_step";
                    break;
                }

                bool accepted_aa_step = false;
                if (aa_m > 0) {
                    Eigen::VectorXd state_current(beta.size() + linear.size());
                    state_current.head(beta.size()) = beta;
                    state_current.tail(linear.size()) = linear;
                    Eigen::VectorXd residual_current(state_current.size());
                    residual_current.head(beta.size()) = beta_phi - beta;
                    residual_current.tail(linear.size()) = linear_phi - linear;
                    aa_hist.push(state_current, residual_current);
                    const int m_cur = static_cast<int>(aa_hist.size()) - 1;
                    if (k > aa_m + 1 && m_cur >= 1) {
                        Eigen::MatrixXd f_mat(state_current.size(), m_cur);
                        for (int j = 1; j <= m_cur; ++j) {
                            f_mat.col(j - 1) =
                                aa_hist.f_back() -
                                aa_hist.f_at(
                                    aa_hist.size() -
                                    static_cast<std::size_t>(j + 1));
                        }
                        Eigen::VectorXd gamma =
                            pyiols::kernel::min_norm_lstsq_anderson(
                                f_mat,
                                aa_hist.f_back())
                                .col(0);
                        Eigen::VectorXd state_phi = state_current + residual_current;
                        Eigen::VectorXd state_aa = state_phi;
                        for (int j = 0; j < m_cur; ++j) {
                            const std::size_t idx =
                                aa_hist.size() -
                                static_cast<std::size_t>(j + 2);
                            const Eigen::VectorXd phi_prev =
                                aa_hist.x_at(idx) + aa_hist.f_at(idx);
                            state_aa += gamma(j) * (phi_prev - state_phi);
                        }
                        const Eigen::VectorXd beta_aa =
                            state_aa.head(beta.size());
                        const Eigen::VectorXd linear_aa =
                            state_aa.tail(linear.size());
                        if (all_finite_vec(beta_aa) && all_finite_vec(linear_aa)) {
                            if (phase == 1) {
                                phase2_evaluations += 1;
                                if (phase2_evaluations >
                                    control.max_phase2_evaluations) {
                                    criterion =
                                        std::numeric_limits<double>::quiet_NaN();
                                    failure_reason =
                                        "phase2_evaluation_budget_exhausted";
                                    phase2_budget_exhausted = true;
                                    break;
                                }
                            }
                            Eigen::VectorXd beta_aa_phi(beta.size());
                            Eigen::VectorXd linear_aa_phi(linear.size());
                            const bool aa_valid = phi_step_joint_fullrecover_core(
                                endog,
                                beta_aa,
                                linear_aa,
                                bundle,
                                tau,
                                state.projection,
                                state.x_demeaned,
                                manyway_q_basis,
                                phase,
                                param,
                                active_fe_demean_tol,
                                control,
                                beta_aa_phi,
                                linear_aa_phi,
                                &joint_phi_workspace,
                                &iteration);
                            if (aa_valid) {
                                Eigen::VectorXd aa_residual(state_current.size());
                                aa_residual.head(beta.size()) =
                                    beta_aa_phi - beta_aa;
                                aa_residual.tail(linear.size()) =
                                    linear_aa_phi - linear_aa;
                                const double res_plain = std::max(
                                    max_abs_change_core(beta_phi, beta),
                                    max_abs_change_core(linear_phi, linear));
                                const double res_aa = std::max(
                                    max_abs_change_core(beta_aa_phi, beta_aa),
                                    max_abs_change_core(linear_aa_phi, linear_aa));
                                if (std::isfinite(res_plain) &&
                                    std::isfinite(res_aa) &&
                                    res_aa < res_plain * 1.1) {
                                    beta = beta_aa;
                                    linear = linear_aa;
                                    beta_path.push_back(beta);
                                    if (k >= 2) {
                                        criterion = std::max(
                                            max_abs_change_core(beta, beta0),
                                            max_abs_change_core(linear, linear0));
                                    }
                                    accepted_aa_step = true;
                                }
                            }
                        }
                    }
                }

                if (!accepted_aa_step) {
                    double beta_criterion =
                        std::numeric_limits<double>::quiet_NaN();
                    double linear_criterion =
                        std::numeric_limits<double>::quiet_NaN();
                    if (k >= 2) {
                        beta_criterion = max_abs_change_core(beta_phi, beta);
                        linear_criterion =
                            max_abs_change_core(linear_phi, linear);
                    }
                    beta = beta_phi;
                    linear = linear_phi;
                    beta_path.push_back(beta);
                    if (k >= 2) {
                        criterion = std::max(beta_criterion, linear_criterion);
                    }
                }
                if (k >= 2) {
                    if (record_criterion_progress_core(
                            criterion_history,
                            criterion,
                            phase,
                            tau,
                            eps_current,
                            control.max_phase2_evaluations,
                            last_finite_criterion,
                            best_finite_criterion,
                            failure_reason)) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                    }
                }

                if (k > 4 && beta_path.size() >= 4 &&
                    !(phase == 1 && std::abs(tau - 2.0) <= 1e-10)) {
                    const double qhat = qhat_from_beta_path_core(beta_path);
                    qhat_last = qhat;
                    if (std::isinf(std::abs(qhat)) && criterion > eps_current * 10.0) {
                        qhat_invalid_count += 1;
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "q_convergence_invalid";
                    }
                }

                if (phase == 1 && std::abs(tau - 2.0) > 1e-10 && k >= 30 &&
                    k % 10 == 0 && beta_path.size() >= 3) {
                    const double kappa_val = kappa_monitor_from_beta_path_core(beta_path);
                    kappa_monitor_last = kappa_val;
                    kappa_monitor_checks += 1;
                    if (std::isfinite(kappa_val)) {
                        kappa_monitor_max = std::isfinite(kappa_monitor_max)
                                                ? std::max(kappa_monitor_max, kappa_val)
                                                : kappa_val;
                    }
                    const int aa_m_new = adapt_anderson_m_core(aa_m, kappa_val);
                    if (aa_m_new > aa_m) {
                        aa_m = aa_m_new;
                        kappa_bad_count = 0;
                    } else if (std::isfinite(kappa_val) && kappa_val >= 1.0) {
                        kappa_bad_count += 1;
                    } else {
                        kappa_bad_count = 0;
                    }
                    if (kappa_bad_count >= 3) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "noncontractive";
                    }
                }

                k += 1;
                iteration_count = k;

                if (std::isnan(criterion)) {
                    restart_index += 1;
                    if (phase == 1 && !phase2_budget_exhausted &&
                        restart_index < static_cast<int>(rho_path.size())) {
                        param = rho_path[static_cast<std::size_t>(restart_index)];
                        phase2_rho_restart_events += 1;
                        phase2_rho_attempts += 1;
                        criterion = 100.0;
                        beta = beta_outer_initial;
                        linear = linear_outer_initial;
                        k = 1;
                        criterion_history.clear();
                        kappa_bad_count = 0;
                        beta_path.clear();
                        beta_path.push_back(beta);
                        aa_hist.clear();
                    } else if (phase == 0) {
                        break;
                    }
                }
            }

            bool converged = !std::isnan(criterion) && criterion <= eps_current;
            has_final_criterion = !std::isnan(criterion);
            final_criterion = criterion;
            if (converged) {
                Eigen::VectorXd beta_check(beta.size());
                Eigen::VectorXd linear_check(linear.size());
                const bool valid = phi_step_joint_fullrecover_core(
                    endog,
                    beta,
                    linear,
                    bundle,
                    tau,
                    state.projection,
                    state.x_demeaned,
                    manyway_q_basis,
                    phase,
                    param,
                    control.fe_demean_tol,
                    control,
                    beta_check,
                    linear_check,
                    &joint_phi_workspace,
                    &iteration);
                if (valid) {
                    post_convergence_residual = std::max(
                        max_abs_change_core(beta_check, beta),
                        max_abs_change_core(linear_check, linear));
                    if (post_convergence_residual > eps_current * 10.0) {
                        converged = false;
                        failure_reason = "post_convergence_residual";
                    }
                }
                if (converged && phase == 1) {
                    const double score_residual = slope_score_residual_with_linear_core(
                        endog, linear, state.x_demeaned, tau);
                    score_tol = std::max(eps_current * 10.0, 1e-8);
                    post_convergence_score_residual = score_residual;
                    score_exceeds_tolerance =
                        !std::isfinite(score_residual) || score_residual > score_tol;
                    if (score_exceeds_tolerance) {
                        score_warning_codes.push_back("post_convergence_score_residual");
                        score_warning_messages.push_back(
                            "Slope score residual exceeds tolerance at this "
                            "tau-path state.");
                        score_validation_status = "diagnostic_warning";
                    } else if (score_validation_status == "not_checked") {
                        score_validation_status = "passed";
                    }
                }
                if (converged && phase == 1) {
                    const double fe_score_residual =
                        fe_score_residual_bundle_core(endog, linear, bundle, tau);
                    score_tol = std::max(eps_current * 10.0, 1e-8);
                    post_convergence_fe_score_residual = fe_score_residual;
                    fe_score_exceeds_tolerance =
                        !std::isfinite(fe_score_residual) ||
                        fe_score_residual > score_tol;
                    if (fe_score_exceeds_tolerance) {
                        score_warning_codes.push_back(
                            "post_convergence_fe_score_residual");
                        score_warning_messages.push_back(
                            "Recovered-FE score residual exceeds tolerance at "
                            "this tau-path state.");
                        score_validation_status = "diagnostic_warning";
                    } else if (score_validation_status == "not_checked") {
                        score_validation_status = "passed";
                    }
                }
                if (converged && beta_path.size() >= 4 &&
                    !(phase == 1 && std::abs(tau - 2.0) <= 1e-10)) {
                    const double d0 =
                        max_abs_change_core(beta_path[beta_path.size() - 1],
                                            beta_path[beta_path.size() - 2]);
                    const double d1 =
                        max_abs_change_core(beta_path[beta_path.size() - 2],
                                            beta_path[beta_path.size() - 3]);
                    const double d2 =
                        max_abs_change_core(beta_path[beta_path.size() - 3],
                                            beta_path[beta_path.size() - 4]);
                    if (d1 > 1e-15 && d2 > 1e-15) {
                        const double kappa_est = d0 / d1;
                        post_convergence_kappa_est = kappa_est;
                        if (kappa_est >= 1.0) {
                            converged = false;
                            failure_reason = "post_convergence_noncontractive";
                        }
                    }
                }
            }

            if (!converged && phase == 1 && !phase2_budget_exhausted &&
                restart_index < static_cast<int>(rho_path.size()) - 1) {
                const int previous_restart_index = restart_index;
                restart_index = std::min(
                    restart_index + std::max(restart_index, 5),
                    static_cast<int>(rho_path.size()) - 1);
                if (restart_index != previous_restart_index) {
                    param = rho_path[static_cast<std::size_t>(restart_index)];
                    phase2_rho_restart_events += 1;
                    phase2_rho_attempts += 1;
                    beta = beta_outer_initial;
                    linear = linear_outer_initial;
                    aa_hist.clear();
                } else {
                    verified = true;
                }
            } else {
                verified = true;
            }
            aa_m_final = aa_m;

            if (phase == 0) {
                phase1_attempts += 1;
                const bool candidate_usable =
                    converged &&
                    phase2_initialization_usable_joint_fullrecover_core(
                        endog,
                        beta,
                        linear,
                        bundle,
                        tau,
                        state.projection,
                        state.x_demeaned,
                        manyway_q_basis,
                        control,
                        &iteration);
                if (candidate_usable) {
                    phase1_accepted += 1;
                    phase1_accepted_beta = beta;
                    phase1_accepted_linear = linear;
                } else {
                    phase1_rejected += 1;
                    beta = phase1_accepted_beta;
                    linear = phase1_accepted_linear;
                    if (converged && failure_reason.empty()) {
                        failure_reason = "phase1_candidate_unusable";
                    }
                }
                final_diag.converged = candidate_usable;
            } else {
                if (!converged) {
                    beta = beta_outer_initial;
                    linear = linear_outer_initial;
                }
                final_diag.converged = converged;
            }

            final_diag.phase = phase == 0 ? "P1" : "P2";
            final_diag.iterations = iteration_count;
            final_diag.restarts = restart_index;
            final_diag.has_final_rho = phase != 0;
            final_diag.final_rho = param;
            final_diag.rho_path.clear();
            if (phase != 0) {
                const int upto =
                    std::min(restart_index + 1, static_cast<int>(rho_path.size()));
                for (int idx = 0; idx < upto; ++idx) {
                    final_diag.rho_path.push_back(rho_path[static_cast<std::size_t>(idx)]);
                }
            }
            final_diag.has_final_criterion = has_final_criterion;
            final_diag.final_criterion = final_criterion;
            final_diag.anderson_m_initial = control.anderson_m;
            final_diag.anderson_m_final = aa_m_final;
            final_diag.failure_reason =
                final_diag.converged
                    ? ""
                    : (!failure_reason.empty() ? failure_reason : "not_converged");
            final_diag.phase1_attempts = phase1_attempts;
            final_diag.phase1_accepted = phase1_accepted;
            final_diag.phase1_rejected = phase1_rejected;
            final_diag.phase2_evaluations = phase2_evaluations;
            final_diag.max_phase2_evaluations = control.max_phase2_evaluations;
            final_diag.last_finite_criterion = last_finite_criterion;
            final_diag.best_finite_criterion = best_finite_criterion;
            final_diag.fe_state_policy = "pure_hdfe_recover_all";
            final_diag.anderson_policy = joint_anderson_policy;
            final_diag.qhat_last = qhat_last;
            final_diag.qhat_invalid_count = qhat_invalid_count;
            final_diag.kappa_monitor_last = kappa_monitor_last;
            final_diag.kappa_monitor_max = kappa_monitor_max;
            final_diag.kappa_monitor_checks = kappa_monitor_checks;
            final_diag.post_convergence_residual = post_convergence_residual;
            final_diag.post_convergence_kappa_est = post_convergence_kappa_est;
            final_diag.post_convergence_score_residual =
                post_convergence_score_residual;
            final_diag.post_convergence_fe_score_residual =
                post_convergence_fe_score_residual;
            final_diag.score_tolerance = score_tol;
            final_diag.score_validation_policy = "diagnostic";
            final_diag.score_validation_status = score_validation_status;
            final_diag.score_warning_codes = score_warning_codes;
            final_diag.score_warning_messages = score_warning_messages;
            final_diag.post_convergence_score_exceeds_tolerance =
                score_exceeds_tolerance;
            final_diag.post_convergence_fe_score_exceeds_tolerance =
                fe_score_exceeds_tolerance;
            final_diag.joint_hdfe_phi_calls = iteration.joint_hdfe_phi_calls;
            final_diag.joint_hdfe_phi_demean_sweeps =
                iteration.joint_hdfe_phi_demean_sweeps;
            final_diag.research_tiered_fe_tol =
                control.research_tiered_fe_tol;
            final_diag.research_inner_fe_demean_tol =
                control.research_inner_fe_demean_tol;
            final_diag.phase2_rho_restart_events = phase2_rho_restart_events;
            final_diag.phase2_rho_attempts = phase2_rho_attempts;
        }
        if (phase == 1) {
            break;
        }
    }

    FullrecoverConcentratedCore out;
    out.beta = beta;
    out.linear = linear;
    out.has_linear = all_finite_vec(linear);
    out.iteration = final_diag;
    return out;
}

constexpr std::size_t kProfiledBetaPathRetained = 4;

// Appends to the bounded profiled beta path and folds the newest ratio row into
// the running kappa maximum. The kappa monitor is the max over every ratio row
// of the full path, so a running max over rows as they are appended gives the
// same value without retaining or rescanning the full history. The
// post-convergence contraction check only reads the last four entries.
// min_entries selects the first entry that contributes a ratio row: 3 keeps the
// legacy C++ profiled-core window (the first row involves the initial point);
// 4 matches the NumPy reference (beta_path[1:], initial point excluded) and
// is used by the runtime_v2 direct fixed-tau route.
constexpr std::size_t kProfiledKappaMinEntriesLegacy = 3;
constexpr std::size_t kProfiledKappaMinEntriesNumpyParity = 4;

void push_profiled_beta_path_core(
    std::vector<Eigen::VectorXd>& beta_path,
    const Eigen::VectorXd& beta,
    std::size_t& entries_seen,
    double& kappa_row_max,
    std::size_t min_entries = kProfiledKappaMinEntriesLegacy) {
    beta_path.push_back(beta);
    entries_seen += 1;
    if (entries_seen >= min_entries) {
        const std::size_t row = beta_path.size() - 1;
        const Eigen::VectorXd d_now =
            (beta_path[row] - beta_path[row - 1]).cwiseAbs();
        const Eigen::VectorXd d_prev =
            (beta_path[row - 1] - beta_path[row - 2]).cwiseAbs();
        double mean_ratio = 0.0;
        int count = 0;
        for (Eigen::Index idx = 0; idx < d_now.size(); ++idx) {
            const double ratio = d_now(idx) / d_prev(idx);
            if (!std::isnan(ratio)) {
                mean_ratio += ratio;
                count += 1;
            }
        }
        if (count > 0) {
            kappa_row_max = std::max(kappa_row_max, mean_ratio / count);
        }
    }
    if (beta_path.size() > kProfiledBetaPathRetained) {
        beta_path.erase(beta_path.begin());
    }
}

// rho_grid_hook_v1 (research diagnostic, twin of the NumPy hook): opt-in
// Phase-2 rho list for the direct no-FE route. A null hook or the "legacy"
// policy keeps rho_path_core untouched; no other control flow changes.
constexpr const char* kDirectRhoGridHookVersion = "rho_grid_hook_v1_2026_10_05";
constexpr const char* kDirectRhoGridLegacy = "legacy";
constexpr const char* kDirectRhoGridStep = "step_grid";
constexpr const char* kDirectRhoGridCurvatureStep = "curvature_step_grid";

struct DirectRhoGridHookCore {
    std::string policy = kDirectRhoGridLegacy;
    // Outputs, filled once at the Phase-2 handoff.
    bool reached = false;
    int grid_count = 0;
    std::vector<double> values;
    double alpha0 = std::numeric_limits<double>::quiet_NaN();
    double alpha_min = std::numeric_limits<double>::quiet_NaN();
    bool clipped = false;
    bool has_curvature = false;
    double curvature_L0 = std::numeric_limits<double>::quiet_NaN();
    double curvature_lambda_min = std::numeric_limits<double>::quiet_NaN();
    std::int64_t curvature_wall_ns = 0;
};

void validate_direct_rho_grid_policy_core(const std::string& policy) {
    if (policy != kDirectRhoGridLegacy && policy != kDirectRhoGridStep &&
        policy != kDirectRhoGridCurvatureStep) {
        throw std::invalid_argument(
            "rho_grid_policy must be one of ('legacy', 'step_grid', "
            "'curvature_step_grid')");
    }
}

// Halving alpha grid in plain double arithmetic (libm exp only in callers).
std::vector<double> direct_step_rho_grid_core(
    double alpha0,
    double alpha_min,
    bool& clipped) {
    if (!(std::isfinite(alpha0) && alpha0 > 0.0)) {
        throw std::runtime_error("direct_rho_grid_invalid: alpha0");
    }
    if (!(std::isfinite(alpha_min) && alpha_min > 0.0)) {
        throw std::runtime_error("direct_rho_grid_invalid: alpha_min");
    }
    std::vector<double> alphas;
    clipped = alpha0 <= alpha_min;
    if (!clipped) {
        double a = alpha0;
        while (a > alpha_min) {
            alphas.push_back(a);
            a = a * 0.5;
        }
    }
    alphas.push_back(alpha_min);
    std::vector<double> rho;
    rho.reserve(alphas.size());
    for (const double a : alphas) {
        const double value = 1.0 / a - 1.0;
        if (!(std::isfinite(value) && value > 0.0)) {
            throw std::runtime_error("direct_rho_grid_invalid: rho");
        }
        if (!rho.empty() && !(value > rho.back())) {
            throw std::runtime_error(
                "direct_rho_grid_invalid: not strictly increasing");
        }
        rho.push_back(value);
    }
    return rho;
}

// Largest generalized eigenvalue of the observed intercept-profiled negative
// Hessian H against Q = Z'Z/n at beta (Z = centered slope design). Diagnostic
// only: the projection and the update direction are not touched.
void direct_profiled_curvature_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const Eigen::MatrixXd& x_demeaned,
    DirectRhoGridHookCore& hook) {
    const auto start = std::chrono::steady_clock::now();
    const Eigen::VectorXd linear =
        initial_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    if (!all_finite_vec(linear)) {
        throw std::runtime_error("direct_rho_curvature_failed: nonfinite_index");
    }
    const Eigen::Index n = x_demeaned.rows();
    const double c1 = tau - 1.0;
    const double c2 = 2.0 - tau;
    Eigen::VectorXd h = Eigen::VectorXd::Zero(n);
    for (Eigen::Index row = 0; row < n; ++row) {
        double value = 0.0;
        if (c1 != 0.0 && endog(row) > 0.0) {
            value += c1 * exp_guard_scalar(
                              std::log(endog(row)) + (1.0 - tau) * linear(row));
        }
        if (c2 != 0.0) {
            value += c2 * exp_guard_scalar((2.0 - tau) * linear(row));
        }
        h(row) = value;
    }
    if (!all_finite_vec(h)) {
        throw std::runtime_error(
            "direct_rho_curvature_failed: nonfinite_row_curvature");
    }
    const double total = h.sum();
    if (!(std::isfinite(total) && total > 0.0)) {
        throw std::runtime_error(
            "direct_rho_curvature_failed: nonpositive_curvature_sum");
    }
    const double n_double = static_cast<double>(n);
    const Eigen::VectorXd g = x_demeaned.transpose() * h;
    const Eigen::MatrixXd weighted =
        (x_demeaned.array().colwise() * h.array()).matrix();
    const Eigen::MatrixXd hess_raw =
        (x_demeaned.transpose() * weighted - (g * g.transpose()) / total) / n_double;
    const Eigen::MatrixXd hess = 0.5 * (hess_raw + hess_raw.transpose());
    if (!hess.allFinite()) {
        throw std::runtime_error("direct_rho_curvature_failed: nonfinite_hessian");
    }
    const Eigen::MatrixXd gram_raw =
        (x_demeaned.transpose() * x_demeaned) / n_double;
    const Eigen::MatrixXd gram = 0.5 * (gram_raw + gram_raw.transpose());
    const Eigen::LLT<Eigen::MatrixXd> llt(gram);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "direct_rho_curvature_failed: gram_cholesky_failed");
    }
    const Eigen::MatrixXd lower = llt.matrixL();
    const Eigen::MatrixXd left =
        lower.triangularView<Eigen::Lower>().solve(hess);
    const Eigen::MatrixXd cmat_raw =
        lower.triangularView<Eigen::Lower>().solve(left.transpose());
    const Eigen::MatrixXd cmat = 0.5 * (cmat_raw + cmat_raw.transpose());
    if (!cmat.allFinite()) {
        throw std::runtime_error("direct_rho_curvature_failed: nonfinite_hessian");
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        cmat, Eigen::EigenvaluesOnly);
    if (spectrum.info() != Eigen::Success) {
        throw std::runtime_error("direct_rho_curvature_failed: eigensolver_failed");
    }
    if (!spectrum.eigenvalues().allFinite()) {
        throw std::runtime_error(
            "direct_rho_curvature_failed: nonfinite_eigenvalue");
    }
    hook.has_curvature = true;
    hook.curvature_L0 = spectrum.eigenvalues().maxCoeff();
    hook.curvature_lambda_min = spectrum.eigenvalues().minCoeff();
    hook.curvature_wall_ns = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

// Non-legacy rho list at the Phase-2 handoff (curvature computed once here).
std::vector<double> direct_policy_rho_path_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const FullrecoverControlCore& control,
    const FullrecoverInvariantCore& state,
    DirectRhoGridHookCore& hook) {
    const double alpha_max = 1.0 / (1.0 + std::exp(control.rho_log_min));
    const double alpha_min = 1.0 / (1.0 + std::exp(control.rho_log_max));
    double alpha0 = alpha_max;
    if (hook.policy == kDirectRhoGridCurvatureStep) {
        direct_profiled_curvature_core(
            endog, exog, beta, groups, n_groups, tau, state.x_demeaned, hook);
        alpha0 = 1.0 / std::max(1.0 + std::exp(control.rho_log_min), hook.curvature_L0);
    } else if (hook.policy != kDirectRhoGridStep) {
        validate_direct_rho_grid_policy_core(hook.policy);
        throw std::invalid_argument("rho_grid_policy legacy has no policy path");
    }
    bool clipped = false;
    std::vector<double> rho = direct_step_rho_grid_core(alpha0, alpha_min, clipped);
    hook.values = rho;
    hook.alpha0 = alpha0;
    hook.alpha_min = alpha_min;
    hook.clipped = clipped;
    return rho;
}

FullrecoverConcentratedCore concentrated_fullrecover_profiled_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau,
    const FullrecoverControlCore& control,
    const FullrecoverInvariantCore& state,
    std::optional<Eigen::VectorXd> beta_init,
    std::optional<Eigen::VectorXd> linear_init,
    std::optional<double> rho_hint,
    bool direct_nofe_fixed_tau = false,
    DirectRhoGridHookCore* rho_hook = nullptr) {
    // rho_grid_hook_v1: opt-in policy is direct-route only.
    if (rho_hook != nullptr) {
        validate_direct_rho_grid_policy_core(rho_hook->policy);
        if (rho_hook->policy != kDirectRhoGridLegacy && !direct_nofe_fixed_tau) {
            throw std::invalid_argument(
                "rho_grid_policy other than 'legacy' requires direct_nofe_fixed_tau.");
        }
    }
    if (direct_nofe_fixed_tau &&
        (beta_init.has_value() || !is_singleton_intercept_group(groups, n_groups))) {
        throw std::invalid_argument(
            "Direct fixed-tau Phase 1 + Phase 2 is authorized only for the "
            "noFE singleton-intercept route without beta_init");
    }
    if (!beta_init.has_value() && std::abs(tau - 2.0) > 1e-10 &&
        !direct_nofe_fixed_tau) {
        throw std::invalid_argument(
            "Non-anchor iOLS path steps require beta_init from tau=2");
    }
    const std::vector<double> delta_grid =
        beta_init.has_value() ? std::vector<double>{} : delta_grid_core(control);
    Eigen::VectorXd beta;
    Eigen::VectorXd linear;
    int n_outer = 0;
    if (beta_init.has_value()) {
        beta = *beta_init;
        linear = linear_init.has_value()
                     ? *linear_init
                     : initial_linear_profiled_core(
                           endog, exog, beta, groups, n_groups, tau);
        n_outer = 1;
    } else {
        Eigen::VectorXd log_y(endog.size());
        for (Eigen::Index row = 0; row < endog.size(); ++row) {
            log_y(row) = std::log(endog(row) + 1.0);
        }
        const Eigen::MatrixXd log_y_mat = log_y;
        const Eigen::VectorXd y_init =
            demean_profiled_core(log_y_mat, groups, n_groups).col(0);
        beta = state.projection * y_init;
        linear = initial_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
        n_outer = static_cast<int>(delta_grid.size()) + 1;
    }

    const Eigen::VectorXd beta_initial = beta;
    const Eigen::VectorXd linear_initial = linear;
    Eigen::VectorXd phase1_accepted_beta = beta;
    Eigen::VectorXd phase1_accepted_linear = linear;
    int phase1_attempts = 0;
    int phase1_accepted = 0;
    int phase1_rejected = 0;
    // runtime_v2: total Phase 1 map evaluations (plain steps plus Anderson
    // trial maps over every delta attempt); twin of the NumPy counter.
    int phase1_evaluations = 0;
    // runtime_v2 direct route: NumPy-parity kappa window and Anderson history
    // that grows with the adapted memory. Legacy callers keep their semantics.
    const std::size_t kappa_min_entries = direct_nofe_fixed_tau
                                              ? kProfiledKappaMinEntriesNumpyParity
                                              : kProfiledKappaMinEntriesLegacy;
    FullrecoverIterationCore final_diag = not_started_iteration_core(control);
    // runtime_v3 direct-route state (declared at function scope for C++
    // scoping; read and written only inside direct_nofe_fixed_tau blocks).
    bool v3_restart_from_best = false;
    double v3_best_score_norm = std::numeric_limits<double>::infinity();
    double v3_best_score_rho = std::numeric_limits<double>::quiet_NaN();
    bool v3_has_best_score_beta = false;
    Eigen::VectorXd v3_best_score_beta;
    int v3_phase2_warm_restarts = 0;
    std::vector<Phase2AttemptTraceCore> v3_phase2_attempt_trace;
    double v3_kappa_coordinate_row_max = std::numeric_limits<double>::quiet_NaN();
    double v3_kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
    double v3_kappa_monitor_max = std::numeric_limits<double>::quiet_NaN();
    int v3_kappa_monitor_checks = 0;
    int v3_attempt_evaluations_start = 0;
    std::string v3_attempt_start_policy = kDirectRestartStartPhase1;
    // runtime_v4: final verification maps that came back invalid, by
    // internal phase (0 = Phase 1 delta grid, 1 = Phase 2 rho grid).
    int v4_post_convergence_invalid_map_counts[2] = {0, 0};
    if (direct_nofe_fixed_tau) {
        v3_restart_from_best =
            control.direct_restart_start == kDirectRestartStartBestScore;
    }

    for (int outer_index = 0; outer_index < n_outer; ++outer_index) {
        if (!all_finite_vec(beta)) {
            beta = beta_initial;
            linear = linear_initial;
        }
        Eigen::VectorXd beta_outer_initial = beta;
        Eigen::VectorXd linear_outer_initial = linear;
        int phase = 0;
        double param = 0.0;
        double eps_current = control.eps;
        int stopcrit_current = control.stopcrit;
        std::vector<double> rho_path;
        if (outer_index < static_cast<int>(delta_grid.size())) {
            phase = 0;
            param = delta_grid[static_cast<std::size_t>(outer_index)];
            eps_current = std::max(control.eps * 10.0, 1e-5);
            stopcrit_current = std::min(control.stopcrit, control.max_phase1_iterations);
        } else {
            phase = 1;
            if (!beta_init.has_value()) {
                beta = phase1_accepted_beta;
                linear = phase1_accepted_linear;
            }
            if (direct_nofe_fixed_tau && rho_hook != nullptr &&
                rho_hook->policy != kDirectRhoGridLegacy) {
                // rho_grid_hook_v1: step grid; curvature L0 at the Phase-2 pilot.
                rho_path = direct_policy_rho_path_core(
                    endog, exog, beta, groups, n_groups, tau, control, state, *rho_hook);
            } else {
                rho_path = rho_path_core(control, rho_hint);
            }
            if (direct_nofe_fixed_tau && rho_hook != nullptr) {
                rho_hook->reached = true;
                rho_hook->grid_count = static_cast<int>(rho_path.size());
            }
            if (rho_path.empty()) {
                FullrecoverConcentratedCore out;
                out.beta = beta;
                out.linear = linear;
                out.has_linear = all_finite_vec(linear);
                out.iteration = not_started_iteration_core(control);
                out.iteration.phase = "P2";
                out.iteration.failure_reason = "empty_finite_rho_path";
                out.iteration.phase1_attempts = phase1_attempts;
                out.iteration.phase1_accepted = phase1_accepted;
                out.iteration.phase1_rejected = phase1_rejected;
                out.iteration.phase1_evaluations = phase1_evaluations;
                return out;
            }
            param = rho_path.front();
            beta_outer_initial = beta;
            linear_outer_initial = linear;
        }

        int restart_index = 0;
        bool verified = false;
        int phase2_evaluations = 0;
        bool phase2_budget_exhausted = false;
        int iteration_count = 0;
        bool has_final_criterion = false;
        double final_criterion = 0.0;
        int aa_m_final = control.anderson_m;
        std::string failure_reason;
        double last_finite_criterion = std::numeric_limits<double>::quiet_NaN();
        double best_finite_criterion = std::numeric_limits<double>::quiet_NaN();
        double post_convergence_residual =
            std::numeric_limits<double>::quiet_NaN();
        double post_convergence_kappa_est =
            std::numeric_limits<double>::quiet_NaN();
        double post_convergence_score_residual =
            std::numeric_limits<double>::quiet_NaN();
        double post_convergence_fe_score_residual =
            std::numeric_limits<double>::quiet_NaN();
        double score_tol = std::numeric_limits<double>::quiet_NaN();
        std::string score_validation_status = "not_checked";
        std::vector<std::string> score_warning_codes;
        std::vector<std::string> score_warning_messages;
        bool score_exceeds_tolerance = false;
        bool fe_score_exceeds_tolerance = false;

        while (!verified) {
            int k = 1;
            double criterion = 100.0;
            std::vector<double> criterion_history;
            int kappa_bad_count = 0;
            std::vector<Eigen::VectorXd> beta_path;
            EigenVectorPairRingCore aa_hist;
            beta_path.reserve(kProfiledBetaPathRetained + 1);
            beta_path.push_back(beta);
            std::size_t beta_path_entries = 1;
            double kappa_row_max = -std::numeric_limits<double>::infinity();
            int aa_m = control.anderson_m;
            if (direct_nofe_fixed_tau && phase == 1) {
                v3_attempt_evaluations_start = phase2_evaluations;
            }
            criterion_history.reserve(
                static_cast<std::size_t>(std::max(stopcrit_current, 0)));
            if (aa_m > 0) {
                aa_hist.reset(static_cast<std::size_t>(aa_m + 1));
            }
            // runtime_v4 direct-route Phase 2 progress monitor (twin of the
            // NumPy progress_history): max-abs plain-map residual at the
            // current iterate, recorded at the same iterations as the
            // criterion. It drives stagnation, kappa and the Anderson memory
            // adaptation; the eps stop still uses `criterion`.
            std::vector<double> progress_history;
            double plain_residual_maxabs = std::numeric_limits<double>::quiet_NaN();
            int attempt_aa_m_start = aa_m;
            int attempt_aa_accepted = 0;

            while (criterion > eps_current && k < stopcrit_current &&
                   !std::isnan(criterion)) {
                const Eigen::VectorXd beta0 = beta;
                if (phase == 1) {
                    phase2_evaluations += 1;
                    if (phase2_evaluations > control.max_phase2_evaluations) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "phase2_evaluation_budget_exhausted";
                        phase2_budget_exhausted = true;
                        break;
                    }
                } else {
                    phase1_evaluations += 1;
                }
                Eigen::VectorXd beta_phi(beta.size());
                const bool valid = phi_step_profiled_core(
                    endog,
                    exog,
                    beta,
                    groups,
                    n_groups,
                    tau,
                    state.projection,
                    phase,
                    param,
                    beta_phi);
                if (!valid) {
                    criterion = std::numeric_limits<double>::quiet_NaN();
                    failure_reason = "invalid_phi_step";
                    break;
                }
                if (direct_nofe_fixed_tau && phase == 1) {
                    // runtime_v3: rho-free preconditioned score norm of the
                    // current iterate, (1+rho)*||phi(beta)-beta||_2 =
                    // ||P score_obs(beta)||_2 (twin of the NumPy tracker).
                    const Eigen::VectorXd v3_plain_step = beta_phi - beta;
                    const double v3_step_score_norm =
                        pyiols::kernel::stable_l2_norm(v3_plain_step) *
                        (1.0 + param);
                    if (std::isfinite(v3_step_score_norm) &&
                        v3_step_score_norm < v3_best_score_norm) {
                        v3_best_score_norm = v3_step_score_norm;
                        v3_best_score_rho = param;
                        v3_best_score_beta = beta;
                        v3_has_best_score_beta = true;
                    }
                    // runtime_v4: same operands as the plain-step criterion,
                    // so it equals `criterion` whenever no Anderson step is
                    // accepted; no extra map evaluation.
                    plain_residual_maxabs = max_abs_change_core(beta_phi, beta);
                }

                bool accepted_aa_step = false;
                if (aa_m > 0 && k > aa_m + 1) {
                    const Eigen::VectorXd f_k = beta_phi - beta;
                    aa_hist.push(beta, f_k);
                    const int m_cur = static_cast<int>(aa_hist.size()) - 1;
                    if (m_cur >= 1) {
                        Eigen::MatrixXd f_mat(beta.size(), m_cur);
                        for (int j = 1; j <= m_cur; ++j) {
                            f_mat.col(j - 1) =
                                aa_hist.f_back() -
                                aa_hist.f_at(aa_hist.size() - static_cast<std::size_t>(j + 1));
                        }
                        Eigen::VectorXd gamma =
                            pyiols::kernel::min_norm_lstsq_anderson(
                                f_mat,
                                aa_hist.f_back())
                                .col(0);
                        Eigen::VectorXd beta_aa = beta_phi;
                        for (int j = 0; j < m_cur; ++j) {
                            const std::size_t idx =
                                aa_hist.size() - static_cast<std::size_t>(j + 2);
                            const Eigen::VectorXd phi_prev =
                                aa_hist.x_at(idx) + aa_hist.f_at(idx);
                            beta_aa += gamma(j) * (phi_prev - beta_phi);
                        }
                        if (all_finite_vec(beta_aa)) {
                            if (phase == 1) {
                                phase2_evaluations += 1;
                                if (phase2_evaluations >
                                    control.max_phase2_evaluations) {
                                    criterion =
                                        std::numeric_limits<double>::quiet_NaN();
                                    failure_reason =
                                        "phase2_evaluation_budget_exhausted";
                                    phase2_budget_exhausted = true;
                                    break;
                                }
                            } else {
                                phase1_evaluations += 1;
                            }
                            Eigen::VectorXd beta_aa_phi(beta.size());
                            const bool aa_valid = phi_step_profiled_core(
                                endog,
                                exog,
                                beta_aa,
                                groups,
                                n_groups,
                                tau,
                                state.projection,
                                phase,
                                param,
                                beta_aa_phi);
                            if (aa_valid) {
                                const Eigen::VectorXd plain_residual = beta_phi - beta;
                                const Eigen::VectorXd aa_residual =
                                    beta_aa_phi - beta_aa;
                                const double res_plain =
                                    pyiols::kernel::stable_l2_norm(plain_residual);
                                const double res_aa =
                                    pyiols::kernel::stable_l2_norm(aa_residual);
                                if (res_aa < res_plain * 1.1) {
                                    beta = beta_aa;
                                    push_profiled_beta_path_core(
                                        beta_path,
                                        beta,
                                        beta_path_entries,
                                        kappa_row_max,
                                        kappa_min_entries);
                                    if (k >= 2) {
                                        criterion = max_abs_change_core(beta, beta0);
                                    }
                                    accepted_aa_step = true;
                                    attempt_aa_accepted += 1;
                                }
                            }
                        }
                    }
                } else if (aa_m > 0) {
                    aa_hist.push(beta, beta_phi - beta);
                }

                if (!accepted_aa_step) {
                    beta = beta_phi;
                    push_profiled_beta_path_core(
                        beta_path,
                        beta,
                        beta_path_entries,
                        kappa_row_max,
                        kappa_min_entries);
                }
                if (k >= 2) {
                    if (!accepted_aa_step) {
                        criterion = max_abs_change_core(beta, beta0);
                    }
                    const bool direct_progress = direct_nofe_fixed_tau && phase == 1;
                    if (direct_progress && std::isfinite(plain_residual_maxabs)) {
                        progress_history.push_back(plain_residual_maxabs);
                    }
                    if (record_criterion_progress_core(
                            criterion_history,
                            criterion,
                            phase,
                            tau,
                            eps_current,
                            control.max_phase2_evaluations,
                            last_finite_criterion,
                            best_finite_criterion,
                            failure_reason,
                            direct_progress ? &progress_history : nullptr)) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                    }
                }

                if (phase == 1 && std::abs(tau - 2.0) > 1e-10 && k >= 30 &&
                    k % 10 == 0 && beta_path_entries >= kappa_min_entries) {
                    if (direct_nofe_fixed_tau) {
                        // runtime_v3: the lifetime coordinate-mean ratio
                        // maximum is kept as a diagnostic only (+inf rows
                        // included: they are the pollution it exists to show).
                        if (!std::isnan(kappa_row_max) &&
                            kappa_row_max > -std::numeric_limits<double>::infinity()) {
                            v3_kappa_coordinate_row_max =
                                std::isnan(v3_kappa_coordinate_row_max)
                                    ? kappa_row_max
                                    : std::max(v3_kappa_coordinate_row_max,
                                               kappa_row_max);
                        }
                    }
                    // runtime_v3 direct route: windowed geometric-mean
                    // contraction (twin of _windowed_contraction_rate);
                    // legacy routes keep the lifetime coordinate statistic.
                    // runtime_v4: taken over the plain-map residual history,
                    // not accepted moves.
                    const double kappa_val = direct_nofe_fixed_tau ? windowed_contraction_rate_core(progress_history, kDirectKappaWindow) : kappa_row_max;
                    if (direct_nofe_fixed_tau) {
                        v3_kappa_monitor_last = kappa_val;
                        v3_kappa_monitor_checks += 1;
                        if (std::isfinite(kappa_val)) {
                            v3_kappa_monitor_max =
                                std::isnan(v3_kappa_monitor_max)
                                    ? kappa_val
                                    : std::max(v3_kappa_monitor_max, kappa_val);
                        }
                    }
                    const int aa_m_new = adapt_anderson_m_core(aa_m, kappa_val);
                    if (aa_m_new > aa_m) {
                        aa_m = aa_m_new;
                        kappa_bad_count = 0;
                        if (direct_nofe_fixed_tau && aa_m > 0) {
                            // NumPy parity: the history now holds up to aa_m lags.
                            aa_hist.grow(static_cast<std::size_t>(aa_m + 1));
                        }
                    } else if (std::isfinite(kappa_val) && kappa_val >= 1.0) {
                        kappa_bad_count += 1;
                    } else {
                        kappa_bad_count = 0;
                    }
                    if (kappa_bad_count >= 3) {
                        criterion = std::numeric_limits<double>::quiet_NaN();
                        failure_reason = "noncontractive";
                    }
                }

                k += 1;
                iteration_count = k;

                if (std::isnan(criterion)) {
                    restart_index += 1;
                    if (phase == 1 && !phase2_budget_exhausted &&
                        restart_index < static_cast<int>(rho_path.size())) {
                        if (direct_nofe_fixed_tau) {
                            Phase2AttemptTraceCore v3_record;
                            // restart_index was already advanced: the attempt
                            // that just ended is the previous one.
                            v3_record.attempt = restart_index - 1;
                            v3_record.rho = param;
                            v3_record.iterations = k;
                            v3_record.evaluations =
                                phase2_evaluations - v3_attempt_evaluations_start;
                            v3_record.failure_reason = failure_reason;
                            v3_record.has_final_criterion =
                                std::isfinite(last_finite_criterion);
                            v3_record.final_criterion = last_finite_criterion;
                            v3_record.best_criterion =
                                criterion_history.empty()
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : *std::min_element(criterion_history.begin(),
                                                        criterion_history.end());
                            v3_record.kappa_last = v3_kappa_monitor_last;
                            v3_record.start_policy = v3_attempt_start_policy;
                            set_direct_progress_trace_fields(
                                v3_record,
                                progress_history,
                                attempt_aa_m_start,
                                aa_m,
                                attempt_aa_accepted);
                            v3_phase2_attempt_trace.push_back(v3_record);
                        }
                        param = rho_path[static_cast<std::size_t>(restart_index)];
                        criterion = 100.0;
                        beta = beta_outer_initial;
                        if (direct_nofe_fixed_tau) {
                            // runtime_v3 restart start policy.
                            v3_attempt_start_policy = kDirectRestartStartPhase1;
                            if (v3_restart_from_best && v3_has_best_score_beta) {
                                beta = v3_best_score_beta;
                                v3_phase2_warm_restarts += 1;
                                v3_attempt_start_policy = kDirectRestartStartBestScore;
                            }
                            v3_attempt_evaluations_start = phase2_evaluations;
                            v3_kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
                        }
                        k = 1;
                        criterion_history.clear();
                        kappa_bad_count = 0;
                        beta_path.clear();
                        beta_path.push_back(beta);
                        beta_path_entries = 1;
                        kappa_row_max = -std::numeric_limits<double>::infinity();
                        aa_hist.clear();
                        // runtime_v4: aa_m itself is still carried over (v3).
                        progress_history.clear();
                        plain_residual_maxabs = std::numeric_limits<double>::quiet_NaN();
                        attempt_aa_m_start = aa_m;
                        attempt_aa_accepted = 0;
                    } else if (phase == 0) {
                        break;
                    }
                }
            }

            bool converged = !std::isnan(criterion) && criterion <= eps_current;
            has_final_criterion = !std::isnan(criterion);
            final_criterion = criterion;
            if (converged) {
                Eigen::VectorXd beta_check(beta.size());
                bool valid = phi_step_profiled_core(
                    endog,
                    exog,
                    beta,
                    groups,
                    n_groups,
                    tau,
                    state.projection,
                    phase,
                    param,
                    beta_check);
#if defined(PYIOLS_DIRECT_TEST_HOOKS)
                if (direct_nofe_fixed_tau && direct_test_hook_invalidate_post_check(phase)) {
                    valid = false;
                }
#endif
                if (valid) {
                    const double residual = max_abs_change_core(beta_check, beta);
                    post_convergence_residual = residual;
                    if (residual > eps_current * 10.0) {
                        converged = false;
                        failure_reason = "post_convergence_residual";
                    }
                } else if (direct_nofe_fixed_tau) {
                    // runtime_v4: an invalid final verification map clears
                    // native convergence (v3 kept it). No residual exists, so
                    // an earlier attempt's post-check values are not reported.
                    converged = false;
                    failure_reason = "post_convergence_invalid_map";
                    v4_post_convergence_invalid_map_counts[phase] += 1;
                    post_convergence_residual = std::numeric_limits<double>::quiet_NaN();
                    post_convergence_kappa_est = std::numeric_limits<double>::quiet_NaN();
                }
                if (converged && phase == 1) {
                    const double score_residual = profiled_slope_score_residual_core(
                        endog,
                        exog,
                        beta,
                        groups,
                        n_groups,
                        state.x_demeaned,
                        tau);
                    const double fe_score_residual = profiled_fe_score_residual_core(
                        endog,
                        exog,
                        beta,
                        groups,
                        n_groups,
                        tau);
                    score_tol = std::max(eps_current * 10.0, 1e-8);
                    post_convergence_score_residual = score_residual;
                    score_exceeds_tolerance =
                        !std::isfinite(score_residual) || score_residual > score_tol;
                    if (score_exceeds_tolerance) {
                        score_warning_codes.push_back(
                            "post_convergence_score_residual");
                        score_warning_messages.push_back(
                            "Slope score residual exceeds tolerance at this "
                            "tau-path state.");
                        score_validation_status = "diagnostic_warning";
                    } else if (score_validation_status == "not_checked") {
                        score_validation_status = "passed";
                    }
                    post_convergence_fe_score_residual = fe_score_residual;
                    fe_score_exceeds_tolerance =
                        !std::isfinite(fe_score_residual) ||
                        fe_score_residual > score_tol;
                    if (fe_score_exceeds_tolerance) {
                        score_warning_codes.push_back(
                            "post_convergence_fe_score_residual");
                        score_warning_messages.push_back(
                            "Recovered-FE score residual exceeds tolerance at "
                            "this tau-path state.");
                        score_validation_status = "diagnostic_warning";
                    } else if (score_validation_status == "not_checked") {
                        score_validation_status = "passed";
                    }
                }
                if (converged && beta_path.size() >= 4 &&
                    !(phase == 1 && std::abs(tau - 2.0) <= 1e-10)) {
                    const double d0 =
                        max_abs_change_core(beta_path[beta_path.size() - 1],
                                            beta_path[beta_path.size() - 2]);
                    const double d1 =
                        max_abs_change_core(beta_path[beta_path.size() - 2],
                                            beta_path[beta_path.size() - 3]);
                    const double d2 =
                        max_abs_change_core(beta_path[beta_path.size() - 3],
                                            beta_path[beta_path.size() - 4]);
                    if (d1 > 1e-15 && d2 > 1e-15) {
                        const double kappa_est = d0 / d1;
                        post_convergence_kappa_est = kappa_est;
                        if (kappa_est >= 1.0) {
                            converged = false;
                            failure_reason = "post_convergence_noncontractive";
                        }
                    }
                }
            }

            if (direct_nofe_fixed_tau && phase == 1) {
                Phase2AttemptTraceCore v3_record;
                // restart_index can equal rho_path.size() when the last
                // attempt was killed in-loop with no rho left.
                v3_record.attempt =
                    std::min(restart_index, static_cast<int>(rho_path.size()) - 1);
                v3_record.rho = param;
                // k (not iteration_count) so an attempt that ended on its
                // first map does not inherit the previous count.
                v3_record.iterations = k;
                v3_record.evaluations =
                    phase2_evaluations - v3_attempt_evaluations_start;
                // a finite criterion above eps here means the per-rho cap
                // (stopcrit) ended the attempt; failure_reason is per phase
                // and would be stale.
                v3_record.failure_reason =
                    converged ? std::string()
                              : ((!std::isnan(criterion) && criterion > eps_current)
                                     ? std::string("stopcrit_reached")
                                     : failure_reason);
                v3_record.has_final_criterion = has_final_criterion;
                v3_record.final_criterion = final_criterion;
                v3_record.best_criterion =
                    criterion_history.empty()
                        ? std::numeric_limits<double>::quiet_NaN()
                        : *std::min_element(criterion_history.begin(),
                                            criterion_history.end());
                v3_record.kappa_last = v3_kappa_monitor_last;
                v3_record.start_policy = v3_attempt_start_policy;
                set_direct_progress_trace_fields(
                    v3_record,
                    progress_history,
                    attempt_aa_m_start,
                    aa_m,
                    attempt_aa_accepted);
                v3_phase2_attempt_trace.push_back(v3_record);
            }
            if (!converged && phase == 1 && !phase2_budget_exhausted &&
                restart_index < static_cast<int>(rho_path.size()) - 1) {
                restart_index = std::min(
                    restart_index + std::max(restart_index, 5),
                    static_cast<int>(rho_path.size()) - 1);
                param = rho_path[static_cast<std::size_t>(restart_index)];
                beta = beta_outer_initial;
                if (direct_nofe_fixed_tau) {
                    // runtime_v3 restart start policy.
                    v3_attempt_start_policy = kDirectRestartStartPhase1;
                    if (v3_restart_from_best && v3_has_best_score_beta) {
                        beta = v3_best_score_beta;
                        v3_phase2_warm_restarts += 1;
                        v3_attempt_start_policy = kDirectRestartStartBestScore;
                    }
                    v3_kappa_monitor_last = std::numeric_limits<double>::quiet_NaN();
                }
            } else {
                verified = true;
            }
            aa_m_final = aa_m;

            if (phase == 0) {
                phase1_attempts += 1;
                const bool candidate_usable =
                    converged &&
                    phase2_initialization_usable_core(
                        endog,
                        exog,
                        beta,
                        groups,
                        n_groups,
                        tau,
                        state.projection,
                        control);
                if (candidate_usable) {
                    phase1_accepted += 1;
                    phase1_accepted_beta = beta;
                    phase1_accepted_linear = initial_linear_profiled_core(
                        endog, exog, beta, groups, n_groups, tau);
                } else {
                    phase1_rejected += 1;
                    beta = phase1_accepted_beta;
                    if (converged && failure_reason.empty()) {
                        failure_reason = "phase1_candidate_unusable";
                    }
                }
                final_diag.converged = candidate_usable;
            } else {
                if (!converged && !direct_nofe_fixed_tau) {
                    // Legacy tau-path semantics: hand the anchor/previous
                    // state back to the caller. The runtime_v2 direct route
                    // keeps the final iterate of the last rho attempt so that
                    // the external exact-score audit can still accept it.
                    beta = beta_outer_initial;
                }
                if (direct_nofe_fixed_tau) {
                    final_diag.nonconverged_iterate_policy =
                        "last_attempt_final_iterate";
                    if (v3_restart_from_best) {
                        final_diag.nonconverged_iterate_policy =
                            kDirectRestartStartBestScore;
                        if (!converged && v3_has_best_score_beta) {
                            // runtime_v3: hand back the smallest
                            // preconditioned-score iterate of the whole
                            // Phase 2; the external audit decides.
                            beta = v3_best_score_beta;
                        }
                    }
                    final_diag.direct_restart_start = control.direct_restart_start;
                    final_diag.kappa_coordinate_row_max = v3_kappa_coordinate_row_max;
                    final_diag.kappa_monitor_last = v3_kappa_monitor_last;
                    final_diag.kappa_monitor_max = v3_kappa_monitor_max;
                    final_diag.kappa_monitor_checks = v3_kappa_monitor_checks;
                    final_diag.best_score_norm = v3_best_score_norm;
                    final_diag.best_score_rho = v3_best_score_rho;
                    final_diag.phase2_warm_restarts = v3_phase2_warm_restarts;
                    final_diag.phase2_attempts =
                        static_cast<int>(v3_phase2_attempt_trace.size());
                    final_diag.phase2_attempt_trace = v3_phase2_attempt_trace;
                    final_diag.post_convergence_invalid_map_count =
                        v4_post_convergence_invalid_map_counts[1];
                    final_diag.phase1_post_convergence_invalid_map_count =
                        v4_post_convergence_invalid_map_counts[0];
                }
                final_diag.converged = converged;
            }

            final_diag.phase = phase == 0 ? "P1" : "P2";
            final_diag.iterations = iteration_count;
            final_diag.restarts = restart_index;
            final_diag.has_final_rho = phase != 0;
            final_diag.final_rho = param;
            final_diag.rho_path.clear();
            if (phase != 0) {
                const int upto =
                    std::min(restart_index + 1, static_cast<int>(rho_path.size()));
                for (int idx = 0; idx < upto; ++idx) {
                    final_diag.rho_path.push_back(rho_path[static_cast<std::size_t>(idx)]);
                }
            }
            final_diag.has_final_criterion = has_final_criterion;
            final_diag.final_criterion = final_criterion;
            final_diag.anderson_m_initial = control.anderson_m;
            final_diag.anderson_m_final = aa_m_final;
            final_diag.failure_reason =
                final_diag.converged
                    ? ""
                    : (!failure_reason.empty() ? failure_reason : "not_converged");
            final_diag.phase1_attempts = phase1_attempts;
            final_diag.phase1_accepted = phase1_accepted;
            final_diag.phase1_rejected = phase1_rejected;
            final_diag.phase1_evaluations = phase1_evaluations;
            final_diag.phase2_evaluations = phase2_evaluations;
            final_diag.max_phase2_evaluations = control.max_phase2_evaluations;
            final_diag.last_finite_criterion = last_finite_criterion;
            final_diag.best_finite_criterion = best_finite_criterion;
            final_diag.fe_state_policy = "analytical_profile";
            final_diag.anderson_policy = "beta_anderson";
            final_diag.post_convergence_residual = post_convergence_residual;
            final_diag.post_convergence_kappa_est = post_convergence_kappa_est;
            final_diag.post_convergence_score_residual =
                post_convergence_score_residual;
            final_diag.post_convergence_fe_score_residual =
                post_convergence_fe_score_residual;
            final_diag.score_tolerance = score_tol;
            final_diag.score_validation_policy = "diagnostic";
            final_diag.score_validation_status = score_validation_status;
            final_diag.score_warning_codes = score_warning_codes;
            final_diag.score_warning_messages = score_warning_messages;
            final_diag.post_convergence_score_exceeds_tolerance =
                score_exceeds_tolerance;
            final_diag.post_convergence_fe_score_exceeds_tolerance =
                fe_score_exceeds_tolerance;
        }
        if (phase == 1) {
            break;
        }
    }

    FullrecoverConcentratedCore out;
    out.beta = beta;
    out.linear = initial_linear_profiled_core(endog, exog, beta, groups, n_groups, tau);
    out.has_linear = all_finite_vec(out.linear);
    out.iteration = final_diag;
    return out;
}

std::vector<double> tau_path_grid_core(double tau_target, int n_steps) {
    const int n_steps_actual = std::max(1, n_steps);
    const double delta = std::abs(tau_target - 2.0);
    const double sign = tau_target > 2.0 ? 1.0 : -1.0;
    std::vector<double> grid;
    grid.reserve(static_cast<std::size_t>(n_steps_actual));
    for (int idx = 1; idx <= n_steps_actual; ++idx) {
        const double u = static_cast<double>(idx) / static_cast<double>(n_steps_actual);
        const double frac = 1.0 - std::pow(1.0 - u, 2.0);
        grid.push_back(2.0 + sign * delta * frac);
    }
    return grid;
}

constexpr int kTauPathFrontierRetryBudget = 4;

double tau_path_next_try_core(
    double tau_goal,
    double tau_current,
    double sign,
    const std::optional<double>& step_cap) {
    const double remaining_gap = std::abs(tau_goal - tau_current);
    const double step =
        step_cap.has_value() && std::isfinite(*step_cap) && *step_cap > 0.0
            ? std::min(remaining_gap, *step_cap)
            : remaining_gap;
    return tau_current + sign * step;
}

double tau_path_refined_step_cap_core(double failed_gap, double min_dtau) {
    return std::max(min_dtau, failed_gap / 2.0);
}

FullrecoverIterationCore combine_tau_path_iteration_core(
    bool converged,
    const std::string& failure_reason,
    const std::vector<FullrecoverIterationCore>& diagnostics) {
    if (diagnostics.empty()) {
        FullrecoverIterationCore out;
        out.failure_reason = failure_reason.empty() ? "not_started" : failure_reason;
        out.anderson_m_final = 0;
        return out;
    }
    const FullrecoverIterationCore& first = diagnostics.front();
    const FullrecoverIterationCore& final = diagnostics.back();
    FullrecoverIterationCore out;
    out.converged = converged;
    out.phase = "tau_path";
    out.iterations = 0;
    out.restarts = 0;
    out.phase2_evaluations = 0;
    out.max_phase2_evaluations = 0;
    out.joint_hdfe_phi_calls = 0;
    out.joint_hdfe_phi_demean_sweeps = 0;
    out.phase2_rho_restart_events = 0;
    out.phase2_rho_attempts = 0;
    out.rho_path.clear();
    for (const FullrecoverIterationCore& diagnostic : diagnostics) {
        out.iterations += diagnostic.iterations;
        out.restarts += diagnostic.restarts;
        out.phase2_evaluations += diagnostic.phase2_evaluations;
        out.max_phase2_evaluations = std::max(
            out.max_phase2_evaluations, diagnostic.max_phase2_evaluations);
        out.joint_hdfe_phi_calls += diagnostic.joint_hdfe_phi_calls;
        out.joint_hdfe_phi_demean_sweeps += diagnostic.joint_hdfe_phi_demean_sweeps;
        out.phase2_rho_restart_events += diagnostic.phase2_rho_restart_events;
        out.phase2_rho_attempts += diagnostic.phase2_rho_attempts;
        out.rho_path.insert(
            out.rho_path.end(),
            diagnostic.rho_path.begin(),
            diagnostic.rho_path.end());
    }
    out.has_final_rho = final.has_final_rho;
    out.final_rho = final.final_rho;
    out.has_final_criterion = final.has_final_criterion;
    out.final_criterion = final.final_criterion;
    out.anderson_m_initial = first.anderson_m_initial;
    out.anderson_m_final = final.anderson_m_final;
    out.research_tiered_fe_tol = final.research_tiered_fe_tol;
    out.research_inner_fe_demean_tol = final.research_inner_fe_demean_tol;
    out.failure_reason = converged ? "" : (!failure_reason.empty()
                                               ? failure_reason
                                               : final.failure_reason);
    return out;
}

TauPathRecordCore make_path_record_core(
    double tau,
    std::optional<double> from_tau,
    std::optional<double> step,
    const FullrecoverConcentratedCore& solve,
    std::optional<double> beta_init_source_tau,
    std::optional<double> rho_hint) {
    TauPathRecordCore record;
    record.tau = tau;
    if (from_tau.has_value()) {
        record.has_from_tau = true;
        record.from_tau = *from_tau;
    }
    if (step.has_value()) {
        record.has_step = true;
        record.step = *step;
    }
    record.converged = solve.iteration.converged;
    record.has_beta = solve.iteration.converged;
    record.beta = solve.beta;
    record.has_rho = solve.iteration.has_final_rho;
    record.rho = solve.iteration.final_rho;
    record.iterations = solve.iteration.iterations;
    record.restarts = solve.iteration.restarts;
    record.anderson_m_initial = solve.iteration.anderson_m_initial;
    record.anderson_m_final = solve.iteration.anderson_m_final;
    record.failure_reason = solve.iteration.failure_reason;
    if (beta_init_source_tau.has_value()) {
        record.has_beta_init_source_tau = true;
        record.beta_init_source_tau = *beta_init_source_tau;
    }
    if (rho_hint.has_value()) {
        record.has_rho_hint = true;
        record.rho_hint = *rho_hint;
    }
    record.has_linear = solve.iteration.converged && solve.has_linear;
    record.linear = solve.linear;
    record.diagnostics = solve.iteration;
    return record;
}

FullrecoverTauPathCore solve_fullrecover_tau_path_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    double tau_target,
    const FullrecoverControlCore& control) {
    const FullrecoverInvariantCore state =
        fullrecover_invariant_core(exog, groups, n_groups);
    FullrecoverTauPathCore out;
    out.tau_target = tau_target;
    const FullrecoverConcentratedCore anchor = concentrated_fullrecover_profiled_core(
        endog,
        exog,
        groups,
        n_groups,
        2.0,
        control,
        state,
        std::nullopt,
        std::nullopt,
        std::nullopt);
    std::vector<FullrecoverIterationCore> diagnostics{anchor.iteration};
    out.path.push_back(make_path_record_core(
        2.0,
        std::nullopt,
        std::nullopt,
        anchor,
        std::nullopt,
        std::nullopt));
    if (!anchor.iteration.converged) {
        out.beta = Eigen::VectorXd::Constant(anchor.beta.size(), std::numeric_limits<double>::quiet_NaN());
        out.has_linear = false;
        out.tau_fit = 2.0;
        out.converged = false;
        out.iteration = anchor.iteration;
        out.path_status = "anchor_failed";
        out.tau_failed = 2.0;
        out.has_tau_failed = true;
        out.failure_reason =
            anchor.iteration.failure_reason.empty() ? "tau2_anchor_failed"
                                                    : anchor.iteration.failure_reason;
        return out;
    }
    if (!anchor.iteration.has_final_rho) {
        out.beta = Eigen::VectorXd::Constant(anchor.beta.size(), std::numeric_limits<double>::quiet_NaN());
        out.has_linear = false;
        out.tau_fit = 2.0;
        out.converged = false;
        out.iteration = anchor.iteration;
        out.iteration.converged = false;
        out.iteration.failure_reason = "tau2_anchor_missing_rho";
        out.path_status = "anchor_failed";
        out.tau_failed = 2.0;
        out.has_tau_failed = true;
        out.failure_reason = "tau2_anchor_missing_rho";
        return out;
    }
    if (std::abs(tau_target - 2.0) <= 1e-10) {
        out.beta = anchor.beta;
        out.linear = anchor.linear;
        out.has_linear = anchor.has_linear;
        out.tau_fit = 2.0;
        out.converged = true;
        out.iteration = anchor.iteration;
        out.path_status = "target_reached";
        out.tau_attained = 2.0;
        out.has_final_rho = true;
        out.final_rho = anchor.iteration.final_rho;
        return out;
    }

    const std::vector<double> grid =
        tau_path_grid_core(tau_target, control.tau_path_steps);
    const double sign = tau_target > 2.0 ? 1.0 : -1.0;
    const double delta = std::abs(tau_target - 2.0);
    const double min_dtau =
        control.has_min_dtau && control.min_dtau > 0.0
            ? control.min_dtau
            : std::max(delta / std::max(static_cast<double>(control.tau_path_steps) * 128.0, 1.0), 1e-3);
    const double path_tol = std::max(min_dtau * 0.1, 1e-12);
    const bool use_local_frontier_refinement = sign < 0.0;
    Eigen::VectorXd beta_current = anchor.beta;
    Eigen::VectorXd beta_previous = anchor.beta;
    Eigen::VectorXd linear_current = anchor.linear;
    Eigen::VectorXd linear_previous = anchor.linear;
    double rho_previous = anchor.iteration.final_rho;
    double tau_current = 2.0;
    std::string failure_reason;
    std::optional<double> failed_tau;

    for (const double tau_goal : grid) {
        if (sign * (tau_goal - tau_current) <= path_tol) {
            continue;
        }
        std::optional<double> frontier_step_cap;
        bool frontier_locked = false;
        int frontier_expensive_failures = 0;
        while (sign * (tau_goal - tau_current) > path_tol) {
            const double tau_try = tau_path_next_try_core(
                tau_goal, tau_current, sign, frontier_step_cap);
            if (out.n_path_evaluations >= control.max_path_evaluations) {
                failure_reason = "max_path_evaluations_reached";
                failed_tau = tau_try;
                break;
            }
            const double tau_from = tau_current;
            out.n_path_evaluations += 1;
            FullrecoverConcentratedCore step =
                concentrated_fullrecover_profiled_core(
                    endog,
                    exog,
                    groups,
                    n_groups,
                    tau_try,
                    control,
                    state,
                    beta_previous,
                    linear_previous,
                    rho_previous);
            diagnostics.push_back(step.iteration);
            const TauPathRecordCore step_record = make_path_record_core(
                tau_try,
                tau_current,
                std::abs(tau_try - tau_current),
                step,
                tau_current,
                rho_previous);
            const bool score_warning_refinement =
                use_local_frontier_refinement &&
                tau_path_score_warning_refinement_step_core(step.iteration);
            if (step.iteration.converged) {
                if (score_warning_refinement) {
                    out.n_path_failures += 1;
                    frontier_locked = true;
                    frontier_expensive_failures += 1;
                    const double gap = std::abs(tau_try - tau_current);
                    if (gap <= min_dtau * (1.0 + 1e-12)) {
                        failure_reason = "min_dtau_reached";
                        failed_tau = tau_try;
                        break;
                    }
                    if (frontier_expensive_failures > kTauPathFrontierRetryBudget) {
                        failure_reason = step.iteration.failure_reason.empty()
                                             ? "not_converged"
                                             : step.iteration.failure_reason;
                        failed_tau = tau_try;
                        break;
                    }
                    frontier_step_cap =
                        tau_path_refined_step_cap_core(gap, min_dtau);
                    continue;
                }
                out.path.push_back(step_record);
                out.n_path_successes += 1;
                tau_current = tau_try;
                beta_current = step.beta;
                beta_previous = step.beta;
                linear_current = step.linear;
                linear_previous = step.linear;
                if (step.iteration.has_final_rho) {
                    rho_previous = step.iteration.final_rho;
                }
                if (use_local_frontier_refinement && frontier_locked) {
                    const double success_gap = std::abs(tau_current - tau_from);
                    frontier_step_cap = std::max(
                        min_dtau,
                        std::min(
                            success_gap,
                            frontier_step_cap.has_value() ? *frontier_step_cap
                                                          : success_gap));
                } else {
                    frontier_step_cap.reset();
                }
                if (sign * (tau_goal - tau_current) <= path_tol) {
                    break;
                }
                continue;
            }
            out.path.push_back(step_record);
            out.n_path_failures += 1;
            const double gap = std::abs(tau_try - tau_current);
            if (gap <= min_dtau * (1.0 + 1e-12)) {
                failure_reason = step.iteration.failure_reason.empty()
                                     ? "min_dtau_reached"
                                     : step.iteration.failure_reason;
                failed_tau = tau_try;
                break;
            }
            if (use_local_frontier_refinement &&
                tau_path_expensive_failed_step_core(step.iteration, control)) {
                frontier_locked = true;
                frontier_expensive_failures += 1;
                if (frontier_expensive_failures > kTauPathFrontierRetryBudget) {
                    failure_reason = step.iteration.failure_reason.empty()
                                         ? "not_converged"
                                         : step.iteration.failure_reason;
                    failed_tau = tau_try;
                    break;
                }
            }
            frontier_step_cap = tau_path_refined_step_cap_core(gap, min_dtau);
        }
        if (!failure_reason.empty()) {
            break;
        }
    }
    const bool target_reached = failure_reason.empty();
    out.beta = beta_current;
    out.linear = linear_current;
    out.has_linear = all_finite_vec(linear_current);
    out.tau_fit = tau_current;
    out.converged = target_reached;
    out.iteration =
        combine_tau_path_iteration_core(target_reached, failure_reason, diagnostics);
    out.path_status =
        target_reached ? "target_reached" : "partial_target_not_reached";
    out.tau_attained = tau_current;
    if (failed_tau.has_value()) {
        out.has_tau_failed = true;
        out.tau_failed = *failed_tau;
    }
    out.has_tau_gap = true;
    out.tau_gap = std::abs(tau_target - tau_current);
    out.failure_reason = failure_reason;
    out.has_final_rho = true;
    out.final_rho = rho_previous;
    out.has_min_dtau = true;
    out.min_dtau = min_dtau;
    return out;
}

FullrecoverTauPathCore solve_fullrecover_tau_path_manyway_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const GroupBundleCore& bundle,
    double tau_target,
    const FullrecoverControlCore& control) {
    const FullrecoverInvariantCore state =
        fullrecover_invariant_bundle_core(exog, bundle, tau_target, control);
    FullrecoverTauPathCore out;
    out.tau_target = tau_target;
    const FullrecoverConcentratedCore anchor = concentrated_fullrecover_joint_core(
        endog,
        exog,
        bundle,
        2.0,
        control,
        state,
        std::nullopt,
        std::nullopt,
        std::nullopt);
    std::vector<FullrecoverIterationCore> diagnostics{anchor.iteration};
    out.path.push_back(make_path_record_core(
        2.0,
        std::nullopt,
        std::nullopt,
        anchor,
        std::nullopt,
        std::nullopt));
    if (!anchor.iteration.converged) {
        out.beta = Eigen::VectorXd::Constant(
            anchor.beta.size(),
            std::numeric_limits<double>::quiet_NaN());
        out.has_linear = false;
        out.tau_fit = 2.0;
        out.converged = false;
        out.iteration = anchor.iteration;
        out.path_status = "anchor_failed";
        out.tau_failed = 2.0;
        out.has_tau_failed = true;
        out.failure_reason =
            anchor.iteration.failure_reason.empty() ? "tau2_anchor_failed"
                                                    : anchor.iteration.failure_reason;
        return out;
    }
    if (!anchor.iteration.has_final_rho) {
        out.beta = Eigen::VectorXd::Constant(
            anchor.beta.size(),
            std::numeric_limits<double>::quiet_NaN());
        out.has_linear = false;
        out.tau_fit = 2.0;
        out.converged = false;
        out.iteration = anchor.iteration;
        out.iteration.converged = false;
        out.iteration.failure_reason = "tau2_anchor_missing_rho";
        out.path_status = "anchor_failed";
        out.tau_failed = 2.0;
        out.has_tau_failed = true;
        out.failure_reason = "tau2_anchor_missing_rho";
        return out;
    }
    if (std::abs(tau_target - 2.0) <= 1e-10) {
        out.beta = anchor.beta;
        out.linear = anchor.linear;
        out.has_linear = anchor.has_linear;
        out.tau_fit = 2.0;
        out.converged = true;
        out.iteration = anchor.iteration;
        out.path_status = "target_reached";
        out.tau_attained = 2.0;
        out.has_final_rho = true;
        out.final_rho = anchor.iteration.final_rho;
        return out;
    }

    const std::vector<double> grid =
        tau_path_grid_core(tau_target, control.tau_path_steps);
    const double sign = tau_target > 2.0 ? 1.0 : -1.0;
    const double delta = std::abs(tau_target - 2.0);
    const double min_dtau =
        control.has_min_dtau && control.min_dtau > 0.0
            ? control.min_dtau
            : std::max(
                  delta / std::max(
                              static_cast<double>(control.tau_path_steps) * 128.0,
                              1.0),
                  1e-3);
    const double path_tol = std::max(min_dtau * 0.1, 1e-12);
    const bool use_local_frontier_refinement = sign < 0.0;
    const bool accept_score_warning_path_steps =
        control.speed_fullrecover_accept_score_warning_path_steps;
    Eigen::VectorXd beta_current = anchor.beta;
    Eigen::VectorXd beta_previous = anchor.beta;
    Eigen::VectorXd linear_current = anchor.linear;
    Eigen::VectorXd linear_previous = anchor.linear;
    double rho_previous = anchor.iteration.final_rho;
    double tau_current = 2.0;
    std::string failure_reason;
    std::optional<double> failed_tau;

    for (const double tau_goal : grid) {
        if (sign * (tau_goal - tau_current) <= path_tol) {
            continue;
        }
        std::optional<double> frontier_step_cap;
        bool frontier_locked = false;
        int frontier_expensive_failures = 0;
        while (sign * (tau_goal - tau_current) > path_tol) {
            const double tau_try = tau_path_next_try_core(
                tau_goal, tau_current, sign, frontier_step_cap);
            if (out.n_path_evaluations >= control.max_path_evaluations) {
                failure_reason = "max_path_evaluations_reached";
                failed_tau = tau_try;
                break;
            }
            const double tau_from = tau_current;
            out.n_path_evaluations += 1;
            FullrecoverConcentratedCore step =
                concentrated_fullrecover_joint_core(
                    endog,
                    exog,
                    bundle,
                    tau_try,
                    control,
                    state,
                    beta_previous,
                    linear_previous,
                    rho_previous);
            diagnostics.push_back(step.iteration);
            const TauPathRecordCore step_record = make_path_record_core(
                tau_try,
                tau_current,
                std::abs(tau_try - tau_current),
                step,
                tau_current,
                rho_previous);
            const bool score_warning_refinement =
                use_local_frontier_refinement &&
                !accept_score_warning_path_steps &&
                tau_path_score_warning_refinement_step_core(step.iteration);
            if (step.iteration.converged) {
                if (score_warning_refinement) {
                    out.n_path_failures += 1;
                    frontier_locked = true;
                    frontier_expensive_failures += 1;
                    const double gap = std::abs(tau_try - tau_current);
                    if (gap <= min_dtau * (1.0 + 1e-12)) {
                        failure_reason = "min_dtau_reached";
                        failed_tau = tau_try;
                        break;
                    }
                    if (frontier_expensive_failures > kTauPathFrontierRetryBudget) {
                        failure_reason = step.iteration.failure_reason.empty()
                                             ? "not_converged"
                                             : step.iteration.failure_reason;
                        failed_tau = tau_try;
                        break;
                    }
                    frontier_step_cap =
                        tau_path_refined_step_cap_core(gap, min_dtau);
                    continue;
                }
                out.path.push_back(step_record);
                out.n_path_successes += 1;
                tau_current = tau_try;
                beta_current = step.beta;
                beta_previous = step.beta;
                linear_current = step.linear;
                linear_previous = step.linear;
                if (step.iteration.has_final_rho) {
                    rho_previous = step.iteration.final_rho;
                }
                if (use_local_frontier_refinement && frontier_locked) {
                    const double success_gap = std::abs(tau_current - tau_from);
                    frontier_step_cap = std::max(
                        min_dtau,
                        std::min(
                            success_gap,
                            frontier_step_cap.has_value() ? *frontier_step_cap
                                                          : success_gap));
                } else {
                    frontier_step_cap.reset();
                }
                if (sign * (tau_goal - tau_current) <= path_tol) {
                    break;
                }
                continue;
            }
            out.path.push_back(step_record);
            out.n_path_failures += 1;
            const double gap = std::abs(tau_try - tau_current);
            if (gap <= min_dtau * (1.0 + 1e-12)) {
                failure_reason = step.iteration.failure_reason.empty()
                                     ? "min_dtau_reached"
                                     : step.iteration.failure_reason;
                failed_tau = tau_try;
                break;
            }
            if (use_local_frontier_refinement &&
                tau_path_expensive_failed_step_core(step.iteration, control)) {
                frontier_locked = true;
                frontier_expensive_failures += 1;
                if (frontier_expensive_failures > kTauPathFrontierRetryBudget) {
                    failure_reason = step.iteration.failure_reason.empty()
                                         ? "not_converged"
                                         : step.iteration.failure_reason;
                    failed_tau = tau_try;
                    break;
                }
            }
            frontier_step_cap = tau_path_refined_step_cap_core(gap, min_dtau);
        }
        if (!failure_reason.empty()) {
            break;
        }
    }
    const bool target_reached = failure_reason.empty();
    out.beta = beta_current;
    out.linear = linear_current;
    out.has_linear = all_finite_vec(linear_current);
    out.tau_fit = tau_current;
    out.converged = target_reached;
    out.iteration =
        combine_tau_path_iteration_core(target_reached, failure_reason, diagnostics);
    out.path_status =
        target_reached ? "target_reached" : "partial_target_not_reached";
    out.tau_attained = tau_current;
    if (failed_tau.has_value()) {
        out.has_tau_failed = true;
        out.tau_failed = *failed_tau;
    }
    out.has_tau_gap = true;
    out.tau_gap = std::abs(tau_target - tau_current);
    out.failure_reason = failure_reason;
    out.has_final_rho = true;
    out.final_rho = rho_previous;
    out.has_min_dtau = true;
    out.min_dtau = min_dtau;
    out.speed_accept_score_warning_path_steps =
        accept_score_warning_path_steps;
    return out;
}

FullrecoverTauPathCore solve_fullcancel_tau_path_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog,
    const GroupBundleCore& bundle,
    double tau_target,
    const FullrecoverControlCore& control) {
    FullrecoverTauPathCore out;
    out.tau_target = tau_target;
    FullrecoverControlCore path_control = control;
    if (tau_target > 2.0 + 1e-10 && path_control.anderson_m != 0) {
        path_control.anderson_m = 0;
    }
    FullcancelPhiWorkspaceCore phi_workspace;
    const FullrecoverConcentratedCore anchor = concentrated_fullcancel_core(
        endog,
        exog,
        bundle,
        2.0,
        path_control,
        std::nullopt,
        std::nullopt,
        &phi_workspace);
    std::vector<FullrecoverIterationCore> diagnostics{anchor.iteration};
    out.path.push_back(make_path_record_core(
        2.0,
        std::nullopt,
        std::nullopt,
        anchor,
        std::nullopt,
        std::nullopt));
    if (!anchor.iteration.converged) {
        out.beta = Eigen::VectorXd::Constant(anchor.beta.size(), std::numeric_limits<double>::quiet_NaN());
        out.has_linear = false;
        out.tau_fit = 2.0;
        out.converged = false;
        out.iteration = anchor.iteration;
        out.path_status = "anchor_failed";
        out.tau_failed = 2.0;
        out.has_tau_failed = true;
        out.failure_reason =
            anchor.iteration.failure_reason.empty() ? "tau2_anchor_failed"
                                                    : anchor.iteration.failure_reason;
        return out;
    }
    if (!anchor.iteration.has_final_rho) {
        out.beta = Eigen::VectorXd::Constant(anchor.beta.size(), std::numeric_limits<double>::quiet_NaN());
        out.has_linear = false;
        out.tau_fit = 2.0;
        out.converged = false;
        out.iteration = anchor.iteration;
        out.iteration.converged = false;
        out.iteration.failure_reason = "tau2_anchor_missing_rho";
        out.path_status = "anchor_failed";
        out.tau_failed = 2.0;
        out.has_tau_failed = true;
        out.failure_reason = "tau2_anchor_missing_rho";
        return out;
    }
    if (std::abs(tau_target - 2.0) <= 1e-10) {
        out.beta = anchor.beta;
        out.linear = anchor.linear;
        out.has_linear = anchor.has_linear;
        out.tau_fit = 2.0;
        out.converged = true;
        out.iteration = anchor.iteration;
        out.path_status = "target_reached";
        out.tau_attained = 2.0;
        out.has_final_rho = true;
        out.final_rho = anchor.iteration.final_rho;
        return out;
    }

    const std::vector<double> grid =
        tau_path_grid_core(tau_target, control.tau_path_steps);
    const double sign = tau_target > 2.0 ? 1.0 : -1.0;
    const double delta = std::abs(tau_target - 2.0);
    const double min_dtau =
        control.has_min_dtau && control.min_dtau > 0.0
            ? control.min_dtau
            : std::max(delta / std::max(static_cast<double>(control.tau_path_steps) * 128.0, 1.0), 1e-3);
    const double path_tol = std::max(min_dtau * 0.1, 1e-12);
    const bool use_local_frontier_refinement = sign < 0.0;
    Eigen::VectorXd beta_current = anchor.beta;
    Eigen::VectorXd beta_previous = anchor.beta;
    Eigen::VectorXd linear_current = anchor.linear;
    double rho_previous = anchor.iteration.final_rho;
    double tau_current = 2.0;
    std::string failure_reason;
    std::optional<double> failed_tau;

    for (const double tau_goal : grid) {
        if (sign * (tau_goal - tau_current) <= path_tol) {
            continue;
        }
        std::optional<double> frontier_step_cap;
        bool frontier_locked = false;
        int frontier_expensive_failures = 0;
        while (sign * (tau_goal - tau_current) > path_tol) {
            const double tau_try = tau_path_next_try_core(
                tau_goal, tau_current, sign, frontier_step_cap);
            if (out.n_path_evaluations >= control.max_path_evaluations) {
                failure_reason = "max_path_evaluations_reached";
                failed_tau = tau_try;
                break;
            }
            const double tau_from = tau_current;
            out.n_path_evaluations += 1;
            FullrecoverConcentratedCore step = concentrated_fullcancel_core(
                endog,
                exog,
                bundle,
                tau_try,
                path_control,
                beta_previous,
                rho_previous,
                &phi_workspace);
            diagnostics.push_back(step.iteration);
            out.path.push_back(make_path_record_core(
                tau_try,
                tau_current,
                std::abs(tau_try - tau_current),
                step,
                tau_current,
                rho_previous));
            if (step.iteration.converged) {
                out.n_path_successes += 1;
                tau_current = tau_try;
                beta_current = step.beta;
                beta_previous = step.beta;
                linear_current = step.linear;
                if (step.iteration.has_final_rho) {
                    rho_previous = step.iteration.final_rho;
                }
                if (use_local_frontier_refinement && frontier_locked) {
                    const double success_gap = std::abs(tau_current - tau_from);
                    frontier_step_cap = std::max(
                        min_dtau,
                        std::min(
                            success_gap,
                            frontier_step_cap.has_value() ? *frontier_step_cap
                                                          : success_gap));
                } else {
                    frontier_step_cap.reset();
                }
                if (sign * (tau_goal - tau_current) <= path_tol) {
                    break;
                }
                continue;
            }
            out.n_path_failures += 1;
            const double gap = std::abs(tau_try - tau_current);
            if (gap <= min_dtau * (1.0 + 1e-12)) {
                failure_reason = step.iteration.failure_reason.empty()
                                     ? "min_dtau_reached"
                                     : step.iteration.failure_reason;
                failed_tau = tau_try;
                break;
            }
            if (use_local_frontier_refinement &&
                tau_path_expensive_failed_step_core(step.iteration, control)) {
                frontier_locked = true;
                frontier_expensive_failures += 1;
                if (frontier_expensive_failures > kTauPathFrontierRetryBudget) {
                    failure_reason = step.iteration.failure_reason.empty()
                                         ? "not_converged"
                                         : step.iteration.failure_reason;
                    failed_tau = tau_try;
                    break;
                }
            } else if (tau_path_expensive_failed_step_core(
                           step.iteration, control)) {
                failure_reason = step.iteration.failure_reason.empty()
                                     ? "not_converged"
                                     : step.iteration.failure_reason;
                failed_tau = tau_try;
                break;
            }
            frontier_step_cap = tau_path_refined_step_cap_core(gap, min_dtau);
        }
        if (!failure_reason.empty()) {
            break;
        }
    }
    const bool target_reached = failure_reason.empty();
    out.beta = beta_current;
    out.linear = linear_current;
    out.has_linear = all_finite_vec(linear_current);
    out.tau_fit = tau_current;
    out.converged = target_reached;
    out.iteration =
        combine_tau_path_iteration_core(target_reached, failure_reason, diagnostics);
    out.path_status =
        target_reached ? "target_reached" : "partial_target_not_reached";
    out.tau_attained = tau_current;
    if (failed_tau.has_value()) {
        out.has_tau_failed = true;
        out.tau_failed = *failed_tau;
    }
    out.has_tau_gap = true;
    out.tau_gap = std::abs(tau_target - tau_current);
    out.failure_reason = failure_reason;
    out.has_final_rho = true;
    out.final_rho = rho_previous;
    out.has_min_dtau = true;
    out.min_dtau = min_dtau;
    return out;
}

FullrecoverTauPathCore solve_fullcancel_tau_path_mixed_compressed_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const MixedCompressedDesignCore& design,
    const std::vector<std::int64_t>& cancel_groups,
    std::int64_t n_cancel_groups,
    double tau_target,
    const FullrecoverControlCore& control) {
    FullrecoverTauPathCore out;
    out.tau_target = tau_target;
    FullrecoverControlCore path_control = control;
    if (tau_target > 2.0 + 1e-10 && path_control.anderson_m != 0) {
        path_control.anderson_m = 0;
    }
    const FullrecoverConcentratedCore anchor =
        concentrated_fullcancel_mixed_compressed_core(
            endog,
            design,
            cancel_groups,
            n_cancel_groups,
            2.0,
            path_control,
            std::nullopt,
            std::nullopt,
            std::abs(tau_target - 2.0) <= 1.0e-10);
    std::vector<FullrecoverIterationCore> diagnostics{anchor.iteration};
    out.path.push_back(make_path_record_core(
        2.0,
        std::nullopt,
        std::nullopt,
        anchor,
        std::nullopt,
        std::nullopt));
    if (!anchor.iteration.converged) {
        out.beta = Eigen::VectorXd::Constant(
            anchor.beta.size(), std::numeric_limits<double>::quiet_NaN());
        out.has_linear = false;
        out.tau_fit = 2.0;
        out.converged = false;
        out.iteration = anchor.iteration;
        out.path_status = "anchor_failed";
        out.tau_failed = 2.0;
        out.has_tau_failed = true;
        out.failure_reason =
            anchor.iteration.failure_reason.empty() ? "tau2_anchor_failed"
                                                    : anchor.iteration.failure_reason;
        return out;
    }
    if (!anchor.iteration.has_final_rho) {
        out.beta = Eigen::VectorXd::Constant(
            anchor.beta.size(), std::numeric_limits<double>::quiet_NaN());
        out.has_linear = false;
        out.tau_fit = 2.0;
        out.converged = false;
        out.iteration = anchor.iteration;
        out.iteration.converged = false;
        out.iteration.failure_reason = "tau2_anchor_missing_rho";
        out.path_status = "anchor_failed";
        out.tau_failed = 2.0;
        out.has_tau_failed = true;
        out.failure_reason = "tau2_anchor_missing_rho";
        return out;
    }
    if (std::abs(tau_target - 2.0) <= 1e-10) {
        out.beta = anchor.beta;
        out.linear = anchor.linear;
        out.has_linear = anchor.has_linear;
        out.tau_fit = 2.0;
        out.converged = true;
        out.iteration = anchor.iteration;
        out.path_status = "target_reached";
        out.tau_attained = 2.0;
        out.has_final_rho = true;
        out.final_rho = anchor.iteration.final_rho;
        return out;
    }

    const std::vector<double> grid =
        tau_path_grid_core(tau_target, control.tau_path_steps);
    const double sign = tau_target > 2.0 ? 1.0 : -1.0;
    const double delta = std::abs(tau_target - 2.0);
    const double min_dtau =
        control.has_min_dtau && control.min_dtau > 0.0
            ? control.min_dtau
            : std::max(delta / std::max(static_cast<double>(control.tau_path_steps) * 128.0, 1.0), 1e-3);
    const double path_tol = std::max(min_dtau * 0.1, 1e-12);
    Eigen::VectorXd beta_current = anchor.beta;
    Eigen::VectorXd beta_previous = anchor.beta;
    Eigen::VectorXd linear_current = anchor.linear;
    double rho_previous = anchor.iteration.final_rho;
    double tau_current = 2.0;
    std::string failure_reason;
    std::optional<double> failed_tau;

    for (const double tau_goal : grid) {
        if (sign * (tau_goal - tau_current) <= path_tol) {
            continue;
        }
        double tau_try = tau_goal;
        while (sign * (tau_goal - tau_current) > path_tol) {
            if (out.n_path_evaluations >= control.max_path_evaluations) {
                failure_reason = "max_path_evaluations_reached";
                failed_tau = tau_try;
                break;
            }
            out.n_path_evaluations += 1;
            FullrecoverConcentratedCore step =
                concentrated_fullcancel_mixed_compressed_core(
                    endog,
                    design,
                    cancel_groups,
                    n_cancel_groups,
                    tau_try,
                    path_control,
                    beta_previous,
                    rho_previous);
            diagnostics.push_back(step.iteration);
            out.path.push_back(make_path_record_core(
                tau_try,
                tau_current,
                std::abs(tau_try - tau_current),
                step,
                tau_current,
                rho_previous));
            if (step.iteration.converged) {
                out.n_path_successes += 1;
                tau_current = tau_try;
                beta_current = step.beta;
                beta_previous = step.beta;
                linear_current = step.linear;
                if (step.iteration.has_final_rho) {
                    rho_previous = step.iteration.final_rho;
                }
                if (sign * (tau_goal - tau_current) <= path_tol) {
                    break;
                }
                tau_try = tau_goal;
                continue;
            }
            out.n_path_failures += 1;
            const double gap = std::abs(tau_try - tau_current);
            if (tau_path_expensive_failed_step_core(step.iteration, control)) {
                failure_reason = step.iteration.failure_reason.empty()
                                     ? "not_converged"
                                     : step.iteration.failure_reason;
                failed_tau = tau_try;
                break;
            }
            if (gap <= min_dtau * (1.0 + 1e-12)) {
                failure_reason = step.iteration.failure_reason.empty()
                                     ? "min_dtau_reached"
                                     : step.iteration.failure_reason;
                failed_tau = tau_try;
                break;
            }
            tau_try = tau_current + sign * gap / 2.0;
        }
        if (!failure_reason.empty()) {
            break;
        }
    }
    const bool target_reached = failure_reason.empty();
    out.beta = beta_current;
    out.linear = linear_current;
    out.has_linear = all_finite_vec(linear_current);
    out.tau_fit = tau_current;
    out.converged = target_reached;
    out.iteration =
        combine_tau_path_iteration_core(target_reached, failure_reason, diagnostics);
    out.path_status =
        target_reached ? "target_reached" : "partial_target_not_reached";
    out.tau_attained = tau_current;
    if (failed_tau.has_value()) {
        out.has_tau_failed = true;
        out.tau_failed = *failed_tau;
    }
    out.has_tau_gap = true;
    out.tau_gap = std::abs(tau_target - tau_current);
    out.failure_reason = failure_reason;
    out.has_final_rho = true;
    out.final_rho = rho_previous;
    out.has_min_dtau = true;
    out.min_dtau = min_dtau;
    return out;
}

FullrecoverControlCore control_from_mapping(py::object control_obj) {
    FullrecoverControlCore control;
    if (control_obj.is_none()) {
        apply_cpp_speed_research_env_core(control);
        return control;
    }
    const py::dict mapping = py::cast<py::dict>(control_obj);
    auto get_int = [&mapping](const char* key, int current) {
        if (mapping.contains(py::str(key))) {
            return py::cast<int>(mapping[py::str(key)]);
        }
        return current;
    };
    auto get_double = [&mapping](const char* key, double current) {
        if (mapping.contains(py::str(key))) {
            return py::cast<double>(mapping[py::str(key)]);
        }
        return current;
    };
    control.anderson_m = get_int("anderson_m", control.anderson_m);
    control.eps = get_double("eps", control.eps);
    control.stopcrit = get_int("stopcrit", control.stopcrit);
    control.tau_path_steps = get_int("tau_path_steps", control.tau_path_steps);
    if (mapping.contains(py::str("min_dtau")) && !mapping[py::str("min_dtau")].is_none()) {
        control.has_min_dtau = true;
        control.min_dtau = py::cast<double>(mapping[py::str("min_dtau")]);
    }
    control.max_path_evaluations =
        get_int("max_path_evaluations", control.max_path_evaluations);
    control.max_phase1_iterations =
        get_int("max_phase1_iterations", control.max_phase1_iterations);
    control.max_phase2_evaluations =
        get_int("max_phase2_evaluations", control.max_phase2_evaluations);
    control.fe_demean_maxiter =
        get_int("fe_demean_maxiter", control.fe_demean_maxiter);
    control.fe_demean_tol = get_double("fe_demean_tol", control.fe_demean_tol);
    control.delta_log_min = get_double("delta_log_min", control.delta_log_min);
    control.delta_log_max = get_double("delta_log_max", control.delta_log_max);
    control.delta_grid_size = get_int("delta_grid_size", control.delta_grid_size);
    control.rho_log_min = get_double("rho_log_min", control.rho_log_min);
    control.rho_log_max = get_double("rho_log_max", control.rho_log_max);
    control.rho_grid_size = get_int("rho_grid_size", control.rho_grid_size);
    if (mapping.contains(py::str("direct_restart_start")) &&
        !mapping[py::str("direct_restart_start")].is_none()) {
        // runtime_v3: read only by the direct no-FE fixed-tau route.
        control.direct_restart_start =
            py::cast<std::string>(mapping[py::str("direct_restart_start")]);
    }
    apply_cpp_speed_research_env_core(control);
    return control;
}

py::dict iteration_to_dict(const FullrecoverIterationCore& iteration) {
    py::dict out;
    out["converged"] = iteration.converged;
    out["phase"] = iteration.phase;
    out["iterations"] = iteration.iterations;
    out["restarts"] = iteration.restarts;
    out["final_rho"] =
        iteration.has_final_rho ? py::cast(iteration.final_rho) : py::none();
    py::list rho_path;
    for (const double value : iteration.rho_path) {
        rho_path.append(value);
    }
    out["rho_path"] = rho_path;
    out["final_criterion"] = iteration.has_final_criterion
                                 ? py::cast(iteration.final_criterion)
                                 : py::none();
    out["anderson_m_initial"] = iteration.anderson_m_initial;
    out["anderson_m_final"] = iteration.anderson_m_final;
    out["failure_reason"] =
        iteration.failure_reason.empty() ? py::none() : py::cast(iteration.failure_reason);
    py::dict details;
    details["phase1_attempts"] = iteration.phase1_attempts;
    details["phase1_accepted"] = iteration.phase1_accepted;
    details["phase1_rejected"] = iteration.phase1_rejected;
    details["phase2_evaluations"] = iteration.phase2_evaluations;
    details["max_phase2_evaluations"] = iteration.max_phase2_evaluations;
    details["joint_hdfe_phi_calls"] = iteration.joint_hdfe_phi_calls;
    details["joint_hdfe_phi_demean_sweeps"] =
        iteration.joint_hdfe_phi_demean_sweeps;
    details["research_tiered_fe_tol"] = iteration.research_tiered_fe_tol;
    if (std::isfinite(iteration.research_inner_fe_demean_tol)) {
        details["research_inner_fe_demean_tol"] =
            iteration.research_inner_fe_demean_tol;
    }
    details["phase2_rho_restart_events"] = iteration.phase2_rho_restart_events;
    details["phase2_rho_attempts"] = iteration.phase2_rho_attempts;
    details["last_finite_criterion"] = iteration.last_finite_criterion;
    details["best_finite_criterion"] = iteration.best_finite_criterion;
    if (!iteration.rho_policy.empty()) {
        details["rho_policy"] = iteration.rho_policy;
        details["fullcancel_initial_scale"] = iteration.fullcancel_initial_scale;
        details["rho_candidate_count"] = iteration.rho_candidate_count;
        details["rho_hint"] =
            iteration.has_rho_hint ? py::cast(iteration.rho_hint) : py::none();
        details["rho_log_min"] = iteration.rho_log_min;
        details["rho_log_max"] = iteration.rho_log_max;
        details["rho_grid_size"] = iteration.rho_grid_size;
        if (iteration.has_rho_min) {
            details["rho_min"] = iteration.rho_min;
        }
        if (iteration.has_rho_max) {
            details["rho_max"] = iteration.rho_max;
        }
    }
    if (!iteration.algorithm.empty()) {
        details["fullcancel_algorithm"] = iteration.algorithm;
    }
    if (!iteration.index_policy.empty()) {
        details["fullcancel_index_policy"] = iteration.index_policy;
    }
    if (!iteration.fe_policy.empty()) {
        details["fullcancel_fe_policy"] = iteration.fe_policy;
    }
    if (!iteration.fe_state_policy.empty()) {
        details["fe_state_policy"] = iteration.fe_state_policy;
    }
    if (!iteration.anderson_policy.empty()) {
        details["anderson_policy"] = iteration.anderson_policy;
    }
    details["qhat_last"] = iteration.qhat_last;
    details["qhat_invalid_count"] = iteration.qhat_invalid_count;
    details["kappa_monitor_last"] = iteration.kappa_monitor_last;
    details["kappa_monitor_max"] = iteration.kappa_monitor_max;
    details["kappa_monitor_checks"] = iteration.kappa_monitor_checks;
    details["post_convergence_residual"] = iteration.post_convergence_residual;
    details["post_convergence_kappa_est"] = iteration.post_convergence_kappa_est;
    details["post_convergence_score_residual"] =
        iteration.post_convergence_score_residual;
    details["post_convergence_fe_score_residual"] =
        iteration.post_convergence_fe_score_residual;
    details["score_tolerance"] = iteration.score_tolerance;
    details["score_validation_policy"] = iteration.score_validation_policy;
    details["score_validation_status"] = iteration.score_validation_status;
    py::list score_codes;
    for (const std::string& code : iteration.score_warning_codes) {
        score_codes.append(code);
    }
    py::list score_messages;
    for (const std::string& message : iteration.score_warning_messages) {
        score_messages.append(message);
    }
    details["score_warning_codes"] = py::tuple(score_codes);
    details["score_warning_messages"] = py::tuple(score_messages);
    details["post_convergence_score_exceeds_tolerance"] =
        iteration.post_convergence_score_exceeds_tolerance;
    details["post_convergence_fe_score_exceeds_tolerance"] =
        iteration.post_convergence_fe_score_exceeds_tolerance;
    if (!iteration.fullcancel_rho_attempt_values.empty()) {
        py::list rho_attempt_trace;
        const std::size_t n_attempts =
            iteration.fullcancel_rho_attempt_values.size();
        for (std::size_t idx = 0; idx < n_attempts; ++idx) {
            py::dict attempt;
            attempt["rho"] = iteration.fullcancel_rho_attempt_values[idx];
            attempt["iterations"] =
                iteration.fullcancel_rho_attempt_iterations[idx];
            attempt["restarts"] =
                iteration.fullcancel_rho_attempt_restarts[idx];
            attempt["phase2_evaluations"] =
                iteration.fullcancel_rho_attempt_phase2_evaluations[idx];
            const double final_criterion =
                iteration.fullcancel_rho_attempt_final_criteria[idx];
            attempt["final_criterion"] =
                std::isfinite(final_criterion) ? py::cast(final_criterion)
                                               : py::none();
            const double post_residual =
                iteration.fullcancel_rho_attempt_post_residuals[idx];
            attempt["post_convergence_residual"] =
                std::isfinite(post_residual) ? py::cast(post_residual)
                                             : py::none();
            const double post_score =
                iteration.fullcancel_rho_attempt_post_score_residuals[idx];
            attempt["post_convergence_score_residual"] =
                std::isfinite(post_score) ? py::cast(post_score) : py::none();
            attempt["failure_reason"] =
                iteration.fullcancel_rho_attempt_failure_reasons[idx].empty()
                    ? py::none()
                    : py::cast(
                          iteration.fullcancel_rho_attempt_failure_reasons[idx]);
            attempt["converged"] =
                iteration.fullcancel_rho_attempt_converged[idx];
            rho_attempt_trace.append(attempt);
        }
        details["fullcancel_rho_attempt_trace"] = rho_attempt_trace;
    }
    out["details"] = details;
    return out;
}

py::dict path_record_to_dict(const TauPathRecordCore& record) {
    py::dict out;
    out["tau"] = record.tau;
    out["from_tau"] = record.has_from_tau ? py::cast(record.from_tau) : py::none();
    if (record.has_step) {
        out["step"] = record.step;
    }
    out["converged"] = record.converged;
    if (record.has_beta) {
        out["beta_reduced"] = vector_to_array(record.beta);
    } else {
        out["beta_reduced"] = py::none();
    }
    out["rho"] = record.has_rho ? py::cast(record.rho) : py::none();
    out["iterations"] = record.iterations;
    out["restarts"] = record.restarts;
    out["anderson_m_initial"] = record.anderson_m_initial;
    out["anderson_m_final"] = record.anderson_m_final;
    out["failure_reason"] =
        record.failure_reason.empty() ? py::none() : py::cast(record.failure_reason);
    out["beta_init_source_tau"] = record.has_beta_init_source_tau
                                      ? py::cast(record.beta_init_source_tau)
                                      : py::none();
    out["rho_hint"] = record.has_rho_hint ? py::cast(record.rho_hint) : py::none();
    if (record.has_linear) {
        out["_linear_predictor"] = vector_to_array(record.linear);
    } else {
        out["_linear_predictor"] = py::none();
    }
    out["diagnostics"] = iteration_to_dict(record.diagnostics)["details"];
    return out;
}

py::dict tau_path_to_dict(
    const FullrecoverTauPathCore& result,
    bool compact_path = false) {
    py::dict out;
    out["kernel_backend"] = "cpp";
    out["beta"] = vector_to_array(result.beta);
    if (result.has_linear) {
        out["linear"] = vector_to_array(result.linear);
    } else {
        out["linear"] = py::none();
    }
    out["tau_fit"] = result.tau_fit;
    out["converged"] = result.converged;
    out["iteration"] = iteration_to_dict(result.iteration);
    py::dict details;
    details["path_status"] = result.path_status;
    details["tau_anchor"] = result.tau_anchor;
    details["tau_target"] = result.tau_target;
    details["tau_attained"] = result.tau_attained;
    details["tau_failed"] =
        result.has_tau_failed ? py::cast(result.tau_failed) : py::none();
    if (result.has_tau_gap) {
        details["tau_gap"] = result.tau_gap;
    }
    details["failure_reason"] =
        result.failure_reason.empty() ? py::none() : py::cast(result.failure_reason);
    details["final_rho"] =
        result.has_final_rho ? py::cast(result.final_rho) : py::none();
    if (result.has_min_dtau) {
        details["min_dtau"] = result.min_dtau;
    }
    details["n_path_evaluations"] = result.n_path_evaluations;
    details["n_path_successes"] = result.n_path_successes;
    details["n_path_failures"] = result.n_path_failures;
    details["speed_accept_score_warning_path_steps"] =
        result.speed_accept_score_warning_path_steps;
    details["tau_grid_policy"] = "quadratic_endpoint_dense_with_adaptive_bisection";
    details["score_validation_policy"] = "diagnostic";
    if (compact_path) {
        details["path"] = py::tuple();
    } else {
        py::list records;
        for (const TauPathRecordCore& record : result.path) {
            records.append(path_record_to_dict(record));
        }
        details["path"] = py::tuple(records);
    }
    out["details"] = details;
    return out;
}

py::dict fullcancel_tau_path_to_dict(
    const FullrecoverTauPathCore& result,
    bool compact_path = false) {
    py::dict out = tau_path_to_dict(result, compact_path);
    py::dict details = py::cast<py::dict>(out["details"]);
    details["fullcancel_algorithm"] = "weighted_within_fullcancel";
    details["fullcancel_index_policy"] = "slope_only_raw_exog";
    details["fullcancel_fe_policy"] = "weighted_within_cancel_all";
    out["details"] = details;
    return out;
}

GroupBundleCore group_bundle_from_objects(
    py::object groups_obj,
    py::object n_groups_obj,
    Eigen::Index expected_rows,
    const char* empty_group_message = "fullcancel requires at least one group") {
    if (groups_obj.is_none() || n_groups_obj.is_none()) {
        throw std::invalid_argument("groups and n_groups are required");
    }
    py::sequence groups_seq = py::cast<py::sequence>(groups_obj);
    py::sequence counts_seq = py::cast<py::sequence>(n_groups_obj);
    if (py::len(groups_seq) != py::len(counts_seq)) {
        throw std::invalid_argument("groups and n_groups must have the same length");
    }
    if (py::len(groups_seq) == 0) {
        throw std::invalid_argument(empty_group_message);
    }
    GroupBundleCore bundle;
    const py::ssize_t n_items = py::len(groups_seq);
    bundle.groups.reserve(static_cast<std::size_t>(n_items));
    bundle.counts.reserve(static_cast<std::size_t>(n_items));
    bundle.unweighted_inverse_counts.reserve(static_cast<std::size_t>(n_items));
    for (py::ssize_t idx = 0; idx < n_items; ++idx) {
        const std::int64_t count = py::cast<std::int64_t>(counts_seq[idx]);
        std::vector<std::int64_t> groups = groups_from_array(
            py::cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>(
                groups_seq[idx]),
            expected_rows,
            count);
        Eigen::VectorXd inverse_counts =
            Eigen::VectorXd::Zero(static_cast<Eigen::Index>(count));
        for (const std::int64_t group : groups) {
            inverse_counts(static_cast<Eigen::Index>(group)) += 1.0;
        }
        for (Eigen::Index group = 0; group < inverse_counts.size(); ++group) {
            inverse_counts(group) =
                inverse_counts(group) > 0.0 ? 1.0 / inverse_counts(group) : 0.0;
        }
        bundle.counts.push_back(count);
        bundle.groups.push_back(std::move(groups));
        bundle.unweighted_inverse_counts.push_back(std::move(inverse_counts));
    }
    return bundle;
}

std::vector<std::string> string_vector_from_sequence(
    py::object values_obj,
    std::size_t expected_size,
    const char* name) {
    if (values_obj.is_none()) {
        throw std::invalid_argument(std::string(name) + " is required");
    }
    py::sequence values_seq = py::cast<py::sequence>(values_obj);
    if (static_cast<std::size_t>(py::len(values_seq)) != expected_size) {
        throw std::invalid_argument(
            std::string(name) + " must have the same length as recover_groups");
    }
    std::vector<std::string> values;
    values.reserve(expected_size);
    for (py::ssize_t idx = 0; idx < py::len(values_seq); ++idx) {
        values.push_back(py::cast<std::string>(values_seq[idx]));
    }
    return values;
}

py::tuple string_tuple_from_vector(const std::vector<std::string>& values) {
    py::tuple out(values.size());
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        out[idx] = values[idx];
    }
    return out;
}

py::object demean_manyway_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::object groups_obj,
    py::object counts_obj,
    py::object weights_obj,
    double tol,
    int maxiter) {
    bool was_vector = false;
    const Eigen::MatrixXd values_mat = matrix_from_values(values, was_vector);
    const GroupBundleCore bundle = group_bundle_from_objects(
        groups_obj,
        counts_obj,
        values_mat.rows());
    std::optional<Eigen::VectorXd> weights;
    if (!weights_obj.is_none()) {
        weights = vector_from_array(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                weights_obj),
            "weights");
        if (weights->size() != values_mat.rows()) {
            throw std::invalid_argument("weights and values must have the same rows");
        }
    }
    Eigen::MatrixXd result;
    {
        py::gil_scoped_release release;
        DemeanWorkspaceCore workspace;
        bool converged = false;
        const Eigen::VectorXd* weights_ptr =
            weights.has_value() ? &(*weights) : nullptr;
        demean_for_group_bundle_into_core(
            values_mat,
            bundle,
            weights_ptr,
            tol,
            maxiter,
            result,
            workspace,
            &converged);
        if (!converged) {
            throw std::runtime_error("manyway_fe_demean_not_converged");
        }
    }
    return matrix_or_vector_to_array(result, was_vector);
}

py::dict mixed_design_matrix_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    py::object recover_groups_obj,
    py::object recover_counts_obj,
    py::object recover_names_obj) {
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    const GroupBundleCore recover_bundle = group_bundle_from_objects(
        recover_groups_obj,
        recover_counts_obj,
        exog_mat.rows(),
        "mixed topology requires at least one recover FE block");
    const std::vector<std::string> recover_names = string_vector_from_sequence(
        recover_names_obj,
        recover_bundle.groups.size(),
        "recover_names");
    MixedDesignCore result;
    {
        py::gil_scoped_release release;
        result = mixed_design_matrix_core(exog_mat, recover_bundle, recover_names);
    }
    py::dict out;
    out["design"] = matrix_to_array(result.design);
    out["recover_column_names"] = string_tuple_from_vector(result.recover_column_names);
    out["gil_was_released"] = result.gil_was_released;
    out["input_snapshot_owned"] = result.input_snapshot_owned;
    return out;
}

py::dict mixed_support_conditioning_preflight_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> mixed_design,
    py::object cancel_groups_obj,
    py::object cancel_counts_obj,
    py::object recover_groups_obj,
    py::object recover_counts_obj,
    py::object recover_names_obj,
    py::object control_obj) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd mixed_design_mat = matrix_from_array(mixed_design, "mixed_design");
    if (endog_vec.size() != mixed_design_mat.rows()) {
        throw std::invalid_argument("endog and mixed_design must have the same rows");
    }
    const GroupBundleCore cancel_bundle = group_bundle_from_objects(
        cancel_groups_obj,
        cancel_counts_obj,
        endog_vec.size());
    const GroupBundleCore recover_bundle = group_bundle_from_objects(
        recover_groups_obj,
        recover_counts_obj,
        endog_vec.size(),
        "mixed topology requires at least one recover FE block");
    const std::vector<std::string> recover_names = string_vector_from_sequence(
        recover_names_obj,
        recover_bundle.groups.size(),
        "recover_names");
    const FullrecoverControlCore control = control_from_mapping(control_obj);
    MixedSupportPreflightCore result;
    {
        py::gil_scoped_release release;
        result = mixed_support_conditioning_preflight_core(
            endog_vec,
            mixed_design_mat,
            cancel_bundle,
            recover_bundle,
            recover_names,
            control);
    }
    py::dict out;
    out["mixed_recover_cell_count"] = result.recover_cells.total_cells;
    out["mixed_recover_min_cell_size"] = result.recover_cells.min_cell_size;
    out["mixed_recover_max_cell_size"] = result.recover_cells.max_cell_size;
    out["mixed_recover_min_positive_count"] =
        result.recover_cells.min_positive_count;
    out["mixed_recover_all_zero_cell_count"] =
        result.recover_cells.all_zero_cell_count;
    out["mixed_recover_one_positive_cell_count"] =
        result.recover_cells.one_positive_cell_count;
    out["mixed_recover_all_zero_cells"] =
        string_tuple_from_vector(result.recover_cells.all_zero_cells);
    out["mixed_recover_one_positive_cells"] =
        string_tuple_from_vector(result.recover_cells.one_positive_cells);
    out["mixed_raw_retained_design_rows"] = result.raw_rank.rows;
    out["mixed_raw_retained_design_columns"] = result.raw_rank.cols;
    out["mixed_raw_retained_design_rank"] = result.raw_rank.rank;
    out["mixed_raw_retained_design_condition_number"] =
        result.raw_rank.condition_number;
    out["mixed_raw_retained_design_min_singular_value"] =
        result.raw_rank.min_singular_value;
    out["mixed_raw_retained_design_max_singular_value"] =
        result.raw_rank.max_singular_value;
    if (!result.raw_rank.svd_status.empty()) {
        out["mixed_raw_retained_design_rank_tolerance"] =
            result.raw_rank.rank_tolerance;
        out["mixed_raw_retained_design_svd_status"] = result.raw_rank.svd_status;
    }
    if (!result.raw_rank.rank_method.empty()) {
        out["mixed_raw_retained_design_rank_method"] =
            result.raw_rank.rank_method;
    }
    out["mixed_cancel_quotient_status"] = result.cancel_quotient_status;
    if (!result.cancel_quotient_error.empty()) {
        out["mixed_cancel_quotient_error"] = result.cancel_quotient_error;
    }
    out["mixed_cancel_quotient_design"] = matrix_to_array(result.cancel_quotient);
    out["mixed_cancel_quotient_design_rows"] = result.cancel_quotient_rank.rows;
    out["mixed_cancel_quotient_design_columns"] = result.cancel_quotient_rank.cols;
    out["mixed_cancel_quotient_design_rank"] = result.cancel_quotient_rank.rank;
    out["mixed_cancel_quotient_design_condition_number"] =
        result.cancel_quotient_rank.condition_number;
    out["mixed_cancel_quotient_design_min_singular_value"] =
        result.cancel_quotient_rank.min_singular_value;
    out["mixed_cancel_quotient_design_max_singular_value"] =
        result.cancel_quotient_rank.max_singular_value;
    if (!result.cancel_quotient_rank.svd_status.empty()) {
        out["mixed_cancel_quotient_design_rank_tolerance"] =
            result.cancel_quotient_rank.rank_tolerance;
        out["mixed_cancel_quotient_design_svd_status"] =
            result.cancel_quotient_rank.svd_status;
    }
    if (!result.cancel_quotient_rank.rank_method.empty()) {
        out["mixed_cancel_quotient_design_rank_method"] =
            result.cancel_quotient_rank.rank_method;
    }
    out["gil_was_released"] = result.gil_was_released;
    out["input_snapshot_owned"] = result.input_snapshot_owned;
    return out;
}

py::dict sandwich_covariance_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> design,
    py::array_t<double, py::array::c_style | py::array::forcecast> bread_weight,
    py::array_t<double, py::array::c_style | py::array::forcecast> score,
    const std::string& se_type,
    int df_denom,
    py::object cluster_codes_obj,
    py::object n_clusters_obj) {
    const RowMatrixXd design_mat = matrix_from_array(design, "design");
    const Eigen::VectorXd bread_weight_vec =
        vector_from_array(bread_weight, "bread_weight");
    const Eigen::VectorXd score_vec = vector_from_array(score, "score");
    if (bread_weight_vec.size() != design_mat.rows() ||
        score_vec.size() != design_mat.rows()) {
        throw std::invalid_argument(
            "design, bread_weight, and score row dimensions must match");
    }
    std::optional<std::vector<std::int64_t>> cluster_codes;
    std::int64_t n_clusters = 0;
    if (!cluster_codes_obj.is_none()) {
        if (n_clusters_obj.is_none()) {
            throw std::invalid_argument(
                "n_clusters is required when cluster_codes is provided");
        }
        n_clusters = py::cast<std::int64_t>(n_clusters_obj);
        cluster_codes = groups_from_array(
            py::cast<py::array_t<
                std::int64_t,
                py::array::c_style | py::array::forcecast>>(cluster_codes_obj),
            design_mat.rows(),
            n_clusters);
    }
    SandwichCovarianceCore result;
    {
        py::gil_scoped_release release;
        result = sandwich_covariance_core(
            design_mat,
            bread_weight_vec,
            score_vec,
            se_type,
            df_denom,
            cluster_codes,
            n_clusters);
    }
    py::dict out;
    out["status"] = result.available ? "available" : "withheld";
    out["reason"] = result.available ? py::none() : py::cast(result.reason);
    out["se"] = vector_to_array(result.se);
    out["covariance"] = matrix_to_array(result.covariance);
    if (std::isfinite(result.bread_condition)) {
        out["bread_condition"] = result.bread_condition;
    } else {
        out["bread_condition"] = py::none();
    }
    if (result.n_clusters >= 0) {
        out["n_clusters"] = result.n_clusters;
    } else {
        out["n_clusters"] = py::none();
    }
    out["gil_was_released"] = result.gil_was_released;
    out["input_snapshot_owned"] = result.input_snapshot_owned;
    return out;
}

py::dict mixed_covariance_projection_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_slopes,
    py::object cancel_groups_obj,
    py::object cancel_counts_obj,
    py::object recover_groups_obj,
    py::object recover_counts_obj,
    py::array_t<double, py::array::c_style | py::array::forcecast> demean_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> bread_weight,
    py::array_t<double, py::array::c_style | py::array::forcecast> score,
    const std::string& se_type,
    int df_denom,
    py::object cluster_codes_obj,
    py::object n_clusters_obj,
    double tol,
    int maxiter) {
    const RowMatrixXd slopes_mat = matrix_from_array(x_slopes, "x_slopes");
    const GroupBundleCore cancel_bundle = group_bundle_from_objects(
        cancel_groups_obj,
        cancel_counts_obj,
        slopes_mat.rows(),
        "mixed covariance requires at least one cancel FE block");
    const GroupBundleCore recover_bundle = group_bundle_from_objects(
        recover_groups_obj,
        recover_counts_obj,
        slopes_mat.rows(),
        "mixed covariance requires at least one recover FE block");
    const Eigen::VectorXd demean_weight_vec =
        vector_from_array(demean_weights, "demean_weights");
    const Eigen::VectorXd bread_weight_vec =
        vector_from_array(bread_weight, "bread_weight");
    const Eigen::VectorXd score_vec = vector_from_array(score, "score");
    if (demean_weight_vec.size() != slopes_mat.rows() ||
        bread_weight_vec.size() != slopes_mat.rows() ||
        score_vec.size() != slopes_mat.rows()) {
        throw std::invalid_argument(
            "x_slopes, demean_weights, bread_weight, and score rows must match");
    }
    std::optional<std::vector<std::int64_t>> cluster_codes;
    std::int64_t n_clusters = 0;
    if (!cluster_codes_obj.is_none()) {
        if (n_clusters_obj.is_none()) {
            throw std::invalid_argument(
                "n_clusters is required when cluster_codes is provided");
        }
        n_clusters = py::cast<std::int64_t>(n_clusters_obj);
        cluster_codes = groups_from_array(
            py::cast<py::array_t<
                std::int64_t,
                py::array::c_style | py::array::forcecast>>(cluster_codes_obj),
            slopes_mat.rows(),
            n_clusters);
    }

    MixedCovarianceProjectionCore result;
    {
        py::gil_scoped_release release;
        result = mixed_covariance_projection_core(
            slopes_mat,
            cancel_bundle,
            recover_bundle,
            demean_weight_vec,
            bread_weight_vec,
            score_vec,
            se_type,
            df_denom,
            cluster_codes,
            n_clusters,
            tol,
            maxiter);
    }
    py::dict out;
    out["status"] = result.covariance.available ? "available" : "withheld";
    out["reason"] =
        result.covariance.available ? py::none() : py::cast(result.covariance.reason);
    out["se"] = vector_to_array(result.covariance.se);
    out["covariance"] = matrix_to_array(result.covariance.covariance);
    if (std::isfinite(result.covariance.bread_condition)) {
        out["bread_condition"] = result.covariance.bread_condition;
    } else {
        out["bread_condition"] = py::none();
    }
    if (result.covariance.n_clusters >= 0) {
        out["n_clusters"] = result.covariance.n_clusters;
    } else {
        out["n_clusters"] = py::none();
    }
    out["recover_columns"] = result.recover_columns;
    out["slope_columns"] = result.slope_columns;
    out["projection_method"] = result.projection_method;
    out["gil_was_released"] = result.gil_was_released;
    out["input_snapshot_owned"] = result.input_snapshot_owned;
    return out;
}

py::dict solve_fullrecover_tau_path_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    double tau_target,
    py::object group_obj,
    py::object n_groups_obj,
    py::object control_obj) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    if (endog_vec.size() != exog_mat.rows()) {
        throw std::invalid_argument("endog and exog must have the same rows");
    }
    std::vector<std::int64_t> groups;
    std::int64_t n_groups = 1;
    if (group_obj.is_none()) {
        groups.assign(static_cast<std::size_t>(endog_vec.size()), 0);
    } else {
        if (n_groups_obj.is_none()) {
            throw std::invalid_argument("n_groups is required when group is provided");
        }
        n_groups = py::cast<std::int64_t>(n_groups_obj);
        groups = groups_from_array(
            py::cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>(
                group_obj),
            endog_vec.size(),
            n_groups);
    }
    const FullrecoverControlCore control = control_from_mapping(control_obj);
    FullrecoverTauPathCore result;
    {
        py::gil_scoped_release release;
        result = solve_fullrecover_tau_path_core(
            endog_vec,
            exog_mat,
            groups,
            n_groups,
            tau_target,
            control);
    }
    return tau_path_to_dict(result, control.speed_compact_fixed_tau_details);
}

py::dict solve_nofe_tau_path_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    double tau_target,
    py::object control_obj) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    if (endog_vec.size() != exog_mat.rows()) {
        throw std::invalid_argument("endog and exog must have the same rows");
    }
    std::vector<std::int64_t> groups(
        static_cast<std::size_t>(endog_vec.size()), 0);
    const FullrecoverControlCore control = control_from_mapping(control_obj);
    FullrecoverTauPathCore result;
    {
        py::gil_scoped_release release;
        result = solve_fullrecover_tau_path_core(
            endog_vec,
            exog_mat,
            groups,
            1,
            tau_target,
            control);
    }
    py::dict out = tau_path_to_dict(
        result,
        control.speed_compact_fixed_tau_details);
    py::dict details = py::cast<py::dict>(out["details"]);
    details["nofe_algorithm"] = "singleton_intercept_profiled_iols";
    details["nofe_implicit_intercept_policy"] = "profiled_singleton_intercept";
    out["details"] = details;
    return out;
}

// Research-only direct fixed-tau noFE route, twin of the NumPy
// _solve_nofe_direct_fixed_tau. Messages and check order match the NumPy
// validators so both backends reject the same inputs the same way.
void validate_direct_nofe_inputs_core(
    const Eigen::Ref<const Eigen::VectorXd>& endog,
    const Eigen::Ref<const RowMatrixXd>& exog_slopes,
    double tau) {
    if (exog_slopes.rows() != endog.size()) {
        throw std::invalid_argument(
            "endog and exog_slopes must have the same number of rows");
    }
    if (exog_slopes.cols() < 1) {
        throw std::invalid_argument(
            "exog_slopes must have at least one slope column; the intercept is "
            "profiled");
    }
    bool any_positive = false;
    for (Eigen::Index row = 0; row < endog.size(); ++row) {
        if (!std::isfinite(endog(row)) || endog(row) < 0.0) {
            throw std::invalid_argument("endog must be finite and nonnegative");
        }
        any_positive = any_positive || endog(row) > 0.0;
    }
    if (!any_positive) {
        throw std::invalid_argument("endog must contain at least one positive value");
    }
    for (Eigen::Index row = 0; row < exog_slopes.rows(); ++row) {
        for (Eigen::Index col = 0; col < exog_slopes.cols(); ++col) {
            if (!std::isfinite(exog_slopes(row, col))) {
                throw std::invalid_argument("exog_slopes must be finite");
            }
        }
    }
    for (Eigen::Index col = 0; col < exog_slopes.cols(); ++col) {
        if (exog_slopes.col(col).maxCoeff() == exog_slopes.col(col).minCoeff()) {
            throw std::invalid_argument(
                "exog_slopes column " + std::to_string(col) +
                " is constant; pass slopes only, the intercept is profiled");
        }
    }
    if (!std::isfinite(tau) || tau < 0.0 || tau > 3.0) {
        throw std::invalid_argument("tau must be finite and in [0, 3]");
    }
}

void validate_direct_nofe_control_core(const FullrecoverControlCore& control) {
    if (control.delta_grid_size < 1) {
        throw std::invalid_argument(
            "direct fixed-tau route requires delta_grid_size >= 1 so Phase 1 runs");
    }
    if (control.anderson_m < 0) {
        throw std::invalid_argument("anderson_m must be nonnegative");
    }
    if (!std::isfinite(control.eps) || control.eps <= 0.0) {
        throw std::invalid_argument("eps must be finite and positive");
    }
    if (control.stopcrit < 2 || control.max_phase1_iterations < 2) {
        throw std::invalid_argument(
            "stopcrit and max_phase1_iterations must be at least 2");
    }
    if (control.max_phase2_evaluations < 1) {
        throw std::invalid_argument("max_phase2_evaluations must be at least 1");
    }
    if (control.direct_restart_start != kDirectRestartStartPhase1 &&
        control.direct_restart_start != kDirectRestartStartBestScore) {
        throw std::invalid_argument(
            "direct_restart_start must be one of ('phase1_start', "
            "'best_score_iterate')");
    }
}

// Runs the existing profiled Phase 1 (delta grid) and Phase 2 (rho grid) once
// at the requested tau: no tau=2 anchor and no tau continuation. Validation,
// the fixed-design invariant state, both phases and the profiled index all run
// in C++ with the GIL released.
py::dict solve_nofe_direct_fixed_tau_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog_slopes,
    double tau,
    py::object control_obj,
    const std::string& rho_grid_policy) {
    // rho_grid_hook_v1: validate the opt-in policy before any work.
    validate_direct_rho_grid_policy_core(rho_grid_policy);
    if (control_obj.is_none()) {
        throw std::invalid_argument(
            "direct fixed-tau route requires an explicit control mapping");
    }
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog_slopes, "exog_slopes");
    FullrecoverControlCore control = control_from_mapping(control_obj);
    // Env-derived research switches are not read by the profiled core; reset
    // them so this route never depends on the process environment.
    control.research_tiered_fe_tol = false;
    control.speed_fullrecover_joint_anderson = false;
    control.speed_fullrecover_accept_score_warning_path_steps = false;
    control.speed_small_manyway_exact_demean = false;
    control.speed_compact_fixed_tau_details = false;
    const std::vector<std::int64_t> groups(
        static_cast<std::size_t>(endog_vec.size()), 0);
    FullrecoverConcentratedCore result;
    double intercept = std::numeric_limits<double>::quiet_NaN();
    DirectRhoGridHookCore rho_hook;
    rho_hook.policy = rho_grid_policy;
    {
        py::gil_scoped_release release;
        validate_direct_nofe_inputs_core(endog_vec, exog_mat, tau);
        validate_direct_nofe_control_core(control);
        const FullrecoverInvariantCore state =
            fullrecover_invariant_core(exog_mat, groups, 1);
        result = concentrated_fullrecover_profiled_core(
            endog_vec,
            exog_mat,
            groups,
            1,
            tau,
            control,
            state,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            /*direct_nofe_fixed_tau=*/true,
            &rho_hook);
        if (all_finite_vec(result.beta)) {
            const Eigen::VectorXd base = exog_mat * result.beta;
            intercept = profile_intercept_adjustment_core(endog_vec, base, tau);
        }
    }

    const FullrecoverIterationCore& diag = result.iteration;
    const py::object failure_reason =
        diag.failure_reason.empty() ? py::none() : py::cast(diag.failure_reason);
    py::dict iteration;
    iteration["converged"] = diag.converged;
    iteration["phase"] = diag.phase;
    iteration["iterations"] = diag.iterations;
    iteration["restarts"] = diag.restarts;
    iteration["final_rho"] = diag.has_final_rho ? py::cast(diag.final_rho) : py::none();
    py::list rho_path;
    for (const double value : diag.rho_path) {
        rho_path.append(value);
    }
    iteration["rho_path"] = py::tuple(rho_path);
    iteration["final_criterion"] =
        diag.has_final_criterion ? py::cast(diag.final_criterion) : py::none();
    iteration["anderson_m_initial"] = diag.anderson_m_initial;
    iteration["anderson_m_final"] = diag.anderson_m_final;
    iteration["failure_reason"] = failure_reason;

    py::dict diagnostics;
    diagnostics["phase1_attempts"] = diag.phase1_attempts;
    diagnostics["phase1_accepted"] = diag.phase1_accepted;
    diagnostics["phase1_rejected"] = diag.phase1_rejected;
    diagnostics["phase2_evaluations"] = diag.phase2_evaluations;
    diagnostics["last_finite_criterion"] = diag.last_finite_criterion;
    diagnostics["best_finite_criterion"] = diag.best_finite_criterion;
    diagnostics["post_convergence_residual"] = diag.post_convergence_residual;
    diagnostics["post_convergence_kappa_est"] = diag.post_convergence_kappa_est;
    diagnostics["post_convergence_score_residual"] =
        diag.post_convergence_score_residual;
    diagnostics["post_convergence_fe_score_residual"] =
        diag.post_convergence_fe_score_residual;
    diagnostics["score_tolerance"] = diag.score_tolerance;
    diagnostics["score_validation_status"] = diag.score_validation_status;
    py::list score_codes;
    for (const std::string& code : diag.score_warning_codes) {
        score_codes.append(code);
    }
    diagnostics["score_warning_codes"] = py::tuple(score_codes);

    py::dict control_out;
    control_out["anderson_m"] = control.anderson_m;
    control_out["eps"] = control.eps;
    control_out["stopcrit"] = control.stopcrit;
    control_out["max_phase1_iterations"] = control.max_phase1_iterations;
    control_out["max_phase2_evaluations"] = control.max_phase2_evaluations;
    control_out["delta_log_min"] = control.delta_log_min;
    control_out["delta_log_max"] = control.delta_log_max;
    control_out["delta_grid_size"] = control.delta_grid_size;
    control_out["rho_log_min"] = control.rho_log_min;
    control_out["rho_log_max"] = control.rho_log_max;
    control_out["rho_grid_size"] = control.rho_grid_size;
    // runtime_v3: direct_restart_start is reported in backend_details only
    // (the harness casts non-float control echoes with int()).

    py::dict backend_details;
    backend_details["runtime_version"] = kDirectNofeRuntimeVersion;
    backend_details["gil_policy"] = "released_around_validation_and_solve";
    backend_details["anderson_history_capacity_initial"] =
        control.anderson_m > 0 ? control.anderson_m + 1 : 0;
    backend_details["anderson_history_policy"] = "grows_with_adapted_m";
    backend_details["kappa_window_first_entry"] = 4;
    backend_details["qhat_check"] = "absent_in_profiled_cpp_core";
    backend_details["qhat_guard_policy"] = "absent_in_profiled_cpp_core";
    backend_details["nonconverged_iterate_policy"] =
        diag.nonconverged_iterate_policy.empty()
            ? std::string("last_attempt_final_iterate")
            : diag.nonconverged_iterate_policy;
    backend_details["phase1_evaluations"] = diag.phase1_evaluations;
    backend_details["max_phase2_evaluations"] = diag.max_phase2_evaluations;
    // runtime_v3 direct-route diagnostics (twins of the NumPy rho_details keys).
    backend_details["direct_restart_start"] = control.direct_restart_start;
    backend_details["kappa_monitor_policy"] = kDirectKappaMonitorPolicy;
    backend_details["kappa_monitor_last"] = diag.kappa_monitor_last;
    backend_details["kappa_monitor_max"] = diag.kappa_monitor_max;
    backend_details["kappa_monitor_checks"] = diag.kappa_monitor_checks;
    backend_details["kappa_coordinate_row_max"] = diag.kappa_coordinate_row_max;
    backend_details["phase2_attempts"] = diag.phase2_attempts;
    backend_details["phase2_warm_restarts"] = diag.phase2_warm_restarts;
    backend_details["best_score_norm"] = diag.best_score_norm;
    backend_details["best_score_rho"] = diag.best_score_rho;
    // runtime_v4 direct-route keys (twins of the NumPy rho_details keys).
    backend_details["progress_monitor_policy"] = kDirectProgressMonitorPolicy;
    backend_details["post_check_policy"] = kDirectPostCheckPolicy;
    backend_details["post_convergence_invalid_map_count"] =
        diag.post_convergence_invalid_map_count;
    backend_details["phase1_post_convergence_invalid_map_count"] =
        diag.phase1_post_convergence_invalid_map_count;
    py::list attempt_trace;
    for (const Phase2AttemptTraceCore& record : diag.phase2_attempt_trace) {
        py::dict entry;
        entry["attempt"] = record.attempt;
        entry["rho"] = record.rho;
        entry["iterations"] = record.iterations;
        entry["evaluations"] = record.evaluations;
        entry["failure_reason"] = record.failure_reason.empty()
                                       ? py::object(py::none())
                                       : py::object(py::cast(record.failure_reason));
        entry["final_criterion"] = record.has_final_criterion
                                        ? py::object(py::cast(record.final_criterion))
                                        : py::object(py::none());
        entry["best_criterion"] = record.best_criterion;
        entry["kappa_last"] = record.kappa_last;
        entry["start_policy"] = record.start_policy;
        entry["final_progress_residual"] = record.final_progress_residual;
        entry["best_progress_residual"] = record.best_progress_residual;
        entry["anderson_m_start"] = record.anderson_m_start;
        entry["anderson_m_end"] = record.anderson_m_end;
        entry["anderson_accepted"] = record.anderson_accepted;
        attempt_trace.append(entry);
    }
    backend_details["phase2_attempt_trace"] = py::tuple(attempt_trace);
    // rho_grid_hook_v1 diagnostics (twins of the NumPy keys; absent when the
    // Phase-2 handoff was never reached).
    if (rho_hook.reached) {
        const bool policy_grid = rho_hook.policy != kDirectRhoGridLegacy;
        backend_details["rho_grid_policy"] = rho_hook.policy;
        backend_details["rho_grid_hook_version"] = kDirectRhoGridHookVersion;
        backend_details["rho_grid_count"] = rho_hook.grid_count;
        if (policy_grid) {
            py::list grid_values;
            for (const double value : rho_hook.values) {
                grid_values.append(value);
            }
            backend_details["rho_grid_values"] = py::tuple(grid_values);
            backend_details["rho_grid_alpha0"] = rho_hook.alpha0;
            backend_details["rho_grid_alpha_min"] = rho_hook.alpha_min;
            backend_details["rho_grid_clipped"] = rho_hook.clipped;
        } else {
            backend_details["rho_grid_values"] = py::none();
            backend_details["rho_grid_alpha0"] = py::none();
            backend_details["rho_grid_alpha_min"] = py::none();
            backend_details["rho_grid_clipped"] = py::none();
        }
        if (rho_hook.has_curvature) {
            backend_details["rho_curvature_L0"] = rho_hook.curvature_L0;
            backend_details["rho_curvature_lambda_min"] = rho_hook.curvature_lambda_min;
            backend_details["rho_curvature_wall_ns"] = rho_hook.curvature_wall_ns;
            backend_details["rho_curvature_pilot"] = "phase1_accepted_beta";
        } else {
            backend_details["rho_curvature_L0"] = py::none();
            backend_details["rho_curvature_lambda_min"] = py::none();
            backend_details["rho_curvature_wall_ns"] = py::none();
            backend_details["rho_curvature_pilot"] = py::none();
        }
    }

    py::dict out;
    out["schema"] = "pyiols.direct_nofe_fixed_tau.v1";
    out["route"] = "direct_nofe_fixed_tau";
    out["backend"] = "cpp";
    out["kernel_backend"] = "cpp";
    out["algorithm"] = "iols_phase1_phase2_at_requested_tau";
    out["intercept_policy"] = "profiled_singleton_intercept";
    out["tau_requested"] = tau;
    out["tau_fit"] = tau;
    out["phase1_tau"] = tau;
    out["phase2_tau"] = tau;
    out["tau2_anchor_used"] = false;
    out["tau_continuation_used"] = false;
    out["converged"] = diag.converged;
    out["failure_reason"] = failure_reason;
    out["beta"] = vector_to_array(result.beta);
    out["intercept"] = intercept;
    out["linear"] = result.has_linear ? py::object(vector_to_array(result.linear))
                                      : py::object(py::none());
    out["iteration"] = iteration;
    out["diagnostics"] = diagnostics;
    out["control"] = control_out;
    out["backend_details"] = backend_details;
    return out;
}

py::dict solve_fullrecover_tau_path_manyway_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    double tau_target,
    py::object groups_obj,
    py::object n_groups_obj,
    py::object control_obj) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    if (endog_vec.size() != exog_mat.rows()) {
        throw std::invalid_argument("endog and exog must have the same rows");
    }
    const GroupBundleCore bundle =
        group_bundle_from_objects(groups_obj, n_groups_obj, endog_vec.size());
    if (bundle.groups.size() < 2) {
        throw std::invalid_argument(
            "many-way fullrecover requires at least two recovered FE groups");
    }
    const FullrecoverControlCore control = control_from_mapping(control_obj);
    FullrecoverTauPathCore result;
    {
        py::gil_scoped_release release;
        result = solve_fullrecover_tau_path_manyway_core(
            endog_vec,
            exog_mat,
            bundle,
            tau_target,
            control);
    }
    return tau_path_to_dict(result, control.speed_compact_fixed_tau_details);
}

py::dict solve_fullcancel_tau_path_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    double tau_target,
    py::object groups_obj,
    py::object n_groups_obj,
    py::object control_obj) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    if (endog_vec.size() != exog_mat.rows()) {
        throw std::invalid_argument("endog and exog must have the same rows");
    }
    const GroupBundleCore bundle =
        group_bundle_from_objects(groups_obj, n_groups_obj, endog_vec.size());
    const FullrecoverControlCore control = control_from_mapping(control_obj);
    FullrecoverTauPathCore result;
    {
        py::gil_scoped_release release;
        result = solve_fullcancel_tau_path_core(
            endog_vec,
            exog_mat,
            bundle,
            tau_target,
            control);
    }
    return fullcancel_tau_path_to_dict(
        result,
        control.speed_compact_fixed_tau_details);
}

py::dict solve_mixed_fullcancel_tau_path_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    double tau_target,
    py::object cancel_groups_obj,
    py::object cancel_counts_obj,
    py::object recover_groups_obj,
    py::object recover_counts_obj,
    py::object control_obj) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    if (endog_vec.size() != exog_mat.rows()) {
        throw std::invalid_argument("endog and exog must have the same rows");
    }
    const GroupBundleCore cancel_bundle =
        group_bundle_from_objects(cancel_groups_obj, cancel_counts_obj, endog_vec.size());
    const GroupBundleCore recover_bundle = group_bundle_from_objects(
        recover_groups_obj,
        recover_counts_obj,
        endog_vec.size(),
        "mixed topology requires at least one recover FE block");
    const FullrecoverControlCore control = control_from_mapping(control_obj);
    FullrecoverTauPathCore result;
    std::string compressed_status = "fallback_dense_fullcancel";
    std::string compressed_fallback_reason;
    bool compressed_retry_after_nonconvergence = false;
    std::string compressed_original_path_status;
    std::string compressed_original_failure_reason;
    double compressed_original_tau_attained = std::numeric_limits<double>::quiet_NaN();
    double compressed_original_tau_failed = std::numeric_limits<double>::quiet_NaN();
    double compressed_original_tau_gap = std::numeric_limits<double>::quiet_NaN();
    int compressed_original_path_evaluations = 0;
    int compressed_original_path_successes = 0;
    int compressed_original_path_failures = 0;
    {
        py::gil_scoped_release release;
        const MixedCompressedDesignCore compressed_design =
            mixed_compressed_design_core(exog_mat, recover_bundle);
        const bool compressed_supported =
            cancel_bundle.groups.size() == 1 && compressed_design.total_cols >= 32;
        bool run_dense_fullcancel = true;
        if (compressed_supported) {
            try {
                result = solve_fullcancel_tau_path_mixed_compressed_core(
                    endog_vec,
                    compressed_design,
                    cancel_bundle.groups.front(),
                    cancel_bundle.counts.front(),
                    tau_target,
                    control);
                if (result.converged) {
                    compressed_status = "used_mixed_compressed_fullcancel";
                    run_dense_fullcancel = false;
                } else {
                    compressed_retry_after_nonconvergence = true;
                    compressed_status =
                        "dense_retried_after_mixed_compressed_nonconvergence";
                    compressed_fallback_reason =
                        result.failure_reason.empty()
                            ? "mixed_compressed_not_converged"
                            : result.failure_reason;
                    compressed_original_path_status = result.path_status;
                    compressed_original_failure_reason = result.failure_reason;
                    compressed_original_tau_attained = result.tau_attained;
                    if (result.has_tau_failed) {
                        compressed_original_tau_failed = result.tau_failed;
                    }
                    if (result.has_tau_gap) {
                        compressed_original_tau_gap = result.tau_gap;
                    }
                    compressed_original_path_evaluations =
                        result.n_path_evaluations;
                    compressed_original_path_successes = result.n_path_successes;
                    compressed_original_path_failures = result.n_path_failures;
                }
            } catch (const CompressedMixedFallbackRequired& exc) {
                compressed_fallback_reason = exc.what();
            }
        } else {
            compressed_fallback_reason =
                cancel_bundle.groups.size() == 1
                    ? "mixed_compressed_column_threshold_not_met"
                    : "mixed_compressed_requires_one_cancel_block";
        }
        if (run_dense_fullcancel) {
            std::vector<std::string> recover_names;
            recover_names.reserve(recover_bundle.groups.size());
            for (std::size_t idx = 0; idx < recover_bundle.groups.size(); ++idx) {
                recover_names.push_back("recover" + std::to_string(idx));
            }
            const MixedDesignCore dense_design =
                mixed_design_matrix_core(exog_mat, recover_bundle, recover_names);
            result = solve_fullcancel_tau_path_core(
                endog_vec,
                dense_design.design,
                cancel_bundle,
                tau_target,
                control);
        }
    }
    py::dict out = fullcancel_tau_path_to_dict(
        result,
        control.speed_compact_fixed_tau_details);
    py::dict details = py::cast<py::dict>(out["details"]);
    details["mixed_compressed_fullcancel_status"] = compressed_status;
    if (!compressed_fallback_reason.empty()) {
        details["mixed_compressed_fullcancel_fallback_reason"] =
            compressed_fallback_reason;
    }
    if (compressed_retry_after_nonconvergence) {
        details["mixed_compressed_fullcancel_original_path_status"] =
            compressed_original_path_status;
        details["mixed_compressed_fullcancel_original_failure_reason"] =
            compressed_original_failure_reason.empty()
                ? py::none()
                : py::cast(compressed_original_failure_reason);
        details["mixed_compressed_fullcancel_original_tau_attained"] =
            compressed_original_tau_attained;
        if (std::isfinite(compressed_original_tau_failed)) {
            details["mixed_compressed_fullcancel_original_tau_failed"] =
                compressed_original_tau_failed;
        }
        if (std::isfinite(compressed_original_tau_gap)) {
            details["mixed_compressed_fullcancel_original_tau_gap"] =
                compressed_original_tau_gap;
        }
        details["mixed_compressed_fullcancel_original_path_evaluations"] =
            compressed_original_path_evaluations;
        details["mixed_compressed_fullcancel_original_path_successes"] =
            compressed_original_path_successes;
        details["mixed_compressed_fullcancel_original_path_failures"] =
            compressed_original_path_failures;
    }
    out["details"] = details;
    return out;
}

constexpr double PHASE1_LOG_FLOOR = 1.0e-300;
constexpr double PHASE1_S_REL_FLOOR = 1.0e-12;
constexpr double PHASE1_S_ABS_BACKSTOP = 1.0e-300;
constexpr double PHASE1_R_REL_FLOOR = 1.0e-10;
constexpr double PHASE1_FE_ABSORBED_REL_FLOOR = 1.0e-10;

double phase1_research_inner_within_tol_core(double within_tol) {
    if (!cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_TIERED_FE_TOL") &&
        !cpp_env_flag_enabled_core("PYIOLS_CPP_TIERED_FE_DEMEAN")) {
        return within_tol;
    }
    const double requested = cpp_env_double_or_core(
        "PYIOLS_CPP_SPEED_TIERED_FE_INNER_TOL",
        1.0e-8);
    return std::max(requested, within_tol);
}

struct Phase1NoFENormCore {
    RowMatrixXd x;
    Eigen::VectorXd scale;
    Eigen::MatrixXd x_star;
    Eigen::MatrixXd q;
    Eigen::MatrixXd r;
    double condition_after = std::numeric_limits<double>::quiet_NaN();
    std::string transformed_design_normalization = "scale";
};

struct Phase1NoFEDiagnosticCore {
    int iterations = 0;
    bool converged = false;
    double criterion_star = std::numeric_limits<double>::quiet_NaN();
    double criterion_canonical = std::numeric_limits<double>::quiet_NaN();
    std::string failure_reason;
    double rho = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> rho_path;
    std::vector<Phase1NoFEDiagnosticCore> rho_diagnostics_path;
    std::vector<Phase1NoFEDiagnosticCore> phase1_path;
    double delta = std::numeric_limits<double>::quiet_NaN();
    double score_residual_max = std::numeric_limits<double>::quiet_NaN();
    double guide_to_target_distance = std::numeric_limits<double>::quiet_NaN();
    double guide_to_target_distance_star = std::numeric_limits<double>::quiet_NaN();
    double branch_drift_relative = std::numeric_limits<double>::quiet_NaN();
    bool research_tiered_fe_tol = false;
    double research_inner_fe_demean_tol =
        std::numeric_limits<double>::quiet_NaN();
    std::string support_failure_reason;
    int support_block_index = -1;
    int support_group_index = -1;
    double support_group_sum_y = std::numeric_limits<double>::quiet_NaN();
    std::string score_residual_failure_reason;
};

struct Phase1NoFEStateCore {
    double tau = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd beta;
    Eigen::VectorXd beta_star;
    Eigen::VectorXd linear;
    Eigen::VectorXd fitted;
    std::string status;
    bool converged = false;
    Phase1NoFEDiagnosticCore diagnostics;
    double source_tau = std::numeric_limits<double>::quiet_NaN();
    double target_tau = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> kept_columns;
    std::vector<int> absorbed_columns;
    int n_groups = -1;
    std::vector<int> group_counts;
};

struct Phase1NoFEResultCore {
    bool converged = false;
    std::string failure_reason;
    Phase1NoFEStateCore anchor;
    std::vector<Phase1NoFEStateCore> guide_path;
    std::vector<Phase1NoFEStateCore> target_results;
    std::vector<double> target_taus;
    double condition_after = std::numeric_limits<double>::quiet_NaN();
    double scale_min = std::numeric_limits<double>::quiet_NaN();
    double scale_max = std::numeric_limits<double>::quiet_NaN();
    std::string transformed_design_normalization = "scale";
};

struct Phase1OneWayNormCore {
    Phase1NoFENormCore norm;
    RowMatrixXd x_raw;
    RowMatrixXd x_within;
    std::vector<int> kept_columns;
    std::vector<int> absorbed_columns;
    std::vector<std::int64_t> groups;
    std::int64_t n_groups = 0;
    std::vector<int> group_counts;
    double within_design_condition_before =
        std::numeric_limits<double>::quiet_NaN();
};

struct Phase1OneWayResultCore {
    bool converged = false;
    std::string failure_reason;
    Phase1NoFEStateCore anchor;
    std::vector<Phase1NoFEStateCore> guide_path;
    std::vector<Phase1NoFEStateCore> target_results;
    std::vector<double> target_taus;
    double condition_after = std::numeric_limits<double>::quiet_NaN();
    double scale_min = std::numeric_limits<double>::quiet_NaN();
    double scale_max = std::numeric_limits<double>::quiet_NaN();
    double within_design_condition_before =
        std::numeric_limits<double>::quiet_NaN();
    std::vector<int> kept_columns;
    std::vector<int> absorbed_columns;
    std::string transformed_design_normalization = "scale";
};

struct Phase1ManyWayNormCore {
    Phase1NoFENormCore norm;
    RowMatrixXd x_raw;
    RowMatrixXd x_within;
    std::vector<int> kept_columns;
    std::vector<int> absorbed_columns;
    GroupBundleCore bundle;
    std::vector<std::vector<int>> group_counts;
    double within_design_condition_before =
        std::numeric_limits<double>::quiet_NaN();
    double within_tol = 1.0e-10;
    int within_max_iter = 1000;
};

struct Phase1ManyWayStateCore {
    double tau = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd beta;
    Eigen::VectorXd beta_star;
    Eigen::VectorXd linear;
    Eigen::VectorXd fitted;
    std::string status;
    bool converged = false;
    Phase1NoFEDiagnosticCore diagnostics;
    double source_tau = std::numeric_limits<double>::quiet_NaN();
    double target_tau = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> kept_columns;
    std::vector<int> absorbed_columns;
    std::vector<std::int64_t> n_groups;
    std::vector<std::vector<int>> group_counts;
};

struct Phase1ManyWayResultCore {
    bool converged = false;
    std::string failure_reason;
    Phase1ManyWayStateCore anchor;
    std::vector<Phase1ManyWayStateCore> guide_path;
    std::vector<Phase1ManyWayStateCore> target_results;
    std::vector<double> target_taus;
    double condition_after = std::numeric_limits<double>::quiet_NaN();
    double scale_min = std::numeric_limits<double>::quiet_NaN();
    double scale_max = std::numeric_limits<double>::quiet_NaN();
    double within_design_condition_before =
        std::numeric_limits<double>::quiet_NaN();
    std::vector<int> kept_columns;
    std::vector<int> absorbed_columns;
    std::string transformed_design_normalization = "scale";
};

std::vector<double> double_vector_from_object(
    py::object values_obj,
    const std::string& name) {
    if (values_obj.is_none()) {
        throw std::invalid_argument(name + " must not be None");
    }
    std::vector<double> values;
    for (py::handle item : values_obj) {
        const double value = py::cast<double>(item);
        values.push_back(value);
    }
    return values;
}

std::string normalize_phase1_transformed_design_normalization_core(
    const std::string& mode) {
    if (mode == "none" || mode == "scale") {
        return mode;
    }
    throw std::invalid_argument(
        "transformed_design_normalization must be 'none' or 'scale'");
}

std::pair<std::vector<std::int64_t>, std::int64_t> compact_int64_groups_from_array(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> groups,
    Eigen::Index expected_rows) {
    const py::buffer_info info = groups.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("group must be one-dimensional");
    }
    if (info.shape[0] != expected_rows) {
        throw std::invalid_argument("group and endog must have the same rows");
    }
    const auto* ptr = static_cast<const std::int64_t*>(info.ptr);
    std::vector<std::int64_t> labels(static_cast<std::size_t>(info.shape[0]));
    for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
        labels[static_cast<std::size_t>(row)] = ptr[row];
    }
    std::vector<std::int64_t> unique = labels;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    if (unique.empty()) {
        throw std::invalid_argument("group cannot be empty");
    }
    std::vector<std::int64_t> compact(labels.size());
    for (std::size_t row = 0; row < labels.size(); ++row) {
        const auto iter =
            std::lower_bound(unique.begin(), unique.end(), labels[row]);
        compact[row] = static_cast<std::int64_t>(iter - unique.begin());
    }
    return {compact, static_cast<std::int64_t>(unique.size())};
}

std::vector<double> normalize_phase1_tau_sequence(
    const std::vector<double>& values,
    const std::string& name) {
    std::vector<double> out;
    for (const double tau : values) {
        if (!std::isfinite(tau) || tau < 0.0 || tau > 3.0) {
            throw std::invalid_argument(name + " values must be finite and in [0, 3]");
        }
        bool seen = false;
        for (const double existing : out) {
            if (std::abs(tau - existing) <= 1.0e-12) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            out.push_back(tau);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<double> phase1_grid_from_object(
    py::object values_obj,
    double log_min,
    double log_max,
    int size,
    const std::string& name) {
    std::vector<double> values;
    if (values_obj.is_none()) {
        if (size < 1) {
            throw std::invalid_argument(name + "_grid_size must be positive");
        }
        if (!std::isfinite(log_min) || !std::isfinite(log_max) ||
            log_max <= log_min) {
            throw std::invalid_argument(name + " log bounds must be increasing");
        }
        values.reserve(static_cast<std::size_t>(size));
        for (int idx = 0; idx < size; ++idx) {
            const double frac =
                size == 1 ? 0.0 : static_cast<double>(idx) / static_cast<double>(size - 1);
            values.push_back(std::exp(log_min + frac * (log_max - log_min)));
        }
    } else {
        values = double_vector_from_object(values_obj, name + "_grid");
    }
    const int min_size = name == "rho" ? 2 : 1;
    if (static_cast<int>(values.size()) < min_size) {
        throw std::invalid_argument(name + "_grid has too few values");
    }
    for (const double value : values) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(name + "_grid values must be positive finite");
        }
    }
    return values;
}

double phase1_mean_square_or_inf(const Eigen::Ref<const Eigen::VectorXd>& values) {
    long double accum = 0.0L;
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        const double value = values(idx);
        if (!std::isfinite(value)) {
            return std::numeric_limits<double>::infinity();
        }
        accum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    const long double mean = accum / static_cast<long double>(std::max<Eigen::Index>(values.size(), 1));
    const double out = static_cast<double>(mean);
    return std::isfinite(out) ? out : std::numeric_limits<double>::infinity();
}

double phase1_logaddexp(double a, double b) {
    if (std::isinf(a) && a > 0.0) {
        return a;
    }
    if (std::isinf(b) && b > 0.0) {
        return b;
    }
    if (std::isinf(a) && a < 0.0) {
        return b;
    }
    if (std::isinf(b) && b < 0.0) {
        return a;
    }
    const double high = std::max(a, b);
    const double low = std::min(a, b);
    return high + std::log1p(std::exp(low - high));
}

Eigen::VectorXd phase1_signed_exp_difference(
    const Eigen::Ref<const Eigen::VectorXd>& log_pos,
    const Eigen::Ref<const Eigen::VectorXd>& log_neg) {
    if (log_pos.size() != log_neg.size()) {
        throw std::invalid_argument("signed exp difference inputs must align");
    }
    Eigen::VectorXd out(log_pos.size());
    for (Eigen::Index idx = 0; idx < log_pos.size(); ++idx) {
        const double lp = log_pos(idx);
        const double ln = log_neg(idx);
        const bool lp_pos_inf = std::isinf(lp) && lp > 0.0;
        const bool ln_pos_inf = std::isinf(ln) && ln > 0.0;
        if (std::isinf(lp) && lp < 0.0 && std::isinf(ln) && ln < 0.0) {
            out(idx) = 0.0;
        } else if (lp_pos_inf && !ln_pos_inf) {
            out(idx) = std::numeric_limits<double>::infinity();
        } else if (ln_pos_inf && !lp_pos_inf) {
            out(idx) = -std::numeric_limits<double>::infinity();
        } else if (lp_pos_inf && ln_pos_inf) {
            out(idx) = std::numeric_limits<double>::quiet_NaN();
        } else {
            const bool pos_dominates = lp >= ln;
            const double high = pos_dominates ? lp : ln;
            const double low = pos_dominates ? ln : lp;
            const double sign = pos_dominates ? 1.0 : -1.0;
            const double log_abs = high + std::log1p(-std::exp(low - high));
            out(idx) = sign * std::exp(log_abs);
        }
    }
    return out;
}

Phase1NoFENormCore build_phase1_nofe_norm_core(
    const Eigen::Ref<const RowMatrixXd>& x,
    const std::string& transformed_design_normalization) {
    if (x.rows() <= 0 || x.cols() <= 0) {
        throw std::invalid_argument("exog must be non-empty");
    }
    if (x.rows() < x.cols()) {
        throw std::invalid_argument("X_T_rank_deficient");
    }
    Phase1NoFENormCore norm;
    norm.transformed_design_normalization =
        normalize_phase1_transformed_design_normalization_core(
            transformed_design_normalization);
    norm.x = x;
    norm.scale.resize(x.cols());
    if (norm.transformed_design_normalization == "scale") {
        for (Eigen::Index col = 0; col < x.cols(); ++col) {
            const double mean = x.col(col).mean();
            double var = 0.0;
            for (Eigen::Index row = 0; row < x.rows(); ++row) {
                const double centered = x(row, col) - mean;
                var += centered * centered;
            }
            double scale = std::sqrt(var / static_cast<double>(x.rows()));
            if (scale == 0.0) {
                scale = 1.0;
            }
            norm.scale(col) = scale;
        }
        const double scale_max =
            norm.scale.size() == 0 ? 0.0 : norm.scale.maxCoeff();
        const double scale_min =
            norm.scale.size() == 0 ? 0.0 : norm.scale.minCoeff();
        const double scale_floor =
            std::max(PHASE1_S_REL_FLOOR * scale_max, PHASE1_S_ABS_BACKSTOP);
        if (scale_max == 0.0 || scale_min < scale_floor) {
            throw std::invalid_argument("S_singular");
        }
    } else {
        norm.scale.setOnes();
    }
    norm.x_star = x;
    for (Eigen::Index col = 0; col < x.cols(); ++col) {
        norm.x_star.col(col) /= norm.scale(col);
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(norm.x_star);
    norm.q =
        qr.householderQ() *
        Eigen::MatrixXd::Identity(norm.x_star.rows(), norm.x_star.cols());
    norm.r = qr.matrixQR()
                 .topLeftCorner(norm.x_star.cols(), norm.x_star.cols())
                 .template triangularView<Eigen::Upper>();
    const Eigen::VectorXd diag = norm.r.diagonal().cwiseAbs();
    const double max_diag = diag.size() == 0 ? 0.0 : diag.maxCoeff();
    const double min_diag = diag.size() == 0 ? 0.0 : diag.minCoeff();
    if (max_diag == 0.0 || min_diag < PHASE1_R_REL_FLOOR * max_diag) {
        throw std::invalid_argument("X_T_rank_deficient");
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(norm.r);
    const Eigen::VectorXd singular = svd.singularValues();
    const double max_sv = singular.size() == 0 ? 0.0 : singular.maxCoeff();
    const double min_sv = singular.size() == 0 ? 0.0 : singular.minCoeff();
    norm.condition_after =
        min_sv > 0.0 ? std::pow(max_sv / min_sv, 2.0)
                     : std::numeric_limits<double>::infinity();
    return norm;
}

Eigen::VectorXd phase1_pseudo_response_nofe_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::VectorXd>& g,
    double tau,
    double delta) {
    if (!std::isfinite(delta) || delta <= 0.0) {
        throw std::invalid_argument("delta_must_be_positive_finite");
    }
    Eigen::VectorXd out(y.size());
    const double log_delta = std::log(delta);
    for (Eigen::Index idx = 0; idx < y.size(); ++idx) {
        const double log_y =
            y(idx) > 0.0 ? std::log(y(idx)) : -std::numeric_limits<double>::infinity();
        const double log_top =
            phase1_logaddexp(log_y, log_delta + (tau - 1.0) * g(idx));
        const double log_bot =
            phase1_logaddexp(0.0, log_delta + (tau - 2.0) * g(idx));
        out(idx) = log_top - log_bot;
    }
    return out;
}

Eigen::VectorXd phase2_pseudo_response_nofe_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::VectorXd>& g,
    double tau,
    double rho) {
    if (!std::isfinite(rho) || rho <= 0.0) {
        throw std::invalid_argument("rho_must_be_positive_finite");
    }
    Eigen::VectorXd log_pos(y.size());
    Eigen::VectorXd log_neg(y.size());
    Eigen::VectorXd out(y.size());
    const double log_rho = std::log(rho);
    const double log_denom = std::log1p(rho);
    for (Eigen::Index idx = 0; idx < y.size(); ++idx) {
        const double log_y =
            y(idx) > 0.0 ? std::log(y(idx)) : -std::numeric_limits<double>::infinity();
        const double log_arg1 = phase1_logaddexp(log_y, log_rho + g(idx));
        const double log_u = log_y - g(idx);
        const double log_arg2 = phase1_logaddexp(log_rho, log_u);
        log_pos(idx) = log_y + (1.0 - tau) * g(idx) - log_denom;
        log_neg(idx) = (2.0 - tau) * g(idx) - log_denom;
        out(idx) = log_arg1 - log_arg2;
    }
    return out + phase1_signed_exp_difference(log_pos, log_neg);
}

Eigen::VectorXd phase1_ols_step_nofe_core(
    const Phase1NoFENormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& y_tilde) {
    return norm.r.template triangularView<Eigen::Upper>().solve(norm.q.transpose() * y_tilde);
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase1_fixed_point_nofe_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1NoFENormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    double delta,
    double tol,
    int max_iter) {
    Eigen::VectorXd beta = beta_star_init;
    Phase1NoFEDiagnosticCore diag;
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        diag.iterations = iteration;
        const Eigen::VectorXd pseudo =
            phase1_pseudo_response_nofe_core(y, norm.x_star * beta, tau, delta);
        const Eigen::VectorXd beta_next = phase1_ols_step_nofe_core(norm, pseudo);
        const Eigen::VectorXd step = beta_next - beta;
        diag.criterion_star = phase1_mean_square_or_inf(step);
        diag.criterion_canonical =
            phase1_mean_square_or_inf(step.cwiseQuotient(norm.scale));
        beta = beta_next;
        if (diag.criterion_star < tol) {
            diag.converged = true;
            break;
        }
    }
    return {beta, diag};
}

std::string phase2_failure_nofe_core(
    const std::vector<double>& crit_hist,
    double criterion) {
    if (!std::isfinite(criterion)) {
        return "criterion_nonfinite";
    }
    if (criterion > 1.0e12) {
        return "criterion_explosion";
    }
    if (crit_hist.size() >= 20 && criterion > 1.0e-6) {
        const double ref = crit_hist[crit_hist.size() - 10];
        if (std::isfinite(ref) && ref > 0.0) {
            const double rel_drop =
                (ref - criterion) / std::max(std::abs(ref), PHASE1_LOG_FLOOR);
            if (rel_drop < 1.0e-3) {
                return "criterion_stagnation";
            }
        }
    }
    return "";
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase2_fixed_point_nofe_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1NoFENormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    double rho,
    double tol,
    int max_iter) {
    Eigen::VectorXd beta = beta_star_init;
    Phase1NoFEDiagnosticCore diag;
    diag.rho = rho;
    std::vector<double> crit_hist;
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        diag.iterations = iteration;
        const Eigen::VectorXd pseudo =
            phase2_pseudo_response_nofe_core(y, norm.x_star * beta, tau, rho);
        bool finite = true;
        for (Eigen::Index idx = 0; idx < pseudo.size(); ++idx) {
            if (!std::isfinite(pseudo(idx))) {
                finite = false;
                break;
            }
        }
        if (!finite) {
            diag.failure_reason = "nonfinite_pseudo_response";
            break;
        }
        const Eigen::VectorXd beta_next = phase1_ols_step_nofe_core(norm, pseudo);
        const Eigen::VectorXd step = beta_next - beta;
        diag.criterion_star = phase1_mean_square_or_inf(step);
        diag.failure_reason = phase2_failure_nofe_core(crit_hist, diag.criterion_star);
        if (!diag.failure_reason.empty()) {
            break;
        }
        diag.criterion_canonical =
            phase1_mean_square_or_inf(step.cwiseQuotient(norm.scale));
        crit_hist.push_back(diag.criterion_star);
        beta = beta_next;
        if (diag.criterion_star < tol) {
            diag.converged = true;
            break;
        }
    }
    return {beta, diag};
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase2_grid_nofe_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1NoFENormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    const std::vector<double>& rho_grid,
    double tol,
    int max_iter) {
    Phase1NoFEDiagnosticCore aggregate;
    for (const double rho : rho_grid) {
        auto [beta, diag] = phase2_fixed_point_nofe_core(
            y,
            norm,
            beta_star_init,
            tau,
            rho,
            tol,
            max_iter);
        aggregate.iterations += diag.iterations;
        aggregate.rho_path.push_back(rho);
        if (diag.converged) {
            diag.rho_path = aggregate.rho_path;
            return {beta, diag};
        }
    }
    aggregate.converged = false;
    aggregate.criterion_star = std::numeric_limits<double>::quiet_NaN();
    aggregate.criterion_canonical = std::numeric_limits<double>::quiet_NaN();
    aggregate.failure_reason = "phase2_grid_exhausted";
    aggregate.rho = std::numeric_limits<double>::quiet_NaN();
    return {beta_star_init, aggregate};
}

bool phase1_status_converged(const std::string& status) {
    return status == "anchor" || status == "converged" ||
           status == "target_converged" || status == "branch_drift_warning";
}

Phase1NoFEStateCore make_phase1_nofe_state_core(
    double tau,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star,
    const Phase1NoFENormCore& norm,
    const std::string& status,
    const Phase1NoFEDiagnosticCore& diagnostics) {
    Phase1NoFEStateCore state;
    state.tau = tau;
    state.beta_star = beta_star;
    state.beta = beta_star.cwiseQuotient(norm.scale);
    state.linear = norm.x * state.beta;
    state.fitted.resize(state.linear.size());
    for (Eigen::Index idx = 0; idx < state.linear.size(); ++idx) {
        const double clipped = std::min(std::max(state.linear(idx), -700.0), 700.0);
        state.fitted(idx) = std::exp(clipped);
    }
    state.status = status;
    state.converged = phase1_status_converged(status);
    state.diagnostics = diagnostics;
    return state;
}

std::vector<std::vector<double>> phase1_guide_branches_core(
    const std::vector<double>& targets) {
    std::vector<double> lower;
    std::vector<double> upper;
    for (const double tau : targets) {
        if (std::abs(tau - 2.0) <= 1.0e-12) {
            continue;
        }
        if (tau < 2.0) {
            lower.push_back(tau);
        } else {
            upper.push_back(tau);
        }
    }
    std::sort(lower.begin(), lower.end(), std::greater<double>());
    std::sort(upper.begin(), upper.end());
    return {lower, upper};
}

bool phase1_vector_all_finite_core(
    const Eigen::Ref<const Eigen::VectorXd>& values) {
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        if (!std::isfinite(values(idx))) {
            return false;
        }
    }
    return true;
}

Eigen::VectorXd phase1_safe_exp_core(
    const Eigen::Ref<const Eigen::VectorXd>& values) {
    Eigen::VectorXd out(values.size());
    for (Eigen::Index idx = 0; idx < values.size(); ++idx) {
        const double clipped = std::min(std::max(values(idx), -700.0), 700.0);
        out(idx) = std::exp(clipped);
    }
    return out;
}

std::vector<int> phase1_group_counts_core(
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    std::vector<int> counts(static_cast<std::size_t>(n_groups), 0);
    for (const auto group : groups) {
        counts[static_cast<std::size_t>(group)] += 1;
    }
    return counts;
}

double phase1_matrix_condition_core(const Eigen::MatrixXd& values) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(values);
    const Eigen::VectorXd singular = svd.singularValues();
    const double max_sv = singular.size() == 0 ? 0.0 : singular.maxCoeff();
    const double min_sv = singular.size() == 0 ? 0.0 : singular.minCoeff();
    if (min_sv <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return max_sv / min_sv;
}

RowMatrixXd phase1_select_columns_core(
    const Eigen::Ref<const RowMatrixXd>& values,
    const std::vector<int>& columns) {
    RowMatrixXd out(values.rows(), static_cast<Eigen::Index>(columns.size()));
    for (std::size_t idx = 0; idx < columns.size(); ++idx) {
        out.col(static_cast<Eigen::Index>(idx)) =
            values.col(static_cast<Eigen::Index>(columns[idx]));
    }
    return out;
}

Eigen::VectorXd phase1_demean_oneway_vector_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const std::optional<Eigen::VectorXd>& weights) {
    Eigen::MatrixXd matrix(values.size(), 1);
    matrix.col(0) = values;
    const Eigen::MatrixXd demeaned =
        demean_oneway_matrix(matrix, groups, n_groups, weights);
    return demeaned.col(0);
}

Phase1OneWayNormCore build_phase1_oneway_fullcancel_norm_core(
    const Eigen::Ref<const RowMatrixXd>& x,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const std::string& transformed_design_normalization) {
    if (x.rows() <= 0 || x.cols() <= 0) {
        throw std::invalid_argument("exog must be non-empty");
    }
    Phase1OneWayNormCore out;
    out.x_raw = x;
    out.groups = groups;
    out.n_groups = n_groups;
    out.group_counts = phase1_group_counts_core(groups, n_groups);
    out.x_within = demean_oneway_matrix(x, groups, n_groups, std::nullopt);

    const double typical_norm =
        x.norm() / std::max(std::sqrt(static_cast<double>(x.cols())), 1.0);
    const double floor =
        PHASE1_FE_ABSORBED_REL_FLOOR * typical_norm *
        std::sqrt(static_cast<double>(x.rows()));
    for (Eigen::Index col = 0; col < x.cols(); ++col) {
        const double col_norm = out.x_within.col(col).norm();
        if (col_norm > floor) {
            out.kept_columns.push_back(static_cast<int>(col));
        } else {
            out.absorbed_columns.push_back(static_cast<int>(col));
        }
    }
    if (out.kept_columns.empty()) {
        throw std::invalid_argument("FE_absorbs_all_slopes");
    }
    const RowMatrixXd x_kept =
        phase1_select_columns_core(out.x_within, out.kept_columns);
    out.within_design_condition_before = phase1_matrix_condition_core(x_kept);
    out.norm = build_phase1_nofe_norm_core(
        x_kept,
        transformed_design_normalization);
    return out;
}

Eigen::VectorXd phase1_expand_beta_oneway_core(
    const Eigen::Ref<const Eigen::VectorXd>& beta_star,
    const Phase1OneWayNormCore& norm) {
    Eigen::VectorXd beta =
        Eigen::VectorXd::Zero(norm.x_raw.cols());
    for (std::size_t idx = 0; idx < norm.kept_columns.size(); ++idx) {
        beta(static_cast<Eigen::Index>(norm.kept_columns[idx])) =
            beta_star(static_cast<Eigen::Index>(idx)) /
            norm.norm.scale(static_cast<Eigen::Index>(idx));
    }
    return beta;
}

double phase1_score_residual_oneway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1OneWayNormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    double tau) {
    const Eigen::VectorXd q = norm.x_raw * beta;
    const Eigen::VectorXd log_weights = ((2.0 - tau) * q.array()).matrix();
    const Eigen::VectorXd weights = phase1_safe_exp_core(log_weights);
    const Eigen::VectorXd neg_q = -q;
    const Eigen::VectorXd ratio =
        y.cwiseProduct(phase1_safe_exp_core(neg_q));
    if (!phase1_vector_all_finite_core(weights) ||
        !phase1_vector_all_finite_core(ratio)) {
        return std::numeric_limits<double>::infinity();
    }
    Eigen::MatrixXd x_dm;
    Eigen::VectorXd ratio_dm;
    if (std::abs(tau - 2.0) <= 1.0e-10) {
        x_dm = demean_oneway_matrix(
            norm.x_raw,
            norm.groups,
            norm.n_groups,
            std::nullopt);
        ratio_dm = phase1_demean_oneway_vector_core(
            ratio,
            norm.groups,
            norm.n_groups,
            std::nullopt);
        const Eigen::VectorXd score = x_dm.transpose() * ratio_dm;
        return score.size() == 0
                   ? 0.0
                   : score.cwiseAbs().maxCoeff() /
                         std::max(static_cast<double>(y.size()), 1.0);
    }
    x_dm = demean_oneway_matrix(
        norm.x_raw,
        norm.groups,
        norm.n_groups,
        weights);
    ratio_dm = phase1_demean_oneway_vector_core(
        ratio,
        norm.groups,
        norm.n_groups,
        weights);
    const Eigen::VectorXd score =
        x_dm.transpose() * weights.cwiseProduct(ratio_dm);
    return score.size() == 0
               ? 0.0
               : score.cwiseAbs().maxCoeff() /
                     std::max(static_cast<double>(y.size()), 1.0);
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase1_fixed_point_oneway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1OneWayNormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    double delta,
    double tol,
    int max_iter) {
    Eigen::VectorXd beta_star = beta_star_init;
    Phase1NoFEDiagnosticCore diag;
    std::vector<double> crit_hist;
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        diag.iterations = iteration;
        const Eigen::VectorXd beta =
            phase1_expand_beta_oneway_core(beta_star, norm);
        const Eigen::VectorXd pseudo =
            phase1_pseudo_response_nofe_core(y, norm.x_raw * beta, tau, delta);
        const Eigen::VectorXd pseudo_dm = phase1_demean_oneway_vector_core(
            pseudo,
            norm.groups,
            norm.n_groups,
            std::nullopt);
        const Eigen::VectorXd beta_next =
            phase1_ols_step_nofe_core(norm.norm, pseudo_dm);
        const Eigen::VectorXd step = beta_next - beta_star;
        diag.criterion_star = phase1_mean_square_or_inf(step);
        diag.failure_reason =
            phase2_failure_nofe_core(crit_hist, diag.criterion_star);
        if (!diag.failure_reason.empty()) {
            return {beta_star, diag};
        }
        diag.criterion_canonical =
            phase1_mean_square_or_inf(step.cwiseQuotient(norm.norm.scale));
        crit_hist.push_back(diag.criterion_star);
        beta_star = beta_next;
        if (diag.criterion_star < tol) {
            diag.converged = true;
            return {beta_star, diag};
        }
    }
    diag.failure_reason = "max_iter_exhausted";
    return {beta_star, diag};
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase2_fixed_point_oneway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1OneWayNormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    double rho,
    double tol,
    int max_iter) {
    Eigen::VectorXd beta_star = beta_star_init;
    Phase1NoFEDiagnosticCore diag;
    diag.rho = rho;
    std::vector<double> crit_hist;
    RowMatrixXd x_kept_star_raw =
        phase1_select_columns_core(norm.x_raw, norm.kept_columns);
    for (Eigen::Index col = 0; col < x_kept_star_raw.cols(); ++col) {
        x_kept_star_raw.col(col) /= norm.norm.scale(col);
    }
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        diag.iterations = iteration;
        const Eigen::VectorXd beta =
            phase1_expand_beta_oneway_core(beta_star, norm);
        const Eigen::VectorXd q = norm.x_raw * beta;
        const Eigen::VectorXd log_weights = ((2.0 - tau) * q.array()).matrix();
        const Eigen::VectorXd weights = phase1_safe_exp_core(log_weights);
        const Eigen::VectorXd neg_q = -q;
        const Eigen::VectorXd ratio =
            y.cwiseProduct(phase1_safe_exp_core(neg_q));
        if (!phase1_vector_all_finite_core(weights) ||
            !phase1_vector_all_finite_core(ratio)) {
            diag.failure_reason = "nonfinite_fullcancel_weight_or_ratio";
            break;
        }
        const double eta = 1.0 / (1.0 + rho);
        Eigen::VectorXd beta_next;
        if (std::abs(tau - 2.0) <= 1.0e-10) {
            const Eigen::VectorXd ratio_dm = phase1_demean_oneway_vector_core(
                ratio,
                norm.groups,
                norm.n_groups,
                std::nullopt);
            const Eigen::VectorXd y_star =
                norm.norm.x_star * beta_star + eta * ratio_dm;
            beta_next = phase1_ols_step_nofe_core(norm.norm, y_star);
        } else {
            try {
                const Eigen::MatrixXd x_dm = demean_oneway_matrix(
                    x_kept_star_raw,
                    norm.groups,
                    norm.n_groups,
                    weights);
                const Eigen::VectorXd ratio_dm = phase1_demean_oneway_vector_core(
                    ratio,
                    norm.groups,
                    norm.n_groups,
                    weights);
                Eigen::MatrixXd lhs(x_dm.rows(), x_dm.cols());
                Eigen::MatrixXd rhs(x_dm.rows(), 1);
                for (Eigen::Index row = 0; row < x_dm.rows(); ++row) {
                    const double sqrt_w = std::sqrt(weights(row));
                    lhs.row(row) = sqrt_w * x_dm.row(row);
                    rhs(row, 0) =
                        sqrt_w * ((x_dm.row(row) * beta_star)(0) +
                                  eta * ratio_dm(row));
                }
                beta_next = pyiols::kernel::min_norm_lstsq(lhs, rhs).col(0);
            } catch (const std::exception&) {
                diag.failure_reason = "weighted_lstsq_failed";
                break;
            }
        }
        if (!phase1_vector_all_finite_core(beta_next)) {
            diag.failure_reason = "nonfinite_beta";
            break;
        }
        const Eigen::VectorXd step = beta_next - beta_star;
        diag.criterion_star = phase1_mean_square_or_inf(step);
        diag.failure_reason =
            phase2_failure_nofe_core(crit_hist, diag.criterion_star);
        if (!diag.failure_reason.empty()) {
            break;
        }
        diag.criterion_canonical =
            phase1_mean_square_or_inf(step.cwiseQuotient(norm.norm.scale));
        crit_hist.push_back(diag.criterion_star);
        beta_star = beta_next;
        if (diag.criterion_star < tol) {
            diag.converged = true;
            diag.failure_reason.clear();
            return {beta_star, diag};
        }
    }
    if (!diag.converged && diag.failure_reason.empty()) {
        diag.failure_reason = "max_iter_exhausted";
    }
    return {beta_star, diag};
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase2_grid_oneway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1OneWayNormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    const std::vector<double>& rho_grid,
    double tol,
    int max_iter) {
    Phase1NoFEDiagnosticCore aggregate;
    for (const double rho : rho_grid) {
        auto [beta, diag] = phase2_fixed_point_oneway_fullcancel_core(
            y,
            norm,
            beta_star_init,
            tau,
            rho,
            tol,
            max_iter);
        diag.rho = rho;
        aggregate.rho_diagnostics_path.push_back(diag);
        aggregate.iterations = diag.iterations;
        if (diag.converged) {
            diag.rho_diagnostics_path = aggregate.rho_diagnostics_path;
            return {beta, diag};
        }
    }
    aggregate.converged = false;
    aggregate.criterion_star = std::numeric_limits<double>::quiet_NaN();
    aggregate.criterion_canonical = std::numeric_limits<double>::quiet_NaN();
    aggregate.rho = std::numeric_limits<double>::quiet_NaN();
    aggregate.failure_reason =
        aggregate.rho_diagnostics_path.empty()
            ? "rho_grid_empty"
            : aggregate.rho_diagnostics_path.back().failure_reason;
    if (aggregate.failure_reason.empty()) {
        aggregate.failure_reason = "phase2_grid_exhausted";
    }
    return {beta_star_init, aggregate};
}

Phase1NoFEStateCore make_phase1_oneway_state_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    double tau,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star,
    const Phase1OneWayNormCore& norm,
    const std::string& status,
    Phase1NoFEDiagnosticCore diagnostics) {
    Phase1NoFEStateCore state;
    state.tau = tau;
    state.beta_star = beta_star;
    state.beta = phase1_expand_beta_oneway_core(beta_star, norm);
    state.linear = norm.x_raw * state.beta;
    state.fitted = phase1_safe_exp_core(state.linear);
    state.status = status;
    state.converged = phase1_status_converged(status);
    diagnostics.score_residual_max =
        phase1_score_residual_oneway_fullcancel_core(y, norm, state.beta, tau);
    state.diagnostics = diagnostics;
    state.kept_columns = norm.kept_columns;
    state.absorbed_columns = norm.absorbed_columns;
    state.n_groups = static_cast<int>(norm.n_groups);
    state.group_counts = norm.group_counts;
    return state;
}

Phase1NoFEStateCore make_phase1_oneway_support_failure_state_core(
    Eigen::Index nobs,
    Eigen::Index nvars,
    std::int64_t n_groups,
    const std::vector<int>& group_counts,
    const Phase1NoFEDiagnosticCore& diagnostics) {
    Phase1NoFEStateCore state;
    state.tau = 2.0;
    state.beta =
        Eigen::VectorXd::Constant(nvars, std::numeric_limits<double>::quiet_NaN());
    state.beta_star =
        Eigen::VectorXd::Constant(nvars, std::numeric_limits<double>::quiet_NaN());
    state.linear =
        Eigen::VectorXd::Constant(nobs, std::numeric_limits<double>::quiet_NaN());
    state.fitted =
        Eigen::VectorXd::Constant(nobs, std::numeric_limits<double>::quiet_NaN());
    state.status = "anchor_failed";
    state.converged = false;
    state.diagnostics = diagnostics;
    for (Eigen::Index col = 0; col < nvars; ++col) {
        state.absorbed_columns.push_back(static_cast<int>(col));
    }
    state.n_groups = static_cast<int>(n_groups);
    state.group_counts = group_counts;
    return state;
}

std::optional<Phase1NoFEDiagnosticCore> phase1_oneway_support_failure_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups) {
    Phase1NoFEDiagnosticCore diag;
    for (Eigen::Index row = 0; row < y.size(); ++row) {
        if (y(row) < 0.0) {
            diag.failure_reason = "negative_response_unsupported";
            diag.support_failure_reason = diag.failure_reason;
            diag.support_block_index = 0;
            diag.support_group_index = -1;
            diag.support_group_sum_y = y.minCoeff();
            return diag;
        }
    }
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(n_groups);
    for (Eigen::Index row = 0; row < y.size(); ++row) {
        sums(static_cast<Eigen::Index>(groups[static_cast<std::size_t>(row)])) +=
            y(row);
    }
    for (Eigen::Index group = 0; group < sums.size(); ++group) {
        if (sums(group) <= 0.0) {
            diag.failure_reason = "support_zero_fe_margin";
            diag.support_failure_reason = diag.failure_reason;
            diag.support_block_index = 0;
            diag.support_group_index = static_cast<int>(group);
            diag.support_group_sum_y = sums(group);
            return diag;
        }
    }
    return std::nullopt;
}

std::vector<std::vector<int>> phase1_manyway_group_counts_core(
    const GroupBundleCore& bundle) {
    std::vector<std::vector<int>> out;
    out.reserve(bundle.groups.size());
    for (std::size_t dim = 0; dim < bundle.groups.size(); ++dim) {
        std::vector<int> counts(
            static_cast<std::size_t>(bundle.counts[dim]),
            0);
        for (const auto group : bundle.groups[dim]) {
            counts[static_cast<std::size_t>(group)] += 1;
        }
        out.push_back(std::move(counts));
    }
    return out;
}

std::vector<std::int64_t> phase1_manyway_n_groups_core(
    const GroupBundleCore& bundle) {
    std::vector<std::int64_t> out;
    out.reserve(bundle.counts.size());
    for (const auto count : bundle.counts) {
        out.push_back(count);
    }
    return out;
}

Eigen::MatrixXd phase1_demean_manyway_matrix_core(
    const Eigen::Ref<const Eigen::MatrixXd>& values,
    const GroupBundleCore& bundle,
    const Eigen::VectorXd* weights,
    double tol,
    int maxiter) {
    if (maxiter <= 0) {
        throw std::invalid_argument("manyway_within_projection_not_converged");
    }
    Eigen::MatrixXd out;
    DemeanWorkspaceCore workspace;
    bool converged = false;
    demean_for_group_bundle_into_core(
        values,
        bundle,
        weights,
        tol,
        maxiter,
        out,
        workspace,
        &converged);
    if (!converged) {
        throw std::invalid_argument("manyway_within_projection_not_converged");
    }
    return out;
}

Eigen::VectorXd phase1_demean_manyway_vector_core(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const GroupBundleCore& bundle,
    const Eigen::VectorXd* weights,
    double tol,
    int maxiter) {
    if (weights == nullptr &&
        cpp_env_flag_enabled_core("PYIOLS_CPP_SPEED_PHASE1_MANYWAY_AP_WORKSPACE")) {
        if (maxiter <= 0) {
            throw std::invalid_argument("manyway_within_projection_not_converged");
        }
        Eigen::VectorXd out;
        DemeanVectorWorkspaceCore workspace;
        bool converged = false;
        demean_vector_for_group_bundle_unweighted_into_core(
            values,
            bundle,
            tol,
            maxiter,
            out,
            workspace,
            &converged);
        if (!converged) {
            throw std::invalid_argument("manyway_within_projection_not_converged");
        }
        return out;
    }
    Eigen::MatrixXd matrix(values.size(), 1);
    matrix.col(0) = values;
    const Eigen::MatrixXd demeaned =
        phase1_demean_manyway_matrix_core(matrix, bundle, weights, tol, maxiter);
    return demeaned.col(0);
}

Phase1ManyWayNormCore build_phase1_manyway_fullcancel_norm_core(
    const Eigen::Ref<const RowMatrixXd>& x,
    const GroupBundleCore& bundle,
    double within_tol,
    int within_max_iter,
    const std::string& transformed_design_normalization) {
    if (x.rows() <= 0 || x.cols() <= 0) {
        throw std::invalid_argument("exog must be non-empty");
    }
    if (bundle.groups.size() < 2) {
        throw std::invalid_argument("many-way fullcancel requires at least two FE blocks");
    }
    Phase1ManyWayNormCore out;
    out.x_raw = x;
    out.bundle = bundle;
    out.group_counts = phase1_manyway_group_counts_core(bundle);
    out.within_tol = within_tol;
    out.within_max_iter = within_max_iter;
    out.x_within = phase1_demean_manyway_matrix_core(
        x,
        bundle,
        nullptr,
        within_tol,
        within_max_iter);

    const double typical_norm =
        x.norm() / std::max(std::sqrt(static_cast<double>(x.cols())), 1.0);
    const double floor =
        PHASE1_FE_ABSORBED_REL_FLOOR * typical_norm *
        std::sqrt(static_cast<double>(x.rows()));
    for (Eigen::Index col = 0; col < x.cols(); ++col) {
        const double col_norm = out.x_within.col(col).norm();
        if (col_norm > floor) {
            out.kept_columns.push_back(static_cast<int>(col));
        } else {
            out.absorbed_columns.push_back(static_cast<int>(col));
        }
    }
    if (out.kept_columns.empty()) {
        throw std::invalid_argument("FE_absorbs_all_slopes");
    }
    const RowMatrixXd x_kept =
        phase1_select_columns_core(out.x_within, out.kept_columns);
    out.within_design_condition_before = phase1_matrix_condition_core(x_kept);
    out.norm = build_phase1_nofe_norm_core(
        x_kept,
        transformed_design_normalization);
    return out;
}

Eigen::VectorXd phase1_expand_beta_manyway_core(
    const Eigen::Ref<const Eigen::VectorXd>& beta_star,
    const Phase1ManyWayNormCore& norm) {
    Eigen::VectorXd beta =
        Eigen::VectorXd::Zero(norm.x_raw.cols());
    for (std::size_t idx = 0; idx < norm.kept_columns.size(); ++idx) {
        beta(static_cast<Eigen::Index>(norm.kept_columns[idx])) =
            beta_star(static_cast<Eigen::Index>(idx)) /
            norm.norm.scale(static_cast<Eigen::Index>(idx));
    }
    return beta;
}

double phase1_score_residual_manyway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1ManyWayNormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    double tau,
    std::string* failure_reason = nullptr) {
    try {
        const Eigen::VectorXd q = norm.x_raw * beta;
        const Eigen::VectorXd log_weights = ((2.0 - tau) * q.array()).matrix();
        const Eigen::VectorXd weights = phase1_safe_exp_core(log_weights);
        const Eigen::VectorXd neg_q = -q;
        const Eigen::VectorXd ratio =
            y.cwiseProduct(phase1_safe_exp_core(neg_q));
        if (!phase1_vector_all_finite_core(weights) ||
            !phase1_vector_all_finite_core(ratio)) {
            return std::numeric_limits<double>::infinity();
        }
        Eigen::MatrixXd x_dm;
        Eigen::VectorXd ratio_dm;
        if (std::abs(tau - 2.0) <= 1.0e-10) {
            if (cpp_env_flag_enabled_core(
                    "PYIOLS_CPP_SPEED_PHASE1_MANYWAY_AP_WORKSPACE")) {
                x_dm = norm.x_within;
            } else {
                x_dm = phase1_demean_manyway_matrix_core(
                    norm.x_raw,
                    norm.bundle,
                    nullptr,
                    norm.within_tol,
                    norm.within_max_iter);
            }
            ratio_dm = phase1_demean_manyway_vector_core(
                ratio,
                norm.bundle,
                nullptr,
                norm.within_tol,
                norm.within_max_iter);
            const Eigen::VectorXd score = x_dm.transpose() * ratio_dm;
            return score.size() == 0
                       ? 0.0
                       : score.cwiseAbs().maxCoeff() /
                             std::max(static_cast<double>(y.size()), 1.0);
        }
        x_dm = phase1_demean_manyway_matrix_core(
            norm.x_raw,
            norm.bundle,
            &weights,
            norm.within_tol,
            norm.within_max_iter);
        ratio_dm = phase1_demean_manyway_vector_core(
            ratio,
            norm.bundle,
            &weights,
            norm.within_tol,
            norm.within_max_iter);
        const Eigen::VectorXd score =
            x_dm.transpose() * weights.cwiseProduct(ratio_dm);
        return score.size() == 0
                   ? 0.0
                   : score.cwiseAbs().maxCoeff() /
                         std::max(static_cast<double>(y.size()), 1.0);
    } catch (const std::exception& exc) {
        if (failure_reason != nullptr) {
            *failure_reason =
                std::string(exc.what()).empty()
                    ? "manyway_within_projection_failed"
                    : std::string(exc.what());
        }
        return std::numeric_limits<double>::infinity();
    }
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase1_fixed_point_manyway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1ManyWayNormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    double delta,
    double tol,
    double within_tol,
    int max_iter) {
    Eigen::VectorXd beta_star = beta_star_init;
    Phase1NoFEDiagnosticCore diag;
    std::vector<double> crit_hist;
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        diag.iterations = iteration;
        const Eigen::VectorXd beta =
            phase1_expand_beta_manyway_core(beta_star, norm);
        const Eigen::VectorXd pseudo =
            phase1_pseudo_response_nofe_core(y, norm.x_raw * beta, tau, delta);
        Eigen::VectorXd pseudo_dm;
        try {
            pseudo_dm = phase1_demean_manyway_vector_core(
                pseudo,
                norm.bundle,
                nullptr,
                within_tol,
                norm.within_max_iter);
        } catch (const std::exception& exc) {
            diag.failure_reason =
                std::string(exc.what()).empty()
                    ? "manyway_within_projection_failed"
                    : std::string(exc.what());
            return {beta_star, diag};
        }
        const Eigen::VectorXd beta_next =
            phase1_ols_step_nofe_core(norm.norm, pseudo_dm);
        const Eigen::VectorXd step = beta_next - beta_star;
        diag.criterion_star = phase1_mean_square_or_inf(step);
        diag.failure_reason =
            phase2_failure_nofe_core(crit_hist, diag.criterion_star);
        if (!diag.failure_reason.empty()) {
            return {beta_star, diag};
        }
        diag.criterion_canonical =
            phase1_mean_square_or_inf(step.cwiseQuotient(norm.norm.scale));
        crit_hist.push_back(diag.criterion_star);
        beta_star = beta_next;
        if (diag.criterion_star < tol) {
            diag.converged = true;
            return {beta_star, diag};
        }
    }
    diag.failure_reason = "max_iter_exhausted";
    return {beta_star, diag};
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase2_fixed_point_manyway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1ManyWayNormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    double rho,
    double tol,
    int max_iter) {
    Eigen::VectorXd beta_star = beta_star_init;
    Phase1NoFEDiagnosticCore diag;
    diag.rho = rho;
    std::vector<double> crit_hist;
    RowMatrixXd x_kept_star_raw =
        phase1_select_columns_core(norm.x_raw, norm.kept_columns);
    for (Eigen::Index col = 0; col < x_kept_star_raw.cols(); ++col) {
        x_kept_star_raw.col(col) /= norm.norm.scale(col);
    }
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        diag.iterations = iteration;
        const Eigen::VectorXd beta =
            phase1_expand_beta_manyway_core(beta_star, norm);
        const Eigen::VectorXd q = norm.x_raw * beta;
        const Eigen::VectorXd log_weights = ((2.0 - tau) * q.array()).matrix();
        const Eigen::VectorXd weights = phase1_safe_exp_core(log_weights);
        const Eigen::VectorXd neg_q = -q;
        const Eigen::VectorXd ratio =
            y.cwiseProduct(phase1_safe_exp_core(neg_q));
        if (!phase1_vector_all_finite_core(weights) ||
            !phase1_vector_all_finite_core(ratio)) {
            diag.failure_reason = "nonfinite_fullcancel_weight_or_ratio";
            break;
        }
        const double eta = 1.0 / (1.0 + rho);
        Eigen::VectorXd beta_next;
        if (std::abs(tau - 2.0) <= 1.0e-10) {
            try {
                const Eigen::VectorXd ratio_dm =
                    phase1_demean_manyway_vector_core(
                        ratio,
                        norm.bundle,
                        nullptr,
                        norm.within_tol,
                        norm.within_max_iter);
                const Eigen::VectorXd y_star =
                    norm.norm.x_star * beta_star + eta * ratio_dm;
                beta_next = phase1_ols_step_nofe_core(norm.norm, y_star);
            } catch (const std::exception& exc) {
                diag.failure_reason =
                    std::string(exc.what()).empty()
                        ? "manyway_within_projection_failed"
                        : std::string(exc.what());
                break;
            }
        } else {
            try {
                Eigen::MatrixXd x_dm;
                Eigen::VectorXd ratio_dm;
                if (cpp_env_flag_enabled_core(
                        "PYIOLS_CPP_SPEED_COMBINED_MANYWAY_PHASE2")) {
                    Eigen::MatrixXd combined(
                        x_kept_star_raw.rows(), x_kept_star_raw.cols() + 1);
                    combined.leftCols(x_kept_star_raw.cols()) = x_kept_star_raw;
                    combined.col(x_kept_star_raw.cols()) = ratio;
                    const Eigen::MatrixXd combined_dm =
                        phase1_demean_manyway_matrix_core(
                            combined,
                            norm.bundle,
                            &weights,
                            norm.within_tol,
                            norm.within_max_iter);
                    x_dm = combined_dm.leftCols(x_kept_star_raw.cols());
                    ratio_dm = combined_dm.col(x_kept_star_raw.cols());
                } else {
                    x_dm = phase1_demean_manyway_matrix_core(
                        x_kept_star_raw,
                        norm.bundle,
                        &weights,
                        norm.within_tol,
                        norm.within_max_iter);
                    ratio_dm = phase1_demean_manyway_vector_core(
                        ratio,
                        norm.bundle,
                        &weights,
                        norm.within_tol,
                        norm.within_max_iter);
                }
                Eigen::MatrixXd lhs(x_dm.rows(), x_dm.cols());
                Eigen::MatrixXd rhs(x_dm.rows(), 1);
                for (Eigen::Index row = 0; row < x_dm.rows(); ++row) {
                    const double sqrt_w = std::sqrt(weights(row));
                    lhs.row(row) = sqrt_w * x_dm.row(row);
                    rhs(row, 0) =
                        sqrt_w * ((x_dm.row(row) * beta_star)(0) +
                                  eta * ratio_dm(row));
                }
                beta_next = pyiols::kernel::min_norm_lstsq(lhs, rhs).col(0);
            } catch (const std::exception& exc) {
                const std::string reason = exc.what();
                diag.failure_reason =
                    reason == "manyway_within_projection_not_converged"
                        ? reason
                        : "weighted_lstsq_failed";
                break;
            }
        }
        if (!phase1_vector_all_finite_core(beta_next)) {
            diag.failure_reason = "nonfinite_beta";
            break;
        }
        const Eigen::VectorXd step = beta_next - beta_star;
        diag.criterion_star = phase1_mean_square_or_inf(step);
        diag.failure_reason =
            phase2_failure_nofe_core(crit_hist, diag.criterion_star);
        if (!diag.failure_reason.empty()) {
            break;
        }
        diag.criterion_canonical =
            phase1_mean_square_or_inf(step.cwiseQuotient(norm.norm.scale));
        crit_hist.push_back(diag.criterion_star);
        beta_star = beta_next;
        if (diag.criterion_star < tol) {
            diag.converged = true;
            diag.failure_reason.clear();
            return {beta_star, diag};
        }
    }
    if (!diag.converged && diag.failure_reason.empty()) {
        diag.failure_reason = "max_iter_exhausted";
    }
    return {beta_star, diag};
}

std::pair<Eigen::VectorXd, Phase1NoFEDiagnosticCore>
phase2_grid_manyway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Phase1ManyWayNormCore& norm,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star_init,
    double tau,
    const std::vector<double>& rho_grid,
    double tol,
    int max_iter) {
    Phase1NoFEDiagnosticCore aggregate;
    for (const double rho : rho_grid) {
        auto [beta, diag] = phase2_fixed_point_manyway_fullcancel_core(
            y,
            norm,
            beta_star_init,
            tau,
            rho,
            tol,
            max_iter);
        diag.rho = rho;
        aggregate.rho_diagnostics_path.push_back(diag);
        aggregate.iterations = diag.iterations;
        if (diag.converged) {
            diag.rho_diagnostics_path = aggregate.rho_diagnostics_path;
            return {beta, diag};
        }
    }
    aggregate.converged = false;
    aggregate.criterion_star = std::numeric_limits<double>::quiet_NaN();
    aggregate.criterion_canonical = std::numeric_limits<double>::quiet_NaN();
    aggregate.rho = std::numeric_limits<double>::quiet_NaN();
    aggregate.failure_reason =
        aggregate.rho_diagnostics_path.empty()
            ? "rho_grid_empty"
            : aggregate.rho_diagnostics_path.back().failure_reason;
    if (aggregate.failure_reason.empty()) {
        aggregate.failure_reason = "phase2_grid_exhausted";
    }
    return {beta_star_init, aggregate};
}

Phase1ManyWayStateCore make_phase1_manyway_state_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    double tau,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star,
    const Phase1ManyWayNormCore& norm,
    const std::string& status,
    Phase1NoFEDiagnosticCore diagnostics) {
    Phase1ManyWayStateCore state;
    state.tau = tau;
    state.beta_star = beta_star;
    state.beta = phase1_expand_beta_manyway_core(beta_star, norm);
    state.linear = norm.x_raw * state.beta;
    state.fitted = phase1_safe_exp_core(state.linear);
    state.status = status;
    state.converged = phase1_status_converged(status);
    std::string score_failure;
    diagnostics.score_residual_max =
        phase1_score_residual_manyway_fullcancel_core(
            y,
            norm,
            state.beta,
            tau,
            &score_failure);
    if (!score_failure.empty()) {
        diagnostics.score_residual_failure_reason = score_failure;
    }
    state.diagnostics = diagnostics;
    state.kept_columns = norm.kept_columns;
    state.absorbed_columns = norm.absorbed_columns;
    state.n_groups = phase1_manyway_n_groups_core(norm.bundle);
    state.group_counts = norm.group_counts;
    return state;
}

Phase1ManyWayStateCore make_phase1_manyway_support_failure_state_core(
    Eigen::Index nobs,
    Eigen::Index nvars,
    const GroupBundleCore& bundle,
    const Phase1NoFEDiagnosticCore& diagnostics) {
    Phase1ManyWayStateCore state;
    state.tau = 2.0;
    state.beta =
        Eigen::VectorXd::Constant(nvars, std::numeric_limits<double>::quiet_NaN());
    state.beta_star =
        Eigen::VectorXd::Constant(nvars, std::numeric_limits<double>::quiet_NaN());
    state.linear =
        Eigen::VectorXd::Constant(nobs, std::numeric_limits<double>::quiet_NaN());
    state.fitted =
        Eigen::VectorXd::Constant(nobs, std::numeric_limits<double>::quiet_NaN());
    state.status = "anchor_failed";
    state.converged = false;
    state.diagnostics = diagnostics;
    for (Eigen::Index col = 0; col < nvars; ++col) {
        state.absorbed_columns.push_back(static_cast<int>(col));
    }
    state.n_groups = phase1_manyway_n_groups_core(bundle);
    state.group_counts = phase1_manyway_group_counts_core(bundle);
    return state;
}

std::optional<Phase1NoFEDiagnosticCore> phase1_manyway_support_failure_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const GroupBundleCore& bundle) {
    Phase1NoFEDiagnosticCore diag;
    for (Eigen::Index row = 0; row < y.size(); ++row) {
        if (y(row) < 0.0) {
            diag.failure_reason = "negative_response_unsupported";
            diag.support_failure_reason = diag.failure_reason;
            diag.support_block_index = -1;
            diag.support_group_index = -1;
            diag.support_group_sum_y = y.minCoeff();
            return diag;
        }
    }
    for (std::size_t block = 0; block < bundle.groups.size(); ++block) {
        Eigen::VectorXd sums = Eigen::VectorXd::Zero(bundle.counts[block]);
        for (Eigen::Index row = 0; row < y.size(); ++row) {
            sums(static_cast<Eigen::Index>(
                bundle.groups[block][static_cast<std::size_t>(row)])) += y(row);
        }
        for (Eigen::Index group = 0; group < sums.size(); ++group) {
            if (sums(group) <= 0.0) {
                diag.failure_reason = "support_zero_fe_margin";
                diag.support_failure_reason = diag.failure_reason;
                diag.support_block_index = static_cast<int>(block);
                diag.support_group_index = static_cast<int>(group);
                diag.support_group_sum_y = sums(group);
                return diag;
            }
        }
    }
    return std::nullopt;
}

Phase1NoFEResultCore solve_phase1_tau_guide_nofe_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const RowMatrixXd>& x,
    const std::vector<double>& target_taus,
    const std::vector<double>& delta_grid,
    const std::vector<double>& rho_grid,
    double tol,
    int max_iter_anchor,
    int max_iter_guide,
    int max_iter_target,
    const std::string& transformed_design_normalization) {
    if (y.size() != x.rows()) {
        throw std::invalid_argument("endog and exog must have the same rows");
    }
    if (target_taus.empty()) {
        throw std::invalid_argument("target_taus must be nonempty");
    }
    Phase1NoFEResultCore result;
    result.target_taus = target_taus;
    const Phase1NoFENormCore norm = build_phase1_nofe_norm_core(
        x,
        transformed_design_normalization);
    result.condition_after = norm.condition_after;
    result.scale_min = norm.scale.minCoeff();
    result.scale_max = norm.scale.maxCoeff();
    result.transformed_design_normalization =
        norm.transformed_design_normalization;
    Eigen::VectorXd beta_star = Eigen::VectorXd::Zero(x.cols());
    Phase1NoFEDiagnosticCore phase1_anchor_diag;
    for (const double delta : delta_grid) {
        auto [next_beta, diag] = phase1_fixed_point_nofe_core(
            y,
            norm,
            beta_star,
            2.0,
            delta,
            tol,
            max_iter_anchor);
        diag.delta = delta;
        phase1_anchor_diag.phase1_path.push_back(diag);
        beta_star = next_beta;
        phase1_anchor_diag.iterations += diag.iterations;
        phase1_anchor_diag.converged = diag.converged;
        phase1_anchor_diag.criterion_star = diag.criterion_star;
        phase1_anchor_diag.criterion_canonical = diag.criterion_canonical;
    }
    auto [beta_anchor, anchor_diag] = phase2_grid_nofe_core(
        y,
        norm,
        beta_star,
        2.0,
        rho_grid,
        tol,
        max_iter_anchor);
    anchor_diag.phase1_path = phase1_anchor_diag.phase1_path;
    if (!anchor_diag.converged) {
        result.failure_reason = "phase1_anchor_failed";
        result.anchor = make_phase1_nofe_state_core(
            2.0,
            beta_star,
            norm,
            "anchor_failed",
            anchor_diag);
        return result;
    }
    result.anchor = make_phase1_nofe_state_core(
        2.0,
        beta_anchor,
        norm,
        "anchor",
        anchor_diag);
    result.guide_path.push_back(result.anchor);
    for (const auto& branch : phase1_guide_branches_core(target_taus)) {
        Eigen::VectorXd beta_branch = beta_anchor;
        for (const double tau : branch) {
            auto [beta_next, diag] = phase1_fixed_point_nofe_core(
                y,
                norm,
                beta_branch,
                tau,
                1.0,
                std::max(tol, 1.0e-7),
                max_iter_guide);
            const std::string status = diag.converged ? "converged" : "guide_failed";
            result.guide_path.push_back(
                make_phase1_nofe_state_core(tau, beta_next, norm, status, diag));
            beta_branch = beta_next;
            if (!diag.converged) {
                break;
            }
        }
    }
    std::vector<Phase1NoFEStateCore> converged_guide;
    for (const auto& state : result.guide_path) {
        if (state.status == "anchor" || state.status == "converged") {
            converged_guide.push_back(state);
        }
    }
    for (const double tau : target_taus) {
        auto best = converged_guide.begin();
        for (auto iter = converged_guide.begin(); iter != converged_guide.end(); ++iter) {
            const double best_same = std::abs(best->tau - tau) <= 1.0e-12 ? 0.0 : 1.0;
            const double iter_same = std::abs(iter->tau - tau) <= 1.0e-12 ? 0.0 : 1.0;
            const auto best_key =
                std::make_tuple(best_same, std::abs(best->tau - tau), std::abs(best->tau - 2.0));
            const auto iter_key =
                std::make_tuple(iter_same, std::abs(iter->tau - tau), std::abs(iter->tau - 2.0));
            if (iter_key < best_key) {
                best = iter;
            }
        }
        if (std::abs(tau - 2.0) <= 1.0e-12 && best->status == "anchor") {
            Phase1NoFEStateCore target = *best;
            target.status = "target_converged";
            target.converged = true;
            target.source_tau = 2.0;
            target.target_tau = 2.0;
            result.target_results.push_back(target);
            continue;
        }
        auto [beta_target, diag] = phase2_grid_nofe_core(
            y,
            norm,
            best->beta_star,
            tau,
            rho_grid,
            tol,
            max_iter_target);
        const std::string status = diag.converged ? "target_converged" : "target_failed";
        Phase1NoFEStateCore target =
            make_phase1_nofe_state_core(tau, beta_target, norm, status, diag);
        target.source_tau = best->tau;
        target.target_tau = tau;
        result.target_results.push_back(target);
    }
    result.converged = true;
    for (const auto& target : result.target_results) {
        if (!target.converged) {
            result.converged = false;
            break;
        }
    }
    result.failure_reason = result.converged ? "" : "phase2_target_failed";
    return result;
}

Phase1OneWayResultCore solve_phase1_tau_guide_oneway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const RowMatrixXd>& x,
    const std::vector<std::int64_t>& groups,
    std::int64_t n_groups,
    const std::vector<double>& target_taus,
    const std::vector<double>& delta_grid,
    const std::vector<double>& rho_grid,
    double tol,
    int max_iter_anchor,
    int max_iter_guide,
    int max_iter_target,
    double branch_drift_warn_rel,
    const std::string& transformed_design_normalization) {
    if (y.size() != x.rows()) {
        throw std::invalid_argument("endog and exog must have the same rows");
    }
    if (groups.size() != static_cast<std::size_t>(y.size())) {
        throw std::invalid_argument("groups and endog must have the same rows");
    }
    if (target_taus.empty()) {
        throw std::invalid_argument("target_taus must be nonempty");
    }
    Phase1OneWayResultCore result;
    result.target_taus = target_taus;
    const std::vector<int> group_counts =
        phase1_group_counts_core(groups, n_groups);
    const auto support_failure =
        phase1_oneway_support_failure_core(y, groups, n_groups);
    if (support_failure.has_value()) {
        result.converged = false;
        result.failure_reason = support_failure->failure_reason;
        result.anchor = make_phase1_oneway_support_failure_state_core(
            y.size(),
            x.cols(),
            n_groups,
            group_counts,
            *support_failure);
        return result;
    }

    const Phase1OneWayNormCore norm =
        build_phase1_oneway_fullcancel_norm_core(
            x,
            groups,
            n_groups,
            transformed_design_normalization);
    result.condition_after = norm.norm.condition_after;
    result.scale_min = norm.norm.scale.minCoeff();
    result.scale_max = norm.norm.scale.maxCoeff();
    result.transformed_design_normalization =
        norm.norm.transformed_design_normalization;
    result.within_design_condition_before = norm.within_design_condition_before;
    result.kept_columns = norm.kept_columns;
    result.absorbed_columns = norm.absorbed_columns;

    const Eigen::VectorXd log_y_plus_one = (y.array() + 1.0).log().matrix();
    const Eigen::VectorXd y_init = phase1_demean_oneway_vector_core(
        log_y_plus_one,
        norm.groups,
        norm.n_groups,
        std::nullopt);
    Eigen::VectorXd beta_star = phase1_ols_step_nofe_core(norm.norm, y_init);
    Phase1NoFEDiagnosticCore phase1_anchor_diag;
    for (const double delta : delta_grid) {
        auto [next_beta, diag] = phase1_fixed_point_oneway_fullcancel_core(
            y,
            norm,
            beta_star,
            2.0,
            delta,
            tol,
            max_iter_anchor);
        diag.delta = delta;
        phase1_anchor_diag.phase1_path.push_back(diag);
        beta_star = next_beta;
    }
    auto [beta_anchor, anchor_diag] = phase2_grid_oneway_fullcancel_core(
        y,
        norm,
        beta_star,
        2.0,
        rho_grid,
        tol,
        max_iter_anchor);
    anchor_diag.phase1_path = phase1_anchor_diag.phase1_path;
    if (!anchor_diag.converged) {
        result.converged = false;
        result.failure_reason = "phase1_anchor_failed";
        result.anchor = make_phase1_oneway_state_core(
            y,
            2.0,
            beta_star,
            norm,
            "anchor_failed",
            anchor_diag);
        return result;
    }

    result.anchor = make_phase1_oneway_state_core(
        y,
        2.0,
        beta_anchor,
        norm,
        "anchor",
        anchor_diag);
    result.guide_path.push_back(result.anchor);
    for (const auto& branch : phase1_guide_branches_core(target_taus)) {
        Eigen::VectorXd beta_branch = beta_anchor;
        for (const double tau : branch) {
            auto [beta_next, diag] = phase1_fixed_point_oneway_fullcancel_core(
                y,
                norm,
                beta_branch,
                tau,
                1.0,
                std::max(tol, 1.0e-7),
                max_iter_guide);
            const std::string status = diag.converged ? "converged" : "guide_failed";
            result.guide_path.push_back(
                make_phase1_oneway_state_core(y, tau, beta_next, norm, status, diag));
            beta_branch = beta_next;
            if (!diag.converged) {
                break;
            }
        }
    }

    std::vector<Phase1NoFEStateCore> converged_guide;
    for (const auto& state : result.guide_path) {
        if (state.status == "anchor" || state.status == "converged") {
            converged_guide.push_back(state);
        }
    }
    for (const double tau : target_taus) {
        auto best = converged_guide.begin();
        for (auto iter = converged_guide.begin(); iter != converged_guide.end(); ++iter) {
            const double best_same = std::abs(best->tau - tau) <= 1.0e-12 ? 0.0 : 1.0;
            const double iter_same = std::abs(iter->tau - tau) <= 1.0e-12 ? 0.0 : 1.0;
            const auto best_key =
                std::make_tuple(best_same, std::abs(best->tau - tau), std::abs(best->tau - 2.0));
            const auto iter_key =
                std::make_tuple(iter_same, std::abs(iter->tau - tau), std::abs(iter->tau - 2.0));
            if (iter_key < best_key) {
                best = iter;
            }
        }
        if (std::abs(tau - 2.0) <= 1.0e-12 && best->status == "anchor") {
            Phase1NoFEStateCore target = *best;
            target.status = "target_converged";
            target.converged = true;
            target.source_tau = 2.0;
            target.target_tau = 2.0;
            result.target_results.push_back(target);
            continue;
        }
        auto [beta_target, diag] = phase2_grid_oneway_fullcancel_core(
            y,
            norm,
            best->beta_star,
            tau,
            rho_grid,
            tol,
            max_iter_target);
        std::string status = diag.converged ? "target_converged" : "target_failed";
        if (diag.converged) {
            const Eigen::VectorXd beta_target_kept =
                beta_target.cwiseQuotient(norm.norm.scale);
            Eigen::VectorXd beta_source_kept(norm.kept_columns.size());
            for (std::size_t idx = 0; idx < norm.kept_columns.size(); ++idx) {
                beta_source_kept(static_cast<Eigen::Index>(idx)) =
                    best->beta(static_cast<Eigen::Index>(norm.kept_columns[idx]));
            }
            diag.guide_to_target_distance =
                (beta_target_kept - beta_source_kept).norm();
            diag.guide_to_target_distance_star =
                (beta_target - best->beta_star).norm();
            diag.branch_drift_relative =
                diag.guide_to_target_distance /
                std::max(beta_target_kept.norm(), 1.0e-12);
            if (diag.branch_drift_relative > branch_drift_warn_rel) {
                status = "branch_drift_warning";
            }
        }
        Phase1NoFEStateCore target =
            make_phase1_oneway_state_core(y, tau, beta_target, norm, status, diag);
        target.source_tau = best->tau;
        target.target_tau = tau;
        result.target_results.push_back(target);
    }
    result.converged = true;
    for (const auto& target : result.target_results) {
        if (!target.converged) {
            result.converged = false;
            break;
        }
    }
    result.failure_reason = result.converged ? "" : "phase2_target_failed";
    return result;
}

Phase1ManyWayResultCore solve_phase1_tau_guide_manyway_fullcancel_core(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const RowMatrixXd>& x,
    const GroupBundleCore& bundle,
    const std::vector<double>& target_taus,
    const std::vector<double>& delta_grid,
    const std::vector<double>& rho_grid,
    double tol,
    int max_iter_anchor,
    int max_iter_guide,
    int max_iter_target,
    double branch_drift_warn_rel,
    double within_tol,
    int within_max_iter,
    const std::string& transformed_design_normalization) {
    if (y.size() != x.rows()) {
        throw std::invalid_argument("endog and exog must have the same rows");
    }
    if (bundle.groups.size() < 2) {
        throw std::invalid_argument("many-way fullcancel requires at least two FE blocks");
    }
    if (target_taus.empty()) {
        throw std::invalid_argument("target_taus must be nonempty");
    }
    Phase1ManyWayResultCore result;
    result.target_taus = target_taus;
    const auto support_failure = phase1_manyway_support_failure_core(y, bundle);
    if (support_failure.has_value()) {
        result.converged = false;
        result.failure_reason = support_failure->failure_reason;
        result.anchor = make_phase1_manyway_support_failure_state_core(
            y.size(),
            x.cols(),
            bundle,
            *support_failure);
        return result;
    }

    const Phase1ManyWayNormCore norm =
        build_phase1_manyway_fullcancel_norm_core(
            x,
            bundle,
            within_tol,
            within_max_iter,
            transformed_design_normalization);
    const double inner_within_tol =
        phase1_research_inner_within_tol_core(norm.within_tol);
    result.condition_after = norm.norm.condition_after;
    result.scale_min = norm.norm.scale.minCoeff();
    result.scale_max = norm.norm.scale.maxCoeff();
    result.transformed_design_normalization =
        norm.norm.transformed_design_normalization;
    result.within_design_condition_before = norm.within_design_condition_before;
    result.kept_columns = norm.kept_columns;
    result.absorbed_columns = norm.absorbed_columns;

    const Eigen::VectorXd log_y_plus_one = (y.array() + 1.0).log().matrix();
    const Eigen::VectorXd y_init = phase1_demean_manyway_vector_core(
        log_y_plus_one,
        norm.bundle,
        nullptr,
        norm.within_tol,
        norm.within_max_iter);
    Eigen::VectorXd beta_star = phase1_ols_step_nofe_core(norm.norm, y_init);
    Phase1NoFEDiagnosticCore phase1_anchor_diag;
    for (const double delta : delta_grid) {
        auto [next_beta, diag] = phase1_fixed_point_manyway_fullcancel_core(
            y,
            norm,
            beta_star,
            2.0,
            delta,
            tol,
            inner_within_tol,
            max_iter_anchor);
        diag.research_tiered_fe_tol = inner_within_tol > norm.within_tol;
        diag.research_inner_fe_demean_tol = inner_within_tol;
        diag.delta = delta;
        phase1_anchor_diag.phase1_path.push_back(diag);
        beta_star = next_beta;
    }
    auto [beta_anchor, anchor_diag] = phase2_grid_manyway_fullcancel_core(
        y,
        norm,
        beta_star,
        2.0,
        rho_grid,
        tol,
        max_iter_anchor);
    anchor_diag.phase1_path = phase1_anchor_diag.phase1_path;
    if (!anchor_diag.converged) {
        result.converged = false;
        result.failure_reason = "phase1_anchor_failed";
        result.anchor = make_phase1_manyway_state_core(
            y,
            2.0,
            beta_star,
            norm,
            "anchor_failed",
            anchor_diag);
        return result;
    }

    result.anchor = make_phase1_manyway_state_core(
        y,
        2.0,
        beta_anchor,
        norm,
        "anchor",
        anchor_diag);
    result.guide_path.push_back(result.anchor);
    for (const auto& branch : phase1_guide_branches_core(target_taus)) {
        Eigen::VectorXd beta_branch = beta_anchor;
        for (const double tau : branch) {
            auto [beta_next, diag] = phase1_fixed_point_manyway_fullcancel_core(
                y,
                norm,
                beta_branch,
                tau,
                1.0,
                std::max(tol, 1.0e-7),
                inner_within_tol,
                max_iter_guide);
            diag.research_tiered_fe_tol = inner_within_tol > norm.within_tol;
            diag.research_inner_fe_demean_tol = inner_within_tol;
            const std::string status = diag.converged ? "converged" : "guide_failed";
            result.guide_path.push_back(
                make_phase1_manyway_state_core(y, tau, beta_next, norm, status, diag));
            beta_branch = beta_next;
            if (!diag.converged) {
                break;
            }
        }
    }

    std::vector<Phase1ManyWayStateCore> converged_guide;
    for (const auto& state : result.guide_path) {
        if (state.status == "anchor" || state.status == "converged") {
            converged_guide.push_back(state);
        }
    }
    for (const double tau : target_taus) {
        auto best = converged_guide.begin();
        for (auto iter = converged_guide.begin(); iter != converged_guide.end(); ++iter) {
            const double best_same = std::abs(best->tau - tau) <= 1.0e-12 ? 0.0 : 1.0;
            const double iter_same = std::abs(iter->tau - tau) <= 1.0e-12 ? 0.0 : 1.0;
            const auto best_key =
                std::make_tuple(best_same, std::abs(best->tau - tau), std::abs(best->tau - 2.0));
            const auto iter_key =
                std::make_tuple(iter_same, std::abs(iter->tau - tau), std::abs(iter->tau - 2.0));
            if (iter_key < best_key) {
                best = iter;
            }
        }
        if (std::abs(tau - 2.0) <= 1.0e-12 && best->status == "anchor") {
            Phase1ManyWayStateCore target = *best;
            target.status = "target_converged";
            target.converged = true;
            target.source_tau = 2.0;
            target.target_tau = 2.0;
            result.target_results.push_back(target);
            continue;
        }
        auto [beta_target, diag] = phase2_grid_manyway_fullcancel_core(
            y,
            norm,
            best->beta_star,
            tau,
            rho_grid,
            tol,
            max_iter_target);
        std::string status = diag.converged ? "target_converged" : "target_failed";
        if (diag.converged) {
            const Eigen::VectorXd beta_target_kept =
                beta_target.cwiseQuotient(norm.norm.scale);
            Eigen::VectorXd beta_source_kept(norm.kept_columns.size());
            for (std::size_t idx = 0; idx < norm.kept_columns.size(); ++idx) {
                beta_source_kept(static_cast<Eigen::Index>(idx)) =
                    best->beta(static_cast<Eigen::Index>(norm.kept_columns[idx]));
            }
            diag.guide_to_target_distance =
                (beta_target_kept - beta_source_kept).norm();
            diag.guide_to_target_distance_star =
                (beta_target - best->beta_star).norm();
            diag.branch_drift_relative =
                diag.guide_to_target_distance /
                std::max(beta_target_kept.norm(), 1.0e-12);
            if (diag.branch_drift_relative > branch_drift_warn_rel) {
                status = "branch_drift_warning";
            }
        }
        Phase1ManyWayStateCore target =
            make_phase1_manyway_state_core(y, tau, beta_target, norm, status, diag);
        target.source_tau = best->tau;
        target.target_tau = tau;
        result.target_results.push_back(target);
    }
    result.converged = true;
    for (const auto& target : result.target_results) {
        if (!target.converged) {
            result.converged = false;
            break;
        }
    }
    result.failure_reason = result.converged ? "" : "phase2_target_failed";
    return result;
}

py::dict phase1_diag_to_dict(const Phase1NoFEDiagnosticCore& diag) {
    py::dict out;
    out["iterations"] = diag.iterations;
    out["converged"] = diag.converged;
    out["criterion_star"] = diag.criterion_star;
    out["criterion_canonical"] = diag.criterion_canonical;
    out["failure_reason"] =
        diag.failure_reason.empty() ? py::none() : py::cast(diag.failure_reason);
    out["rho"] = diag.rho;
    py::list rho_path;
    for (const double rho : diag.rho_path) {
        rho_path.append(rho);
    }
    out["rho_path"] = py::tuple(rho_path);
    if (std::isfinite(diag.delta)) {
        out["delta"] = diag.delta;
    }
    out["research_tiered_fe_tol"] = diag.research_tiered_fe_tol;
    if (std::isfinite(diag.research_inner_fe_demean_tol)) {
        out["research_inner_fe_demean_tol"] =
            diag.research_inner_fe_demean_tol;
    }
    if (!diag.phase1_path.empty()) {
        py::list phase1_path;
        for (const auto& item : diag.phase1_path) {
            phase1_path.append(phase1_diag_to_dict(item));
        }
        out["phase1_path"] = py::tuple(phase1_path);
    }
    return out;
}

py::dict phase1_state_to_dict(const Phase1NoFEStateCore& state) {
    py::dict out;
    out["tau"] = state.tau;
    out["beta"] = vector_to_array(state.beta);
    out["beta_star"] = vector_to_array(state.beta_star);
    out["linear_predictor"] = vector_to_array(state.linear);
    out["fittedvalues"] = vector_to_array(state.fitted);
    out["status"] = state.status;
    out["converged"] = state.converged;
    out["diagnostics"] = phase1_diag_to_dict(state.diagnostics);
    if (std::isfinite(state.source_tau)) {
        out["source_tau"] = state.source_tau;
    }
    if (std::isfinite(state.target_tau)) {
        out["target_tau"] = state.target_tau;
    }
    return out;
}

py::dict phase1_nofe_result_to_dict(const Phase1NoFEResultCore& result) {
    py::dict out;
    out["method_family"] = "phase1_tau_guide";
    out["topology"] = "noFE";
    out["converged"] = result.converged;
    out["failure_reason"] =
        result.failure_reason.empty() ? py::none() : py::cast(result.failure_reason);
    py::list targets;
    for (const double tau : result.target_taus) {
        targets.append(tau);
    }
    out["target_taus"] = py::tuple(targets);
    out["anchor"] = phase1_state_to_dict(result.anchor);
    py::list guide_path;
    for (const auto& state : result.guide_path) {
        guide_path.append(phase1_state_to_dict(state));
    }
    out["guide_path"] = py::tuple(guide_path);
    py::list target_results;
    for (const auto& state : result.target_results) {
        target_results.append(phase1_state_to_dict(state));
    }
    out["target_results"] = py::tuple(target_results);
    py::dict normalization;
    normalization["transformed_design_normalization"] =
        result.transformed_design_normalization;
    normalization["condition_after"] = result.condition_after;
    normalization["scale_min"] = result.scale_min;
    normalization["scale_max"] = result.scale_max;
    out["normalization"] = normalization;
    out["kernel_backend"] = "cpp";
    out["cpp_symbol"] = "solve_phase1_tau_guide_nofe";
    return out;
}

py::tuple int_vector_to_tuple(const std::vector<int>& values) {
    py::list out;
    for (const int value : values) {
        out.append(value);
    }
    return py::tuple(out);
}

py::dict phase1_oneway_support_to_dict(const Phase1NoFEDiagnosticCore& diag) {
    py::dict support;
    support["reason"] = diag.support_failure_reason;
    if (diag.support_block_index >= 0) {
        support["block_index"] = diag.support_block_index;
    } else {
        support["block_index"] = py::none();
    }
    if (diag.support_group_index >= 0) {
        support["group_index"] = diag.support_group_index;
    } else {
        support["group_index"] = py::none();
    }
    support["group_sum_y"] = diag.support_group_sum_y;
    return support;
}

py::dict phase1_oneway_rho_diag_to_dict(const Phase1NoFEDiagnosticCore& diag) {
    py::dict out;
    out["rho"] = diag.rho;
    out["iterations"] = diag.iterations;
    out["converged"] = diag.converged;
    out["criterion_star"] = diag.criterion_star;
    out["criterion_canonical"] = diag.criterion_canonical;
    out["failure_reason"] =
        diag.failure_reason.empty() ? py::none() : py::cast(diag.failure_reason);
    return out;
}

py::dict phase1_oneway_diag_to_dict(const Phase1NoFEDiagnosticCore& diag) {
    py::dict out;
    if (!diag.support_failure_reason.empty()) {
        out["support"] = phase1_oneway_support_to_dict(diag);
        out["failure_reason"] = diag.failure_reason;
        return out;
    }
    out["iterations"] = diag.iterations;
    out["converged"] = diag.converged;
    out["criterion_star"] = diag.criterion_star;
    out["criterion_canonical"] = diag.criterion_canonical;
    if (!diag.failure_reason.empty() || std::isfinite(diag.rho) ||
        !diag.rho_diagnostics_path.empty()) {
        out["failure_reason"] =
            diag.failure_reason.empty() ? py::none() : py::cast(diag.failure_reason);
    }
    if (std::isfinite(diag.rho) || !diag.rho_diagnostics_path.empty()) {
        out["rho"] = diag.rho;
    }
    if (!diag.rho_diagnostics_path.empty()) {
        py::list rho_path;
        for (const auto& item : diag.rho_diagnostics_path) {
            rho_path.append(phase1_oneway_rho_diag_to_dict(item));
        }
        out["rho_path"] = py::tuple(rho_path);
    }
    if (std::isfinite(diag.delta)) {
        out["delta"] = diag.delta;
    }
    out["research_tiered_fe_tol"] = diag.research_tiered_fe_tol;
    if (std::isfinite(diag.research_inner_fe_demean_tol)) {
        out["research_inner_fe_demean_tol"] =
            diag.research_inner_fe_demean_tol;
    }
    if (!diag.phase1_path.empty()) {
        py::list phase1_path;
        for (const auto& item : diag.phase1_path) {
            phase1_path.append(phase1_oneway_diag_to_dict(item));
        }
        out["phase1_path"] = py::tuple(phase1_path);
    }
    if (!std::isnan(diag.score_residual_max)) {
        out["score_residual_max"] = diag.score_residual_max;
    }
    if (!diag.score_residual_failure_reason.empty()) {
        out["score_residual_failure_reason"] =
            diag.score_residual_failure_reason;
    }
    if (!std::isnan(diag.guide_to_target_distance)) {
        out["guide_to_target_distance"] = diag.guide_to_target_distance;
    }
    if (!std::isnan(diag.guide_to_target_distance_star)) {
        out["guide_to_target_distance_star"] =
            diag.guide_to_target_distance_star;
    }
    if (!std::isnan(diag.branch_drift_relative)) {
        out["branch_drift_relative"] = diag.branch_drift_relative;
    }
    return out;
}

py::dict phase1_oneway_state_to_dict(const Phase1NoFEStateCore& state) {
    py::dict out;
    out["tau"] = state.tau;
    out["beta"] = vector_to_array(state.beta);
    out["beta_star"] = vector_to_array(state.beta_star);
    out["linear_predictor"] = vector_to_array(state.linear);
    out["fittedvalues"] = vector_to_array(state.fitted);
    out["status"] = state.status;
    out["converged"] = state.converged;
    out["kept_columns"] = int_vector_to_tuple(state.kept_columns);
    out["absorbed_columns"] = int_vector_to_tuple(state.absorbed_columns);
    py::dict state_meta;
    state_meta["n_groups"] = state.n_groups;
    state_meta["group_counts"] = int_vector_to_tuple(state.group_counts);
    out["state"] = state_meta;
    out["diagnostics"] = phase1_oneway_diag_to_dict(state.diagnostics);
    if (std::isfinite(state.source_tau)) {
        out["source_tau"] = state.source_tau;
    }
    if (std::isfinite(state.target_tau)) {
        out["target_tau"] = state.target_tau;
    }
    if (!std::isnan(state.diagnostics.guide_to_target_distance)) {
        out["guide_to_target_distance"] =
            state.diagnostics.guide_to_target_distance;
    }
    if (!std::isnan(state.diagnostics.guide_to_target_distance_star)) {
        out["guide_to_target_distance_star"] =
            state.diagnostics.guide_to_target_distance_star;
    }
    return out;
}

py::dict phase1_oneway_result_to_dict(const Phase1OneWayResultCore& result) {
    py::dict out;
    out["method_family"] = "phase1_tau_guide";
    out["topology"] = "oneway_fullcancel";
    out["converged"] = result.converged;
    out["failure_reason"] =
        result.failure_reason.empty() ? py::none() : py::cast(result.failure_reason);
    py::list targets;
    for (const double tau : result.target_taus) {
        targets.append(tau);
    }
    out["target_taus"] = py::tuple(targets);
    out["anchor"] = phase1_oneway_state_to_dict(result.anchor);
    py::list guide_path;
    for (const auto& state : result.guide_path) {
        guide_path.append(phase1_oneway_state_to_dict(state));
    }
    out["guide_path"] = py::tuple(guide_path);
    py::list target_results;
    for (const auto& state : result.target_results) {
        target_results.append(phase1_oneway_state_to_dict(state));
    }
    out["target_results"] = py::tuple(target_results);
    py::dict normalization;
    if (!result.kept_columns.empty()) {
        normalization["transformed_design_normalization"] =
            result.transformed_design_normalization;
        normalization["condition_after"] = result.condition_after;
        normalization["scale_min"] = result.scale_min;
        normalization["scale_max"] = result.scale_max;
        normalization["kept_columns"] = int_vector_to_tuple(result.kept_columns);
        normalization["absorbed_columns"] =
            int_vector_to_tuple(result.absorbed_columns);
        normalization["within_design_condition_before"] =
            result.within_design_condition_before;
    }
    out["normalization"] = normalization;
    out["kernel_backend"] = "cpp";
    out["cpp_symbol"] = "solve_phase1_tau_guide_oneway_fullcancel";
    return out;
}

py::tuple int64_vector_to_tuple(const std::vector<std::int64_t>& values) {
    py::list out;
    for (const auto value : values) {
        out.append(value);
    }
    return py::tuple(out);
}

py::tuple nested_int_vector_to_tuple(
    const std::vector<std::vector<int>>& values) {
    py::list out;
    for (const auto& item : values) {
        out.append(int_vector_to_tuple(item));
    }
    return py::tuple(out);
}

py::dict phase1_manyway_state_to_dict(const Phase1ManyWayStateCore& state) {
    py::dict out;
    out["tau"] = state.tau;
    out["beta"] = vector_to_array(state.beta);
    out["beta_star"] = vector_to_array(state.beta_star);
    out["linear_predictor"] = vector_to_array(state.linear);
    out["fittedvalues"] = vector_to_array(state.fitted);
    out["status"] = state.status;
    out["converged"] = state.converged;
    out["kept_columns"] = int_vector_to_tuple(state.kept_columns);
    out["absorbed_columns"] = int_vector_to_tuple(state.absorbed_columns);
    py::dict state_meta;
    state_meta["n_fe_blocks"] = static_cast<int>(state.n_groups.size());
    state_meta["n_groups"] = int64_vector_to_tuple(state.n_groups);
    state_meta["group_counts"] = nested_int_vector_to_tuple(state.group_counts);
    out["state"] = state_meta;
    py::dict diagnostics = phase1_oneway_diag_to_dict(state.diagnostics);
    const bool failed_target = state.status == "target_failed";
    if (failed_target) {
        diagnostics["guide_to_target_distance"] =
            state.diagnostics.guide_to_target_distance;
        diagnostics["guide_to_target_distance_star"] =
            state.diagnostics.guide_to_target_distance_star;
        diagnostics["branch_drift_relative"] =
            state.diagnostics.branch_drift_relative;
    }
    out["diagnostics"] = diagnostics;
    if (std::isfinite(state.source_tau)) {
        out["source_tau"] = state.source_tau;
    }
    if (std::isfinite(state.target_tau)) {
        out["target_tau"] = state.target_tau;
    }
    if (failed_target || !std::isnan(state.diagnostics.guide_to_target_distance)) {
        out["guide_to_target_distance"] =
            state.diagnostics.guide_to_target_distance;
    }
    if (failed_target ||
        !std::isnan(state.diagnostics.guide_to_target_distance_star)) {
        out["guide_to_target_distance_star"] =
            state.diagnostics.guide_to_target_distance_star;
    }
    return out;
}

py::dict phase1_manyway_result_to_dict(const Phase1ManyWayResultCore& result) {
    py::dict out;
    out["method_family"] = "phase1_tau_guide";
    out["topology"] = "manyway_fullcancel";
    out["converged"] = result.converged;
    out["failure_reason"] =
        result.failure_reason.empty() ? py::none() : py::cast(result.failure_reason);
    py::list targets;
    for (const double tau : result.target_taus) {
        targets.append(tau);
    }
    out["target_taus"] = py::tuple(targets);
    out["anchor"] = phase1_manyway_state_to_dict(result.anchor);
    py::list guide_path;
    for (const auto& state : result.guide_path) {
        guide_path.append(phase1_manyway_state_to_dict(state));
    }
    out["guide_path"] = py::tuple(guide_path);
    py::list target_results;
    for (const auto& state : result.target_results) {
        target_results.append(phase1_manyway_state_to_dict(state));
    }
    out["target_results"] = py::tuple(target_results);
    py::dict normalization;
    if (!result.kept_columns.empty()) {
        normalization["transformed_design_normalization"] =
            result.transformed_design_normalization;
        normalization["condition_after"] = result.condition_after;
        normalization["scale_min"] = result.scale_min;
        normalization["scale_max"] = result.scale_max;
        normalization["kept_columns"] = int_vector_to_tuple(result.kept_columns);
        normalization["absorbed_columns"] =
            int_vector_to_tuple(result.absorbed_columns);
        normalization["within_design_condition_before"] =
            result.within_design_condition_before;
    }
    out["normalization"] = normalization;
    out["kernel_backend"] = "cpp";
    out["cpp_symbol"] = "solve_phase1_tau_guide_manyway_fullcancel";
    return out;
}

py::dict solve_phase1_tau_guide_nofe_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    py::object target_taus_obj,
    double tol,
    int max_iter_anchor,
    int max_iter_guide,
    int max_iter_target,
    py::object delta_grid_obj,
    py::object rho_grid_obj,
    std::string transformed_design_normalization) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    const std::vector<double> targets = normalize_phase1_tau_sequence(
        double_vector_from_object(target_taus_obj, "target_taus"),
        "target_taus");
    const std::vector<double> delta_grid = phase1_grid_from_object(
        delta_grid_obj,
        -5.0,
        5.0,
        5,
        "delta");
    const std::vector<double> rho_grid = phase1_grid_from_object(
        rho_grid_obj,
        -7.0,
        14.0,
        30,
        "rho");
    Phase1NoFEResultCore result;
    {
        py::gil_scoped_release release;
        result = solve_phase1_tau_guide_nofe_core(
            endog_vec,
            exog_mat,
            targets,
            delta_grid,
            rho_grid,
            tol,
            max_iter_anchor,
            max_iter_guide,
            max_iter_target,
            transformed_design_normalization);
    }
    return phase1_nofe_result_to_dict(result);
}

py::dict solve_phase1_tau_guide_oneway_fullcancel_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> group,
    py::object target_taus_obj,
    double tol,
    int max_iter_anchor,
    int max_iter_guide,
    int max_iter_target,
    py::object delta_grid_obj,
    py::object rho_grid_obj,
    double branch_drift_warn_rel,
    std::string transformed_design_normalization) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    if (endog_vec.size() == 0) {
        throw std::invalid_argument("group cannot be empty");
    }
    const auto [group_vec, n_groups] =
        compact_int64_groups_from_array(group, endog_vec.size());
    const std::vector<double> targets = normalize_phase1_tau_sequence(
        double_vector_from_object(target_taus_obj, "target_taus"),
        "target_taus");
    const std::vector<double> delta_grid = phase1_grid_from_object(
        delta_grid_obj,
        -5.0,
        5.0,
        5,
        "delta");
    const std::vector<double> rho_grid = phase1_grid_from_object(
        rho_grid_obj,
        -7.0,
        14.0,
        30,
        "rho");
    Phase1OneWayResultCore result;
    {
        py::gil_scoped_release release;
        result = solve_phase1_tau_guide_oneway_fullcancel_core(
            endog_vec,
            exog_mat,
            group_vec,
            n_groups,
            targets,
            delta_grid,
            rho_grid,
            tol,
            max_iter_anchor,
            max_iter_guide,
            max_iter_target,
            branch_drift_warn_rel,
            transformed_design_normalization);
    }
    return phase1_oneway_result_to_dict(result);
}

py::dict solve_phase1_tau_guide_manyway_fullcancel_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> endog,
    py::array_t<double, py::array::c_style | py::array::forcecast> exog,
    py::object groups_obj,
    py::object n_groups_obj,
    py::object target_taus_obj,
    double tol,
    int max_iter_anchor,
    int max_iter_guide,
    int max_iter_target,
    py::object delta_grid_obj,
    py::object rho_grid_obj,
    double branch_drift_warn_rel,
    double within_tol,
    int within_max_iter,
    std::string transformed_design_normalization) {
    const Eigen::VectorXd endog_vec = vector_from_array(endog, "endog");
    const RowMatrixXd exog_mat = matrix_from_array(exog, "exog");
    if (endog_vec.size() == 0) {
        throw std::invalid_argument("groups cannot be empty");
    }
    const GroupBundleCore bundle = group_bundle_from_objects(
        groups_obj,
        n_groups_obj,
        endog_vec.size(),
        "many-way fullcancel requires at least two FE blocks");
    if (bundle.groups.size() < 2) {
        throw std::invalid_argument("many-way fullcancel requires at least two FE blocks");
    }
    const std::vector<double> targets = normalize_phase1_tau_sequence(
        double_vector_from_object(target_taus_obj, "target_taus"),
        "target_taus");
    const std::vector<double> delta_grid = phase1_grid_from_object(
        delta_grid_obj,
        -5.0,
        5.0,
        5,
        "delta");
    const std::vector<double> rho_grid = phase1_grid_from_object(
        rho_grid_obj,
        -7.0,
        14.0,
        30,
        "rho");
    Phase1ManyWayResultCore result;
    {
        py::gil_scoped_release release;
        result = solve_phase1_tau_guide_manyway_fullcancel_core(
            endog_vec,
            exog_mat,
            bundle,
            targets,
            delta_grid,
            rho_grid,
            tol,
            max_iter_anchor,
            max_iter_guide,
            max_iter_target,
            branch_drift_warn_rel,
            within_tol,
            within_max_iter,
            transformed_design_normalization);
    }
    return phase1_manyway_result_to_dict(result);
}

}  // namespace

PYBIND11_MODULE(_kernel_cpp, m) {
    m.doc() = "pybind11/Eigen backend for pyiols estimator kernels.";
    m.def("backend_info", &backend_info);
    m.def("exp_guard", &exp_guard, py::arg("values"));
    m.def("group_sum", &group_sum, py::arg("values"), py::arg("groups"),
          py::arg("n_groups"));
    m.def("group_logsumexp", &group_logsumexp, py::arg("values"),
          py::arg("groups"), py::arg("n_groups"));
    m.def("demean_oneway", &demean_oneway, py::arg("values"),
          py::arg("groups"), py::arg("n_groups"), py::arg("weights") = py::none());
    m.def("demean_oneway_with_instruments", &demean_oneway_with_instruments,
          py::arg("values"), py::arg("groups"), py::arg("n_groups"),
          py::arg("instruments") = py::none());
    m.def("demean_manyway", &demean_manyway_cpp, py::arg("values"),
          py::arg("groups"), py::arg("n_groups"),
          py::arg("weights") = py::none(), py::arg("tol") = 1.0e-12,
          py::arg("maxiter") = 5000);
    m.def("lstsq_min_norm", &lstsq_min_norm, py::arg("a"), py::arg("b"));
    m.def("anderson_lstsq", &anderson_lstsq, py::arg("f_mat"),
          py::arg("residual"));
    m.def("solve_fullrecover_tau_path", &solve_fullrecover_tau_path_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("tau_target"),
          py::arg("group") = py::none(), py::arg("n_groups") = py::none(),
          py::arg("control") = py::none());
    m.def("solve_nofe_tau_path", &solve_nofe_tau_path_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("tau_target"),
          py::arg("control") = py::none());
    m.def("solve_nofe_direct_fixed_tau", &solve_nofe_direct_fixed_tau_cpp,
          py::arg("endog"), py::arg("exog_slopes"), py::arg("tau"),
          py::arg("control"), py::arg("rho_grid_policy") = "legacy");
    // runtime_v4: provenance flag; the pinned campaign build must report false.
#if defined(PYIOLS_DIRECT_TEST_HOOKS)
    m.attr("direct_test_hooks_compiled") = true;
    m.def("_test_arm_direct_post_check_fault", [](int phase, int count) {
        g_direct_test_fault_phase = phase;
        g_direct_test_fault_remaining = count;
        g_direct_test_fault_fired = 0;
    }, py::arg("phase"), py::arg("count"));
    m.def("_test_direct_post_check_fault_fired", []() {
        return g_direct_test_fault_fired;
    });
#else
    m.attr("direct_test_hooks_compiled") = false;
#endif
    m.def("solve_i2sls_nofe_tau2", &solve_i2sls_nofe_tau2_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("z_full"),
          py::arg("control") = py::none(), py::arg("x_breve") = py::none(),
          py::arg("transform_groups") = py::none(),
          py::arg("transform_counts") = py::none(),
          py::arg("transform_tol") = 1.0e-10,
          py::arg("transform_maxiter") = 1000);
    m.def("solve_i2sls_nofe_tau_path", &solve_i2sls_nofe_tau_path_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("z_full"),
          py::arg("tau_target"), py::arg("beta_anchor"),
          py::arg("rho_anchor") = py::none(), py::arg("control") = py::none(),
          py::arg("transform_groups") = py::none(),
          py::arg("transform_counts") = py::none(),
          py::arg("transform_tol") = 1.0e-10,
          py::arg("transform_maxiter") = 1000);
    m.def("solve_fullrecover_tau_path_manyway", &solve_fullrecover_tau_path_manyway_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("tau_target"),
          py::arg("groups"), py::arg("n_groups"), py::arg("control") = py::none());
    m.def("solve_fullcancel_tau_path", &solve_fullcancel_tau_path_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("tau_target"),
          py::arg("groups"), py::arg("n_groups"), py::arg("control") = py::none());
    m.def("solve_mixed_fullcancel_tau_path",
          &solve_mixed_fullcancel_tau_path_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("tau_target"),
          py::arg("cancel_groups"), py::arg("cancel_counts"),
          py::arg("recover_groups"), py::arg("recover_counts"),
          py::arg("control") = py::none());
    m.def("solve_phase1_tau_guide_nofe",
          &solve_phase1_tau_guide_nofe_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("target_taus"),
          py::arg("tol") = 1.0e-8, py::arg("max_iter_anchor") = 5000,
          py::arg("max_iter_guide") = 5000,
          py::arg("max_iter_target") = 10000,
          py::arg("delta_grid") = py::none(), py::arg("rho_grid") = py::none(),
          py::arg("transformed_design_normalization") = "scale");
    m.def("solve_phase1_tau_guide_oneway_fullcancel",
          &solve_phase1_tau_guide_oneway_fullcancel_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("group"),
          py::arg("target_taus"), py::arg("tol") = 1.0e-8,
          py::arg("max_iter_anchor") = 5000,
          py::arg("max_iter_guide") = 5000,
          py::arg("max_iter_target") = 10000,
          py::arg("delta_grid") = py::none(), py::arg("rho_grid") = py::none(),
          py::arg("branch_drift_warn_rel") = 1.0,
          py::arg("transformed_design_normalization") = "scale");
    m.def("solve_phase1_tau_guide_manyway_fullcancel",
          &solve_phase1_tau_guide_manyway_fullcancel_cpp,
          py::arg("endog"), py::arg("exog"), py::arg("groups"),
          py::arg("n_groups"), py::arg("target_taus"),
          py::arg("tol") = 1.0e-8,
          py::arg("max_iter_anchor") = 5000,
          py::arg("max_iter_guide") = 5000,
          py::arg("max_iter_target") = 10000,
          py::arg("delta_grid") = py::none(), py::arg("rho_grid") = py::none(),
          py::arg("branch_drift_warn_rel") = 1.0,
          py::arg("within_tol") = 1.0e-10,
          py::arg("within_max_iter") = 1000,
          py::arg("transformed_design_normalization") = "scale");
    m.def("mixed_support_conditioning_preflight",
          &mixed_support_conditioning_preflight_cpp,
          py::arg("endog"), py::arg("mixed_design"), py::arg("cancel_groups"),
          py::arg("cancel_counts"), py::arg("recover_groups"),
          py::arg("recover_counts"), py::arg("recover_names"),
          py::arg("control") = py::none());
    m.def("mixed_design_matrix", &mixed_design_matrix_cpp,
          py::arg("exog"), py::arg("recover_groups"),
          py::arg("recover_counts"), py::arg("recover_names"));
    m.def("sandwich_covariance", &sandwich_covariance_cpp,
          py::arg("design"), py::arg("bread_weight"), py::arg("score"),
          py::arg("se_type"), py::arg("df_denom"),
          py::arg("cluster_codes") = py::none(),
          py::arg("n_clusters") = py::none());
    m.def("mixed_covariance_projection", &mixed_covariance_projection_cpp,
          py::arg("x_slopes"), py::arg("cancel_groups"),
          py::arg("cancel_counts"), py::arg("recover_groups"),
          py::arg("recover_counts"), py::arg("demean_weights"),
          py::arg("bread_weight"), py::arg("score"), py::arg("se_type"),
          py::arg("df_denom"), py::arg("cluster_codes") = py::none(),
          py::arg("n_clusters") = py::none(), py::arg("tol") = 1.0e-12,
          py::arg("maxiter") = 5000);
}
