use crate::linear::sparse_lu::NativeSparseLuSolver;
use crate::linear::gmres::GmresSolver;
use crate::solver::shared::diagnostics::Diagnostics;
use crate::ffi::callbacks::NativeJvpFn;

pub enum LinearSolver {
    SparseLu(NativeSparseLuSolver),
    Gmres(GmresSolver),
}

impl LinearSolver {
    pub fn new(n: usize, bw: isize) -> Self {
        if bw == -1 {
            Self::Gmres(GmresSolver::new(n))
        } else {
            Self::SparseLu(NativeSparseLuSolver::new(n, bw))
        }
    }

    #[inline]
    pub fn solve(&mut self, b: &mut [f64], diag: &mut Diagnostics) -> Result<(), String> {
        match self {
            Self::SparseLu(lu) => lu.solve(b, diag),
            Self::Gmres(gmres) => gmres.solve(b, diag),
        }
    }

    #[inline]
    pub fn is_stale(&self) -> bool {
        match self {
            Self::SparseLu(lu) => lu.is_stale,
            Self::Gmres(_) => false,
        }
    }

    #[inline]
    pub fn mark_stale(&mut self) {
        if let Self::SparseLu(lu) = self {
            lu.mark_stale();
        }
    }

    #[inline]
    pub fn set_stale(&mut self, stale: bool) {
        if let Self::SparseLu(lu) = self {
            lu.is_stale = stale;
        }
    }

    #[inline]
    pub fn factorize_from_triplets(&mut self, diag: &mut Diagnostics) -> Result<(), String> {
        match self {
            Self::SparseLu(lu) => lu.factorize_from_triplets(diag),
            Self::Gmres(_) => Ok(()),
        }
    }

    #[inline]
    pub fn triplets(&self) -> &Vec<(usize, usize, f64)> {
        match self {
            Self::SparseLu(lu) => &lu.triplets,
            Self::Gmres(gmres) => &gmres.triplets,
        }
    }

    #[inline]
    pub fn triplets_mut(&mut self) -> &mut Vec<(usize, usize, f64)> {
        match self {
            Self::SparseLu(lu) => &mut lu.triplets,
            Self::Gmres(gmres) => &mut gmres.triplets,
        }
    }

    #[inline]
    pub fn is_gmres(&self) -> bool {
        matches!(self, Self::Gmres(_))
    }

    #[inline]
    pub fn set_gmres_context(
        &mut self,
        y: *const f64,
        ydot: *const f64,
        p: *const f64,
        m: *const f64,
        c_j: f64,
        id: *const f64,
        spatial_diag: *const f64,
        jvp_fn: Option<NativeJvpFn>,
    ) {
        if let Self::Gmres(gmres) = self {
            gmres.set_context(y, ydot, p, m, c_j, id, spatial_diag, jvp_fn);
        }
    }

    #[inline]
    pub fn as_sparse_lu(&self) -> Option<&NativeSparseLuSolver> {
        match self {
            Self::SparseLu(lu) => Some(lu),
            Self::Gmres(_) => None,
        }
    }

    #[inline]
    pub fn as_sparse_lu_mut(&mut self) -> Option<&mut NativeSparseLuSolver> {
        match self {
            Self::SparseLu(lu) => Some(lu),
            Self::Gmres(_) => None,
        }
    }
}
