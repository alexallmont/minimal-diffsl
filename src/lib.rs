use numpy::PyArray1;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Experimental, test creating a 1D numpy array.
/// Only creates if n > 10. This is arbirary whilst I get my head around framework.
#[pyfunction]
fn array_test<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if n >= 10 {
        let result = unsafe { PyArray1::new(py, [n], false)};
        Ok(result)
    } else {
        Err(PyTypeError::new_err("🚧"))
    }
}

// FIXME copied from diffsol examples
use diffsol::{MatrixCommon, OdeBuilder};
use diffsol::{NalgebraMat, OdeEquationsImplicit, OdeSolverProblem, Vector};
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

pub fn problem_implicit() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .p(vec![1.0, 10.0])
        .rhs_implicit(
            |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
            |x, p, _t, v, y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
        )
        .init(|_p, _t, y| y.fill(0.1), 1)
        .build()
        .unwrap()
}

/// A Python module implemented in Rust.
#[pymodule]
fn minimal_diffsl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(array_test, m)?)?;
    Ok(())
}
