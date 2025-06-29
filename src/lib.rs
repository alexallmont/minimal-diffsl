use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Experimental, test creating a 1D numpy array.
/// Only creates if n >= 10. This is arbirary whilst I get my head around framework.
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
use diffsol::{
    MatrixCommon,
    NalgebraLU, NalgebraMat,
    OdeBuilder, OdeSolverMethod, OdeEquationsImplicit, OdeSolverProblem,
    Vector
};
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;
type LS = NalgebraLU<f64>;

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

#[pyfunction]
fn make_numpy_array_from_raw<'py>(py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    let vec = vec![1.0, 2.0, 3.0];
    let arr: Array1<f64> = Array1::from(vec);
    PyArray1::from_owned_array(py, arr)
}

#[pyfunction]
pub fn solver_test<'py>(py: Python<'py>) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let problem = problem_implicit();
    let mut solver = problem.bdf::<LS>().unwrap();
    let (ys, ts): (M, Vec<T>) = solver.solve(10.0).unwrap();

    // FIXME experiment attempt to get ptr for creating Array2/PyArray2
    let shape = ys.inner().shape();
    // let ptr = ys.inner().as_ptr();
    // let len = shape.0 * shape.1;
    // let vec = unsafe { Vec::<f64>::from_raw_parts(ptr, len, len) };
    // let ys_arr = unsafe { Array2::from_shape_vec(shape, vec) };
    let ts_arr = Array1::from(ts);
    (
        //PyArray2::from_owned_array(py, ys_arr.unwrap()),
        unsafe { PyArray2::new(py, shape, false) }, // FIXME init from data
        PyArray1::from_owned_array(py, ts_arr)
    )
}

#[pymodule]
fn minimal_diffsl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(array_test, m)?)?;
    m.add_function(wrap_pyfunction!(solver_test, m)?)?;
    m.add_function(wrap_pyfunction!(make_numpy_array_from_raw, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixme() {
        pyo3::append_to_inittab!(minimal_diffsl);
        Python::with_gil(|py| {
            solver_test(py);
            // FIXME assert_eq! on results
        });
    }
}