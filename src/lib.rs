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

/// A Python module implemented in Rust.
#[pymodule]
fn minimal_diffsl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(array_test, m)?)?;
    Ok(())
}
