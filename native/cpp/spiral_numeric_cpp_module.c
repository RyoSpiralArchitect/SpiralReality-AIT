#include <Python.h>

static PyObject *backend_module = NULL;

static int ensure_backend(void) {
    if (backend_module) {
        return 0;
    }
    backend_module = PyImport_ImportModule("spiralreality_AIT_onepass_aifcore_integrated.integrated._python_numeric_backend");
    if (!backend_module) {
        return -1;
    }
    return 0;
}

static PyObject *call_backend(const char *name, PyObject *args) {
    if (ensure_backend() != 0) {
        return NULL;
    }
    PyObject *func = PyObject_GetAttrString(backend_module, name);
    if (!func) {
        return NULL;
    }
    PyObject *result = PyObject_CallObject(func, args);
    Py_DECREF(func);
    return result;
}

static PyObject *wrap_matmul(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("matmul", args);
}

static PyObject *wrap_dot(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("dot", args);
}

static PyObject *wrap_mean(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("mean", args);
}

static PyObject *wrap_std(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("std", args);
}

static PyObject *wrap_sum(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("sum_reduce", args);
}

static PyObject *wrap_tanh(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("tanh_map", args);
}

static PyObject *wrap_exp(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("exp_map", args);
}

static PyObject *wrap_log(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("log_map", args);
}

static PyObject *wrap_logaddexp(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("logaddexp_map", args);
}

static PyObject *wrap_median(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("median_all", args);
}

static PyObject *wrap_abs(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("abs_map", args);
}

static PyObject *wrap_clip(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("clip_map", args);
}

static PyObject *wrap_sqrt(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("sqrt_map", args);
}

static PyObject *wrap_diff(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("diff_vec", args);
}

static PyObject *wrap_argsort(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("argsort_indices", args);
}

static PyObject *wrap_argmax(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("argmax_index", args);
}

static PyObject *wrap_trace(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("trace_value", args);
}

static PyObject *wrap_norm(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("norm_value", args);
}

static PyObject *wrap_inv(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("inv_matrix", args);
}

static PyObject *wrap_slogdet(PyObject *self, PyObject *args) {
    (void)self;
    return call_backend("slogdet_pair", args);
}

static PyMethodDef module_methods[] = {
    {"matmul", wrap_matmul, METH_VARARGS, "Matrix multiplication"},
    {"dot", wrap_dot, METH_VARARGS, "Dot product"},
    {"mean", wrap_mean, METH_VARARGS, "Mean reduction"},
    {"std", wrap_std, METH_VARARGS, "Standard deviation"},
    {"sum", wrap_sum, METH_VARARGS, "Sum reduction"},
    {"tanh", wrap_tanh, METH_VARARGS, "Hyperbolic tangent"},
    {"exp", wrap_exp, METH_VARARGS, "Exponential"},
    {"log", wrap_log, METH_VARARGS, "Natural logarithm"},
    {"logaddexp", wrap_logaddexp, METH_VARARGS, "Log-add-exp"},
    {"median", wrap_median, METH_VARARGS, "Median"},
    {"abs", wrap_abs, METH_VARARGS, "Absolute value"},
    {"clip", wrap_clip, METH_VARARGS, "Clip values"},
    {"sqrt", wrap_sqrt, METH_VARARGS, "Square root"},
    {"diff", wrap_diff, METH_VARARGS, "Discrete difference"},
    {"argsort", wrap_argsort, METH_VARARGS, "Argsort"},
    {"argmax", wrap_argmax, METH_VARARGS, "Argmax"},
    {"trace", wrap_trace, METH_VARARGS, "Trace"},
    {"linalg_norm", wrap_norm, METH_VARARGS, "Vector norm"},
    {"linalg_inv", wrap_inv, METH_VARARGS, "Matrix inverse"},
    {"linalg_slogdet", wrap_slogdet, METH_VARARGS, "Sign and log-determinant"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "spiral_numeric_cpp",
    "Compiled bridge dispatching numeric helpers.",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_spiral_numeric_cpp(void) {
    return PyModule_Create(&module_def);
}
