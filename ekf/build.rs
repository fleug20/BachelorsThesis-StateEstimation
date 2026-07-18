fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_C_API");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_STATS");

    // Only generate the C header when the c_api feature is enabled — otherwise
    // the `RkfState` symbol it references is cfg'd out and cbindgen would fail.
    if std::env::var_os("CARGO_FEATURE_C_API").is_none() {
        return;
    }

    let stats_enabled = std::env::var_os("CARGO_FEATURE_STATS").is_some();

    let mut builder = cbindgen::Builder::new()
        .with_crate(".")
        .with_language(cbindgen::Language::C)
        .with_include_guard("RKF_H")
        .with_no_includes()
        .with_sys_include("stdint.h")
        .include_item("RkfState");

    // cbindgen parses raw source without evaluating cfg conditions, so it
    // always sees the stats functions.  Exclude them explicitly when the stats
    // feature is not active.
    if !stats_enabled {
        builder = builder
            .exclude_item("rkf_get_uwb_residual")
            .exclude_item("rkf_get_tof_residual")
            .exclude_item("rkf_get_uwb_adaptive_residual")
            .exclude_item("rkf_get_tof_adaptive_residual")
            .exclude_item("rkf_get_uwb_innovation_covariance")
            .exclude_item("rkf_get_tof_innovation_covariance")
            .exclude_item("rkf_get_uwb_adaptive_innovation_covariance")
            .exclude_item("rkf_get_tof_adaptive_innovation_covariance");
    }

    builder
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("rkf.h");
}
