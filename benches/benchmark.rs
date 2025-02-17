use criterion::{criterion_group, criterion_main, Criterion, PlotConfiguration, AxisScale, BenchmarkId};
use matmul_rs::*;

fn benchmark(c: &mut Criterion) {
    let sizes = vec![64, 128, 256, 512, 1024];
    let tile_size = 128;
    
    let mut group = c.benchmark_group("matrix_multiplication");
    group.plot_config(PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic));
    
    for size in sizes {
        let left: Vec<f32> = vec![1.0; size * size];
        let right: Vec<f32> = vec![1.0; size * size];
        let mut result = vec![0.0; size * size];
        
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, &size| {
            b.iter(|| naive_mat_mul(size, size, size, &left, &right, &mut result))
        });
        
        group.bench_with_input(BenchmarkId::new("naive_in_reg", size), &size, |b, &size| {
            b.iter(|| naive_mat_mul_in_reg(size, size, size, &left, &right, &mut result))
        });
        
        group.bench_with_input(BenchmarkId::new("good_loop_order", size), &size, |b, &size| {
            b.iter(|| matmul_with_good_loop_order(size, size, size, &left, &right, &mut result))
        });
        
        group.bench_with_input(BenchmarkId::new("tiling", size), &size, |b, &size| {
            b.iter(|| matmul_tiling(size, size, size, tile_size, &left, &right, &mut result))
        });
        
        group.bench_with_input(BenchmarkId::new("multithreaded_tiled", size), &size, |b, &size| {
            b.iter(|| multhreaded_tiled_mat_mul(size, size, size, tile_size, &left, &right, &mut result))
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
