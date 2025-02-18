use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use matmul_rs::*; // Assuming matmul_rs contains your matrix multiplication functions

fn benchmark(c: &mut Criterion) {
    let sizes = vec![64, 512, 1024]; // Matrix sizes to benchmark
    let tile_sizes = vec![32, 64, 128]; // Tile sizes for tiling and multithreading

    let mut group = c.benchmark_group("matrix_multiplication");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &sizes {
        let left = vec![1.0; size * size];
        let right = vec![1.0; size * size];

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, &size| {
            let mut result = vec![0.0; size * size]; // Ensure fresh result buffer
            b.iter(|| naive_mat_mul(size, size, size, &left, &right, &mut result))
        });

        group.bench_with_input(BenchmarkId::new("naive_in_reg", size), &size, |b, &size| {
            let mut result = vec![0.0; size * size];
            b.iter(|| naive_mat_mul_in_reg(size, size, size, &left, &right, &mut result))
        });

        group.bench_with_input(BenchmarkId::new("good_loop_order", size), &size, |b, &size| {
            let mut result = vec![0.0; size * size];
            b.iter(|| matmul_with_good_loop_order(size, size, size, &left, &right, &mut result))
        });

        for &tile_size in &tile_sizes {
            group.bench_with_input(
                BenchmarkId::new(format!("tiling_{}_{}", size, tile_size), size),
                &size,
                |b, &size| {
                    let mut result = vec![0.0; size * size];
                    b.iter(|| matmul_tiling(size, size, size, tile_size, &left, &right, &mut result))
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("multithreaded_tiled_{}_{}", size, tile_size), size),
                &size,
                |b, &size| {
                    let mut result = vec![0.0; size * size];
                    b.iter(|| {
                        multithreaded_tiled_mat_mul(
                            size,
                            size,
                            size,
                            tile_size,
                            &left,
                            &right,
                            &mut result,
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
