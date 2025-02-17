// http://matrixmultiplication.xyz/
// If you have forgotten how ole Gilbert teacher matrix multiplication
//
// Trying to recreate this? https://siboehm.com/articles/22/Fast-MMM-on-CPU
use rayon::prelude::*;

pub fn naive_mat_mul(
    rows: usize,
    columns: usize,
    inners: usize,
    left: &Vec<f32>,
    right: &Vec<f32>,
    result: &mut Vec<f32>,
) {
    for row in 0..rows {
        for col in 0..columns {
            for inner in 0..inners {
                result[row * columns + col] +=
                    left[row * inners + inner] * right[inner * columns + col];
            }
        }
    }
}

// Apparently doing the calculation in two steps makes calculation happen in the CPU register
// itself but I don't think my Rust code is that good
pub fn naive_mat_mul_in_reg(
    rows: usize,
    columns: usize,
    inners: usize,
    left: &Vec<f32>,
    right: &Vec<f32>,
    result: &mut Vec<f32>,
) {
    for row in 0..rows {
        for col in 0..columns {
            let mut acc: f32 = 0.0;
            for inner in 0..inners {
                acc += left[row * inners + inner] * right[inner * columns + col];
            }
            result[row * columns + col] = acc;
        }
    }
}
// Fixing Loop order now to magically NOT do cache miss. Read the article if you have forgotten
pub fn matmul_with_good_loop_order(
    rows: usize,
    columns: usize,
    inners: usize,
    left: &Vec<f32>,
    right: &Vec<f32>,
    result: &mut Vec<f32>,
) {
    for row in 0..rows {
        for inner in 0..inners {
            for col in 0..columns {
                result[row * columns + col] +=
                    left[row * inners + inner] * right[inner * columns + col];
            }
        }
    }
}

pub fn matmul_tiling(
    rows: usize,
    columns: usize,
    inners: usize,
    tile_size: usize,
    left: &Vec<f32>,
    right: &Vec<f32>,
    result: &mut Vec<f32>,
) {
    for inner_tile in (0..inners).step_by(tile_size) {
        for row in 0..rows {
            let inner_tile_end: usize = std::cmp::min(inners, inner_tile + tile_size);
            for inner in inner_tile..inner_tile_end {
                for col in 0..columns {
                    result[row * columns + col] +=
                        left[row * inners + inner] * right[inner * columns + col];
                }
            }
        }
    }
}

pub fn multithreaded_tiled_mat_mul(
    rows: usize,
    columns: usize,
    inners: usize,
    tile_size: usize,
    left: &[f32],
    right: &[f32],
    result: &mut [f32],
) {
    let tile_size_outer = 256; // Same outer tiling as the C++ code
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()
        .unwrap();

    pool.install(|| {
        result
            .par_chunks_mut(tile_size_outer * columns) // Each chunk is a contiguous block of `tile_size_outer` rows
            .enumerate()
            .for_each(|(tile_row_id, result_chunk)| {
                let row_tile_start = tile_row_id * tile_size_outer;
                let row_tile_end = (row_tile_start + tile_size_outer).min(rows);

                for column_tile in (0..columns).step_by(tile_size_outer) {
                    for inner_tile in (0..inners).step_by(tile_size) {
                        for row in row_tile_start..row_tile_end {
                            let inner_tile_end = (inner_tile + tile_size).min(inners);
                            for inner in inner_tile..inner_tile_end {
                                for col in column_tile..(column_tile + tile_size_outer).min(columns)
                                {
                                    result_chunk[(row - row_tile_start) * columns + col] +=
                                        left[row * inners + inner] * right[inner * columns + col];
                                }
                            }
                        }
                    }
                }
            });
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matrix_eq(result: &[f32], expected: &[f32]) {
        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "{} != {}", a, b);
        }
    }

    fn setup_matrices() -> (usize, usize, usize, Vec<f32>, Vec<f32>, Vec<f32>) {
        let rows = 3;
        let columns = 3;
        let inners = 3;
        let left = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let right = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected = vec![30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0];
        (rows, columns, inners, left, right, expected)
    }

    #[test]
    fn test_naive_mat_mul() {
        let (rows, columns, inners, left, right, expected) = setup_matrices();

        let mut result = vec![0.0; rows * columns];
        naive_mat_mul(rows, columns, inners, &left, &right, &mut result);
        assert_matrix_eq(&result, &expected);

        result.fill(0.0);
        naive_mat_mul_in_reg(rows, columns, inners, &left, &right, &mut result);
        assert_matrix_eq(&result, &expected);

        result.fill(0.0);
        matmul_with_good_loop_order(rows, columns, inners, &left, &right, &mut result);
        assert_matrix_eq(&result, &expected);

        result.fill(0.0);
        let tile_size = 2;
        matmul_tiling(rows, columns, inners, tile_size, &left, &right, &mut result);
        assert_matrix_eq(&result, &expected);

        result.fill(0.0);
        multithreaded_tiled_mat_mul(rows, columns, inners, tile_size, &left, &right, &mut result);
        assert_matrix_eq(&result, &expected);
    }
}
