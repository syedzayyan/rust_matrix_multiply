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
                let left_idx = row * inners + inner;
                let right_idx = inner * columns + col;
                let result_idx = row * columns + col;

                result[result_idx] += left[left_idx] * right[right_idx];
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
            let result_idx = row * columns + col;
            for inner in 0..inners {
                let left_idx = row * inners + inner;
                let right_idx = inner * columns + col;
                acc += left[left_idx] * right[right_idx]
            }
            result[result_idx] = acc;
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
                let left_idx = row * columns + inner;
                let right_idx = inner * columns + col;
                let result_idx = row * columns + col;

                result[result_idx] += left[left_idx] * right[right_idx];
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
            let inner_tile_end = (inner_tile + tile_size).min(inners);
            for inner in inner_tile..inner_tile_end {
                for column in 0..columns {
                    result[row * column + column] +=
                        left[row * inners + inner] * right[inner * columns + column];
                }
            }
        }
    }
}
pub fn multhreaded_tiled_mat_mul(
    rows: usize,
    columns: usize,
    inners: usize,
    tile_size: usize,
    left: &[f32],
    right: &[f32],
    result: &mut [f32],
) {
    let chunk_size = 512;
    result
        .par_chunks_mut(columns * chunk_size)
        .for_each(|result_chunk| {
            for row_tile in (0..rows).step_by(chunk_size) {
                for column_tile in (0..columns).step_by(chunk_size) {
                    for inner_tile in (0..inners).step_by(tile_size) {
                        for row in row_tile..std::cmp::min(row_tile + chunk_size, rows) {
                            let inner_tile_end = std::cmp::min(inners, inner_tile + tile_size);
                            for inner in inner_tile..inner_tile_end {
                                for col in column_tile..std::cmp::min(column_tile + chunk_size, columns) {
                                    let result_index = row * columns + col;
                                    result_chunk[result_index % (columns * chunk_size)] +=
                                        left[row * inners + inner] * right[inner * columns + col];
                                }
                            }
                        }
                    }
                }
            }
        });
}
