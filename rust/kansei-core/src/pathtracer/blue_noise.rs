/// Blue noise table dimensions.
pub const BLUE_NOISE_SIZE: u32 = 128;

/// Generate a tileable blue noise texture using the void-and-cluster algorithm
/// (Ulichney 1993, sigma=1.9). Produces a progressive blue noise pattern where
/// thresholding at any value gives a well-distributed point set.
///
/// Returns `128 * 128 = 16384` float values in [0, 1).
pub fn generate_blue_noise() -> Vec<f32> {
    let size = BLUE_NOISE_SIZE as usize;
    let n = size * size;
    let sigma: f32 = 1.9;
    let sigma2x2 = 2.0 * sigma * sigma;
    let radius = (3.0 * sigma).ceil() as i32;

    let mut binary = vec![0u8; n];
    let mut energy = vec![0.0f32; n];
    let mut ranks = vec![0.0f32; n];

    // xorshift32 RNG
    let mut seed: u32 = 2463534242;
    let mut rng = || -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32) / 4294967296.0
    };

    // Add/subtract a pixel's Gaussian contribution to energy
    let update_energy =
        |energy: &mut [f32], idx: usize, sign: f32| {
            let cx = (idx % size) as i32;
            let cy = (idx / size) as i32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = ((cx + dx) % size as i32 + size as i32) as usize % size;
                    let ny = ((cy + dy) % size as i32 + size as i32) as usize % size;
                    let d2 = (dx * dx + dy * dy) as f32;
                    energy[ny * size + nx] += sign * (-d2 / sigma2x2).exp();
                }
            }
        };

    // Find pixel with binary[i]==val that has max (find_max=true) or min energy
    let find_extreme = |binary: &[u8], energy: &[f32], val: u8, find_max: bool| -> usize {
        let mut best_idx = 0;
        let mut best_e = if find_max { f32::NEG_INFINITY } else { f32::INFINITY };
        for i in 0..n {
            if binary[i] != val {
                continue;
            }
            if (find_max && energy[i] > best_e) || (!find_max && energy[i] < best_e) {
                best_e = energy[i];
                best_idx = i;
            }
        }
        best_idx
    };

    // Phase 1: Create initial binary pattern (~10% seed density)
    let initial_count = (n as f32 * 0.1).max(1.0) as usize;
    let mut ones_count = 0;
    while ones_count < initial_count {
        let idx = (rng() * n as f32) as usize % n;
        if binary[idx] != 0 {
            continue;
        }
        binary[idx] = 1;
        update_energy(&mut energy, idx, 1.0);
        ones_count += 1;
    }

    // Stabilize: move tightest cluster → largest void until convergent
    for _ in 0..n {
        let cluster_idx = find_extreme(&binary, &energy, 1, true);
        binary[cluster_idx] = 0;
        update_energy(&mut energy, cluster_idx, -1.0);

        let void_idx = find_extreme(&binary, &energy, 0, false);
        if void_idx == cluster_idx {
            binary[cluster_idx] = 1;
            update_energy(&mut energy, cluster_idx, 1.0);
            break;
        }

        binary[void_idx] = 1;
        update_energy(&mut energy, void_idx, 1.0);
    }

    // Rank initial seed points (remove tightest clusters)
    let saved_binary = binary.clone();
    let saved_energy = energy.clone();
    let mut remaining = ones_count;

    while remaining > 0 {
        let cluster_idx = find_extreme(&binary, &energy, 1, true);
        remaining -= 1;
        ranks[cluster_idx] = remaining as f32;
        binary[cluster_idx] = 0;
        update_energy(&mut energy, cluster_idx, -1.0);
    }

    // Phase 2: Fill from initial pattern to 50%
    binary.copy_from_slice(&saved_binary);
    energy.copy_from_slice(&saved_energy);
    let mut current_rank = ones_count;
    let half_n = n / 2;

    while current_rank < half_n {
        let void_idx = find_extreme(&binary, &energy, 0, false);
        ranks[void_idx] = current_rank as f32;
        binary[void_idx] = 1;
        update_energy(&mut energy, void_idx, 1.0);
        current_rank += 1;
    }

    // Phase 3: Fill from 50% to 100% (inverted energy)
    energy.fill(0.0);
    for i in 0..n {
        if binary[i] == 0 {
            update_energy(&mut energy, i, 1.0);
        }
    }

    while current_rank < n {
        let cluster_idx = find_extreme(&binary, &energy, 0, true);
        ranks[cluster_idx] = current_rank as f32;
        binary[cluster_idx] = 1;
        update_energy(&mut energy, cluster_idx, -1.0);
        current_rank += 1;
    }

    // Normalize to [0, 1)
    let n_f = n as f32;
    for r in &mut ranks {
        *r /= n_f;
    }

    ranks
}
