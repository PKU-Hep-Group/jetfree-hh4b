#!/scratchfs/cms/tyyang99/miniforge3/bin/python

import os
import sys
import time
import argparse
import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit
from scipy import ndimage

def double_crystal_ball_1d(x, mean, sigma, alpha_low, n_low, alpha_high, n_high):
    """
    Correct implementation of 1D Double Crystal Ball function with continuity
    """
    t = (x - mean) / sigma
    result = np.zeros_like(t, dtype=float)
    
    # Calculate correct B parameters to ensure continuity
    B_low = np.exp(0.5 * alpha_low**2 / n_low) - alpha_low
    B_high = np.exp(0.5 * alpha_high**2 / n_high) - alpha_high
    
    # Central Gaussian region
    mask_central = ((-alpha_low <= t) & (t <= alpha_high))
    result[mask_central] = np.exp(-0.5 * t[mask_central]**2)
    
    # Low tail (t < -alpha_low)
    mask_low = (t < -alpha_low)
    if np.any(mask_low):
        # Gaussian value at t = -alpha_low
        gaussian_at_switch = np.exp(-0.5 * alpha_low**2)
        # Power law function, ensuring continuity at switch point
        result[mask_low] = gaussian_at_switch * \
                          (B_low + alpha_low)**n_low * \
                          (B_low - t[mask_low])**(-n_low)
    
    # High tail (t > alpha_high)  
    mask_high = (t > alpha_high)
    if np.any(mask_high):
        # Gaussian value at t = alpha_high
        gaussian_at_switch = np.exp(-0.5 * alpha_high**2)
        # Power law function, ensuring continuity at switch point
        result[mask_high] = gaussian_at_switch * \
                           (B_high + alpha_high)**n_high * \
                           (B_high + t[mask_high])**(-n_high)
    
    return result


def bivariate_double_crystal_ball(xy_mesh, amplitude, x_mean, y_mean, x_sigma, y_sigma, theta, 
                                 alpha_low_x, n_low_x, alpha_high_x, n_high_x,
                                 alpha_low_y, n_low_y, alpha_high_y, n_high_y):
    """
    2D Double Crystal Ball function with rotation and asymmetric tails
    """
    x, y = xy_mesh
    
    # Rotate coordinates
    x_rot = (x - x_mean) * np.cos(theta) + (y - y_mean) * np.sin(theta)
    y_rot = -(x - x_mean) * np.sin(theta) + (y - y_mean) * np.cos(theta)
    
    # Apply 1D Double Crystal Ball in each rotated dimension
    x_component = double_crystal_ball_1d(x_rot, 0, x_sigma, alpha_low_x, n_low_x, alpha_high_x, n_high_x)
    y_component = double_crystal_ball_1d(y_rot, 0, y_sigma, alpha_low_y, n_low_y, alpha_high_y, n_high_y)
    
    # Combine components (product of distributions)
    return amplitude * x_component * y_component

def background_2d_new(xy_mesh, amp, x_center, y_center, x_slope, y_slope, x_log, y_log):
    """
    New flexible 2D background function with arbitrary slope directions
    """
    x, y = xy_mesh
    
    # Use linear + logarithmic combination with center as reference
    x_component = 1.0 + x_slope * (x - x_center) + x_log * np.log(x/x_center)
    y_component = 1.0 + y_slope * (y - y_center) + y_log * np.log(y/y_center)
    
    # Ensure non-negative
    result = amp * np.maximum(x_component * y_component, 1e-10)
    return result

def symmetric_background_pair_new(xy_mesh, amp, x_center, y_center, x_slope, y_slope, x_log, y_log):
    """
    New symmetric pair of 2D background functions
    Creates f(x,y) + f(y,x) which is symmetric about the diagonal y=x
    """
    x, y = xy_mesh
    
    # Original background: f(x,y)
    bg1 = background_2d_new((x, y), amp, x_center, y_center, x_slope, y_slope, x_log, y_log)
    
    # Mirrored background: f(y,x) - swap coordinates and parameters appropriately
    bg2 = background_2d_new((y, x), amp, y_center, x_center, y_slope, x_slope, y_log, x_log)
    
    return bg1 + bg2


def single_peak_with_mirror(xy, *params):
    """
    Single DCB peak with its mirror
    """
    amp, x_mean, y_mean, x_sigma, y_sigma, theta, alpha_low, n_low, alpha_high, n_high = params
    x, y = xy
    
    # Main peak
    peak = bivariate_double_crystal_ball(
        (x, y), amp, x_mean, y_mean, x_sigma, y_sigma, theta,
        alpha_low, n_low, alpha_high, n_high, alpha_low, n_low, alpha_high, n_high
    )
    
    # Mirror peak
    mirror = bivariate_double_crystal_ball(
        (x, y), amp, y_mean, x_mean, y_sigma, x_sigma, -theta,
        alpha_low, n_low, alpha_high, n_high, alpha_low, n_low, alpha_high, n_high
    )
    
    return peak + mirror

def peak_plus_background_with_mirror(xy, *params):
    """
    Combined model: DCB peak pair + symmetric background pair (new version)
    """
    # Split parameters
    # DCB parameters: amp, x_mean, y_mean, x_sigma, y_sigma, theta, alpha_low, n_low, alpha_high, n_high (10 params)
    # Background parameters: bg_amp, x_center, y_center, x_slope, y_slope, x_log, y_log (7 params)
    
    dcb_params = params[:10]
    bg_params = params[10:]
    
    # DCB component
    dcb_component = single_peak_with_mirror(xy, *dcb_params)
    
    # Background component (new version)
    bg_component = symmetric_background_pair_new(xy, *bg_params)
    
    return dcb_component + bg_component

def extract_full_histogram_data(scores, bins):
    """
    Extract data for the full symmetric histogram from triangular scores
    """
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    n_bins = len(bin_centers)
    
    # Create empty histogram
    H = np.zeros((n_bins, n_bins))
    
    # Fill triangular region
    idx = 0
    for i in range(n_bins):
        for j in range(i, n_bins):
            H[i, j] = scores[idx]
            idx += 1
    
    # Create symmetric histogram
    H_sym = (H + H.T) / 2
    
    return H_sym, bin_centers

def fit_symmetric_four_dcb_staged(scores, bins):
    """
    Staged fitting strategy: first fit peak+background, then fit secondary peak
    Uses new flexible background function
    """
    # Extract full symmetric histogram
    H_sym, bin_centers = extract_full_histogram_data(scores, bins)
    
    # Normalize the histogram
    H_sym_norm = H_sym / np.sum(H_sym)
    
    # Create mesh grid for fitting
    X, Y = np.meshgrid(bin_centers, bin_centers)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = H_sym_norm.flatten()
    
    # Find the maximum value point (main peak)
    max_idx = np.argmax(H_sym_norm)
    max_i, max_j = np.unravel_index(max_idx, H_sym_norm.shape)
    max_x, max_y = bin_centers[max_i], bin_centers[max_j]
    max_val = H_sym_norm[max_i, max_j]
    
    # Ensure peak is in the upper triangle region (x ≤ y)
    if max_x > max_y:
        max_x, max_y = max_y, max_x
    
    # Calculate center of the data range for background reference
    x_center = (bin_centers[0] + bin_centers[-1]) / 2
    y_center = (bin_centers[0] + bin_centers[-1]) / 2
    
    # Stage 1: Fit peak + background model
    
    # Initial parameters for peak + background (new version)
    # DCB parameters: amp, x_mean, y_mean, x_sigma, y_sigma, theta, alpha_low, n_low, alpha_high, n_high
    # Background parameters: bg_amp, x_center, y_center, x_slope, y_slope, x_log, y_log
    initial_guess_peak_bg = [
        max_val * 0.7, max_x, max_y, 15.0, 15.0, 0.0,  # DCB parameters
        1.5, 3.0, 1.5, 3.0,                            # DCB tail parameters
        max_val * 0.3, x_center, y_center, 0.0, 0.0, 0.0, 0.0  # Background parameters (new)
    ]
    
    # Parameter bounds (updated for new background function)
    lower_bounds_peak_bg = [
        0.0, 0.0, 0.0, 2.5, 2.5, -np.pi/4,  # DCB bounds
        0.5, 1.5, 0.5, 1.5,                  # DCB tail bounds
        0.0, 40.0, 40.0, -0.01, -0.01, -0.5, -0.5  # Background bounds (new)
    ]
    
    upper_bounds_peak_bg = [
        np.inf, np.inf, np.inf, 50.0, 50.0, np.pi/4,  # DCB bounds (reduced sigma limit)
        3.0, 10.0, 3.0, 10.0,                         # DCB tail bounds
        np.inf, 200.0, 200.0, 0.01, 0.01, 0.5, 0.5   # Background bounds (new)
    ]
    
    bounds_peak_bg = (lower_bounds_peak_bg, upper_bounds_peak_bg)
    
    try:
        # Fit peak + background
        params_peak_bg, cov_peak_bg = curve_fit(
            peak_plus_background_with_mirror, 
            (x_flat, y_flat), 
            z_flat, 
            p0=initial_guess_peak_bg,
            bounds=bounds_peak_bg,
            maxfev=15000,
            method='trf'
        )
        
        peak1_success = True
        # # print("Peak + background fit successful")
    except (RuntimeError, ValueError) as e:
        peak1_success = False
        params_peak_bg = np.array(initial_guess_peak_bg)
        # # print(f"Peak + background fit failed: {e}")
    
    # Calculate peak + background model
    peak_bg_model_flat = peak_plus_background_with_mirror((x_flat, y_flat), *params_peak_bg)
    peak_bg_model = peak_bg_model_flat.reshape(X.shape)
    
    # Calculate residual (original data minus peak + background model)
    residual_flat = z_flat - peak_bg_model_flat
    residual = residual_flat.reshape(X.shape)
    
    # Extract parameters from peak + background fit
    dcb_params = params_peak_bg[:10]
    bg_params = params_peak_bg[10:]
    
    amp1, x_mean1, y_mean1, x_sigma1, y_sigma1, theta1, alpha_low1, n_low1, alpha_high1, n_high1 = dcb_params
    
    # Create mask for finding secondary peak
    mask = np.ones_like(residual, dtype=bool)
    mask_radius = max(15, 2.0 * max(x_sigma1, y_sigma1))
    
    # Mask area around main peak and its mirror
    for i in range(len(bin_centers)):
        for j in range(len(bin_centers)):
            # Calculate distance to main peak and its mirror
            dist1 = np.sqrt((bin_centers[i] - x_mean1)**2 + (bin_centers[j] - y_mean1)**2)
            dist2 = np.sqrt((bin_centers[i] - y_mean1)**2 + (bin_centers[j] - x_mean1)**2)
            if dist1 < mask_radius or dist2 < mask_radius:
                mask[i, j] = False
    
    # Apply mask to residual
    masked_residual = residual.copy()
    masked_residual[~mask] = 0
    
    # Find secondary peaks
    neighborhood_size = 2
    local_max = ndimage.maximum_filter(masked_residual, size=neighborhood_size)
    threshold = max(0.02 * max_val, 0.01 * np.max(masked_residual))
    local_max_mask = (masked_residual == local_max) & (masked_residual > threshold)
    
    # Find coordinates of local maxima
    local_max_coords = np.where(local_max_mask)
    
    if len(local_max_coords[0]) > 0:
        # Get all local maxima with their values
        candidates = []
        for idx in range(len(local_max_coords[0])):
            i, j = local_max_coords[0][idx], local_max_coords[1][idx]
            x_pos, y_pos = bin_centers[i], bin_centers[j]
            val = masked_residual[i, j]
            candidates.append((x_pos, y_pos, val, i, j))
        
        # Sort by value (strongest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Take the strongest candidate
        max2_x, max2_y, max2_val, max2_i, max2_j = candidates[0]
        secondary_peak_found = True
        
        # # print(f"Found {len(candidates)} potential secondary peaks, strongest at ({max2_x:.1f}, {max2_y:.1f}) with value {max2_val:.4f}")
        
    else:
        max2_x, max2_y = 500.0, 500.0
        max2_val = 0.0
        secondary_peak_found = False
        # # print("No secondary peak found in residual")
    
    # Stage 2: Fit secondary peak if found
    if secondary_peak_found:
        lower_bounds_peak2 = [
            0.0, 0.0, 0.0, 1.0, 1.0, -np.pi/2,
            0.3, 1.0, 0.3, 1.0
        ]
        
        upper_bounds_peak2 = [
            np.inf, np.inf, np.inf, 300.0, 300.0, np.pi/2,
            5.0, 15.0, 5.0, 15.0
        ]
        
        bounds_peak2 = (lower_bounds_peak2, upper_bounds_peak2)
        
        # Try multiple strategies for better fitting
        best_fit = None
        best_residual_sum = np.inf
        
        sigma_candidates = [(5.0, 5.0), (10.0, 10.0), (15.0, 15.0), (20.0, 20.0)]
        
        for x_sig, y_sig in sigma_candidates:
            initial_guess_peak2 = [
                max2_val * 0.8, max2_x, max2_y, x_sig, y_sig, 0.0,
                1.5, 3.0, 1.5, 3.0
            ]
            
            try:
                params, _ = curve_fit(
                    single_peak_with_mirror, 
                    (x_flat, y_flat), 
                    residual_flat, 
                    p0=initial_guess_peak2,
                    bounds=bounds_peak2,
                    maxfev=8000,
                    method='trf'
                )
                
                model = single_peak_with_mirror((x_flat, y_flat), *params)
                current_residual = np.sum((residual_flat - model)**2)
                
                if current_residual < best_residual_sum:
                    best_residual_sum = current_residual
                    best_fit = params
                    
            except (RuntimeError, ValueError):
                continue
        
        if best_fit is not None:
            params_peak2 = best_fit
            peak2_success = True
            # # print("Secondary peak fit successful")
        else:
            params_peak2 = np.array([
                0.0, 500.0, 500.0, 15.0, 15.0, 0.0,
                1.5, 3.0, 1.5, 3.0
            ])
            peak2_success = False
            # # print("Secondary peak fitting failed")
            
    else:
        params_peak2 = np.array([
            0.0, 500.0, 500.0, 15.0, 15.0, 0.0,
            1.5, 3.0, 1.5, 3.0
        ])
        peak2_success = False
    
    # Check peak1 and peak2 positions, swap if needed
    amp1, x_mean1, y_mean1, x_sigma1, y_sigma1, theta1, alpha_low1, n_low1, alpha_high1, n_high1 = dcb_params
    amp2, x_mean2, y_mean2, x_sigma2, y_sigma2, theta2, alpha_low2, n_low2, alpha_high2, n_high2 = params_peak2
    
    # Check if peak1 is in valid range
    peak1_in_range = (40 <= x_mean1 <= 200) and (40 <= y_mean1 <= 200)
    peak2_in_range = (40 <= x_mean2 <= 200) and (40 <= y_mean2 <= 200)

    # peak1_in_range = (20 <= x_mean1 <= 220) and (20 <= y_mean1 <= 220)
    # peak2_in_range = (20 <= x_mean2 <= 220) and (20 <= y_mean2 <= 220)
    
    peaks_swapped = False
    
    if not peak1_in_range and peak2_in_range and peak2_success:
        # # print(f"Peak1 at ({x_mean1:.1f}, {y_mean1:.1f}) is out of range, Peak2 at ({x_mean2:.1f}, {y_mean2:.1f}) is in range. Checking swap feasibility...")
    
        valid_range_mask = ((X >= 40) & (X <= 200) & (Y >= 40) & (Y <= 200)).flatten()
        
        current_peak1_model = single_peak_with_mirror((x_flat, y_flat), *dcb_params)
        current_peak1_error = np.sum((z_flat[valid_range_mask] - current_peak1_model[valid_range_mask])**2)
        
        temp_peak1_model = single_peak_with_mirror((x_flat, y_flat), *params_peak2)
        temp_peak1_error = np.sum((z_flat[valid_range_mask] - temp_peak1_model[valid_range_mask])**2)
        
        swap_acceptable = temp_peak1_error < current_peak1_error
        
        # # print(f"Current peak1 fitting error in valid range: {current_peak1_error:.6f}")
        # # print(f"Swap candidate peak1 fitting error in valid range: {temp_peak1_error:.6f}")
        # # print(f"Swap acceptable (better fit): {swap_acceptable}")
        
        if swap_acceptable:
            # print("Proceeding with peak swap...")
            
            dcb_params, params_peak2 = params_peak2, dcb_params
            amp1, x_mean1, y_mean1, x_sigma1, y_sigma1, theta1, alpha_low1, n_low1, alpha_high1, n_high1 = dcb_params
            amp2, x_mean2, y_mean2, x_sigma2, y_sigma2, theta2, alpha_low2, n_low2, alpha_high2, n_high2 = params_peak2
            peaks_swapped = True
        
            # # Swap peak parameters
            # dcb_params, params_peak2 = params_peak2, dcb_params
            # amp1, x_mean1, y_mean1, x_sigma1, y_sigma1, theta1, alpha_low1, n_low1, alpha_high1, n_high1 = dcb_params
            # amp2, x_mean2, y_mean2, x_sigma2, y_sigma2, theta2, alpha_low2, n_low2, alpha_high2, n_high2 = params_peak2
            # peaks_swapped = True
            
            # Fine-tune peak1 amplitude and background parameters with new peak1
            def peak1_plus_background_adjustable(xy, amp1_new, bg_amp_new, bg_x_center_new, bg_y_center_new, 
                                               bg_x_slope_new, bg_y_slope_new, bg_x_log_new, bg_y_log_new):
                peak1_params_adjustable = [amp1_new] + list(dcb_params[1:])
                peak_component = single_peak_with_mirror(xy, *peak1_params_adjustable)
                bg_component = symmetric_background_pair_new(xy, bg_amp_new, bg_x_center_new, bg_y_center_new,
                                                            bg_x_slope_new, bg_y_slope_new, bg_x_log_new, bg_y_log_new)
                return peak_component + bg_component
            
            finetune_initial = [amp1] + list(bg_params)
            
            finetune_lower = [
                0.0,  # amp1 lower bound
                0.0, 40.0, 40.0, -0.01, -0.01, -0.5, -0.5  # background bounds
            ]
            
            finetune_upper = [
                np.inf,  # amp1 upper bound  
                np.inf, 200.0, 200.0, 0.01, 0.01, 0.5, 0.5  # background bounds
            ]
            
            finetune_bounds = (finetune_lower, finetune_upper)
            
            try:
                optimized_params, _ = curve_fit(
                    peak1_plus_background_adjustable,
                    (x_flat, y_flat),
                    z_flat,
                    p0=finetune_initial,
                    bounds=finetune_bounds,
                    maxfev=5000,
                    method='trf'
                )
                
                amp1 = optimized_params[0]
                bg_params = optimized_params[1:]
                dcb_params[0] = amp1
                
                # # print("Peak1 amplitude and background fine-tuning after peak swap successful")
                
            except (RuntimeError, ValueError) as e:
                pass
                # print(f"Peak1 amplitude and background fine-tuning failed: {e}, keeping original parameters")
            
            # Recalculate residual for second peak2 fitting
            updated_params_peak_bg = list(dcb_params) + list(bg_params)
            peak_bg_model_flat = peak_plus_background_with_mirror((x_flat, y_flat), *updated_params_peak_bg)
            residual_flat = z_flat - peak_bg_model_flat
            residual = residual_flat.reshape(X.shape)
            
            # Recreate mask for second peak2 fitting
            mask = np.ones_like(residual, dtype=bool)
            mask_radius = max(15, 2.0 * max(x_sigma1, y_sigma1))
            
            for i in range(len(bin_centers)):
                for j in range(len(bin_centers)):
                    dist1 = np.sqrt((bin_centers[i] - x_mean1)**2 + (bin_centers[j] - y_mean1)**2)
                    dist2 = np.sqrt((bin_centers[i] - y_mean1)**2 + (bin_centers[j] - x_mean1)**2)
                    if dist1 < mask_radius or dist2 < mask_radius:
                        mask[i, j] = False
            
            masked_residual = residual.copy()
            masked_residual[~mask] = 0
            
            # Re-find second peak
            local_max = ndimage.maximum_filter(masked_residual, size=neighborhood_size)
            threshold = max(0.02 * max_val, 0.01 * np.max(masked_residual))
            local_max_mask = (masked_residual == local_max) & (masked_residual > threshold)
            local_max_coords = np.where(local_max_mask)
            
            if len(local_max_coords[0]) > 0:
                candidates = []
                for idx in range(len(local_max_coords[0])):
                    i, j = local_max_coords[0][idx], local_max_coords[1][idx]
                    x_pos, y_pos = bin_centers[i], bin_centers[j]
                    val = masked_residual[i, j]
                    candidates.append((x_pos, y_pos, val, i, j))
                
                candidates.sort(key=lambda x: x[2], reverse=True)
                max2_x, max2_y, max2_val, max2_i, max2_j = candidates[0]
                
                # Re-fit peak2
                best_fit = None
                best_residual_sum = np.inf
                
                for x_sig, y_sig in sigma_candidates:
                    initial_guess_peak2 = [
                        max2_val * 0.8, max2_x, max2_y, x_sig, y_sig, 0.0,
                        1.5, 3.0, 1.5, 3.0
                    ]
                    
                    try:
                        params, _ = curve_fit(
                            single_peak_with_mirror, 
                            (x_flat, y_flat), 
                            residual_flat, 
                            p0=initial_guess_peak2,
                            bounds=bounds_peak2,
                            maxfev=8000,
                            method='trf'
                        )
                        
                        model = single_peak_with_mirror((x_flat, y_flat), *params)
                        current_residual = np.sum((residual_flat - model)**2)
                        
                        if current_residual < best_residual_sum:
                            best_residual_sum = current_residual
                            best_fit = params
                            
                    except (RuntimeError, ValueError):
                        continue
                
                if best_fit is not None:
                    params_peak2 = best_fit
                    peak2_success = True
                    # print("Secondary peak re-fitting after swap successful")
                else:
                    params_peak2 = np.array([
                        0.0, 500.0, 500.0, 15.0, 15.0, 0.0,
                        1.5, 3.0, 1.5, 3.0
                    ])
                    peak2_success = False
                    # print("Secondary peak re-fitting after swap failed")
            else:
                params_peak2 = np.array([
                    0.0, 500.0, 500.0, 15.0, 15.0, 0.0,
                    1.5, 3.0, 1.5, 3.0
                ])
                peak2_success = False
                # print("No secondary peak found after swap")
        else:
            # print("Swap rejected due to high residual. Keeping original peak assignment.")
            peaks_swapped = False
    
    # Re-parse parameters (possibly swapped)
    amp1, x_mean1, y_mean1, x_sigma1, y_sigma1, theta1, alpha_low1, n_low1, alpha_high1, n_high1 = dcb_params
    amp2, x_mean2, y_mean2, x_sigma2, y_sigma2, theta2, alpha_low2, n_low2, alpha_high2, n_high2 = params_peak2
    
    # Normalize amplitudes
    if peak2_success and x_mean2 < 400:
        total_amp = np.sum(single_peak_with_mirror((x_flat, y_flat), *dcb_params)) + \
                   np.sum(single_peak_with_mirror((x_flat, y_flat), *params_peak2))
        scale_factor = 1.0 / total_amp if total_amp > 0 else 1.0
        amp1 *= scale_factor
        amp2 *= scale_factor
        bg_params[0] *= scale_factor
    else:
        total_amp = np.sum(single_peak_with_mirror((x_flat, y_flat), *dcb_params))
        scale_factor = 1.0 / total_amp if total_amp > 0 else 1.0
        amp1 *= scale_factor
        amp2 = 0.0
        bg_params[0] *= scale_factor
    
    # Update parameter arrays
    dcb_params[0] = amp1
    params_peak2[0] = amp2
    
    # Combine all parameters including background
    params = np.array([
        amp1, x_mean1, y_mean1, x_sigma1, y_sigma1, theta1,
        alpha_low1, n_low1, alpha_high1, n_high1,
        amp2, x_mean2, y_mean2, x_sigma2, y_sigma2, theta2,
        alpha_low2, n_low2, alpha_high2, n_high2
    ] + list(bg_params))
    
    # Calculate metrics
    if x_mean2 < 400 and y_mean2 < 400:
        distance = np.sqrt((x_mean1 - x_mean2)**2 + (y_mean1 - y_mean2)**2)
        amplitude_ratio = amp1 / amp2 if amp2 > 0 else float('inf')
    else:
        distance = float('inf')
        amplitude_ratio = float('inf')
    
    dominant_peak = amplitude_ratio > 10.0
    
    # Create metrics dictionary
    metrics = {
        'distance': distance,
        'amplitude_ratio': amplitude_ratio,
        'dominant_peak': dominant_peak,
        'peak1_success': peak1_success,
        'peak2_success': peak2_success,
        'secondary_peak_found': secondary_peak_found,
        'valid_secondary': x_mean2 < 400 and y_mean2 < 400,
        'background_params': bg_params,
        'peaks_swapped': peaks_swapped
    }
    
    fit_success = peak1_success
    
    return params, fit_success, metrics

def fit_symmetric_four_dcb(scores, bins):
    """
    Main fitting function
    """
    params, fit_success, metrics = fit_symmetric_four_dcb_staged(scores, bins)
    
    return params, fit_success, metrics

def process_events_range(file_path, start_event, end_event, cut_value=0.8):
    """
    Process a specific range of events from a ROOT file
    """
    start_time = time.time()
    
    try:
        # Open ROOT file and load data with selections
        with uproot.open(file_path) as file:
            tree = file["Events"]
            
            # First load selection variables and apply selections
            selection_branches = ["pass_selection", "pass_4j3b_selection"] + [f"score_{j}" for j in range(138)]
            data = tree.arrays(selection_branches, library="np")
            
            # Apply initial selection
            selection_mask = (data["pass_selection"] == 1) & (data["pass_4j3b_selection"] == 1)
            selected_indices = np.where(selection_mask)[0]
            
            if len(selected_indices) == 0:
                print("No events pass the initial selection")
                return None
            
            # Calculate score_hhvsboth
            scores_ALLHH4b = np.zeros_like(data["score_0"][selection_mask])
            _scaledprobs = np.zeros((len(selected_indices), 136))
            
            for j in range(136):
                branch_name = f"score_{j}"
                scores_ALLHH4b += data[branch_name][selection_mask]
                _scaledprobs[:, j] = data[branch_name][selection_mask]
            
            score_hhvsboth = scores_ALLHH4b / (scores_ALLHH4b + 
                                             data["score_136"][selection_mask] + 
                                             data["score_137"][selection_mask])
            
            # Apply score cut
            score_cut_mask = (score_hhvsboth > cut_value)
            final_indices = selected_indices[score_cut_mask]
            
            print(f"Rate for hhvsboth > {cut_value}: {np.sum(score_cut_mask)/len(score_hhvsboth):.4f}")
            print(f"Total events passing all selections: {np.sum(score_cut_mask)}")
            
            if len(final_indices) == 0:
                print("No events pass the score cut")
                return None
            
            # Apply event range filter
            if start_event >= len(final_indices):
                print(f"Start event {start_event} is beyond available events ({len(final_indices)})")
                return None
            
            # Adjust end_event if it's beyond available events
            actual_end_event = min(end_event, len(final_indices))
            
            if start_event >= actual_end_event:
                print(f"No events in range [{start_event}, {actual_end_event})")
                return None
            
            # Select the specific range of events
            range_indices = np.arange(start_event, actual_end_event)
            actual_file_indices = final_indices[range_indices]
            
            # Get the scores for selected events
            selected_scores = _scaledprobs[score_cut_mask][range_indices]
            selected_score_hhvsboth = score_hhvsboth[score_cut_mask][range_indices]
            
            print(f"Processing events {start_event} to {actual_end_event-1} ({len(range_indices)} events)")
    
    except Exception as e:
        print(f"Error processing ROOT file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Define bin edges
    bins = np.arange(40, 201, 10)
    
    # Initialize result arrays (now with new background parameters)
    n_results = len(range_indices)
    
    # Peak 1 parameters
    p1_amp = np.zeros(n_results)
    p1_x_mean = np.zeros(n_results)
    p1_y_mean = np.zeros(n_results)
    p1_x_sigma = np.zeros(n_results)
    p1_y_sigma = np.zeros(n_results)
    p1_theta = np.zeros(n_results)
    p1_alpha_low = np.zeros(n_results)
    p1_n_low = np.zeros(n_results)
    p1_alpha_high = np.zeros(n_results)
    p1_n_high = np.zeros(n_results)
    
    # Peak 2 parameters
    p2_amp = np.zeros(n_results)
    p2_x_mean = np.zeros(n_results)
    p2_y_mean = np.zeros(n_results)
    p2_x_sigma = np.zeros(n_results)
    p2_y_sigma = np.zeros(n_results)
    p2_theta = np.zeros(n_results)
    p2_alpha_low = np.zeros(n_results)
    p2_n_low = np.zeros(n_results)
    p2_alpha_high = np.zeros(n_results)
    p2_n_high = np.zeros(n_results)
    
    # Background parameters (new format)
    bg_amp = np.zeros(n_results)
    bg_x_center = np.zeros(n_results)
    bg_y_center = np.zeros(n_results)
    bg_x_slope = np.zeros(n_results)
    bg_y_slope = np.zeros(n_results)
    bg_x_log = np.zeros(n_results)
    bg_y_log = np.zeros(n_results)
    
    # Metrics
    distance = np.zeros(n_results)
    amplitude_ratio = np.zeros(n_results)
    dominant_peak = np.zeros(n_results, dtype=bool)
    fit_success = np.zeros(n_results, dtype=bool)
    
    # Process each event
    for i, event_idx in enumerate(range(len(selected_scores))):
        try:
            # Extract scores for the specific event
            event_scores = selected_scores[event_idx]
            
            # Perform symmetric four-DCB fitting with new background
            params, success, metrics = fit_symmetric_four_dcb(event_scores, bins)
            
            # Store results
            # Peak 1
            p1_amp[i] = params[0]
            p1_x_mean[i] = params[1]
            p1_y_mean[i] = params[2]
            p1_x_sigma[i] = params[3]
            p1_y_sigma[i] = params[4]
            p1_theta[i] = params[5]
            p1_alpha_low[i] = params[6]
            p1_n_low[i] = params[7]
            p1_alpha_high[i] = params[8]
            p1_n_high[i] = params[9]
            
            # Peak 2
            p2_amp[i] = params[10]
            p2_x_mean[i] = params[11]
            p2_y_mean[i] = params[12]
            p2_x_sigma[i] = params[13]
            p2_y_sigma[i] = params[14]
            p2_theta[i] = params[15]
            p2_alpha_low[i] = params[16]
            p2_n_low[i] = params[17]
            p2_alpha_high[i] = params[18]
            p2_n_high[i] = params[19]
            
            # Background parameters (new format)
            bg_amp[i] = params[20]
            bg_x_center[i] = params[21]
            bg_y_center[i] = params[22]
            bg_x_slope[i] = params[23]
            bg_y_slope[i] = params[24]
            bg_x_log[i] = params[25]
            bg_y_log[i] = params[26]
            
            # Metrics
            distance[i] = metrics['distance']
            amplitude_ratio[i] = metrics['amplitude_ratio']
            dominant_peak[i] = metrics['dominant_peak']
            fit_success[i] = success
            
            if (i+1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i+1}/{len(selected_scores)} events, elapsed time: {elapsed:.2f}s")
                
        except Exception as e:
            print(f"Error processing event {event_idx}: {e}")
            # Keep zeros for this event
    
    # Create result dictionary
    results = {
        'event_indices': actual_file_indices,
        'start_event': start_event,
        'end_event': actual_end_event,
        
        # Peak 1 parameters
        'p1_amp': p1_amp,
        'p1_x_mean': p1_x_mean,
        'p1_y_mean': p1_y_mean,
        'p1_x_sigma': p1_x_sigma,
        'p1_y_sigma': p1_y_sigma,
        'p1_theta': p1_theta,
        'p1_alpha_low': p1_alpha_low,
        'p1_n_low': p1_n_low,
        'p1_alpha_high': p1_alpha_high,
        'p1_n_high': p1_n_high,
        
        # Peak 2 parameters
        'p2_amp': p2_amp,
        'p2_x_mean': p2_x_mean,
        'p2_y_mean': p2_y_mean,
        'p2_x_sigma': p2_x_sigma,
        'p2_y_sigma': p2_y_sigma,
        'p2_theta': p2_theta,
        'p2_alpha_low': p2_alpha_low,
        'p2_n_low': p2_n_low,
        'p2_alpha_high': p2_alpha_high,
        'p2_n_high': p2_n_high,
        
        # Background parameters (new format)
        'bg_amp': bg_amp,
        'bg_x_center': bg_x_center,
        'bg_y_center': bg_y_center,
        'bg_x_slope': bg_x_slope,
        'bg_y_slope': bg_y_slope,
        'bg_x_log': bg_x_log,
        'bg_y_log': bg_y_log,
        
        # Metrics
        'distance': distance,
        'amplitude_ratio': amplitude_ratio,
        'dominant_peak': dominant_peak,
        'fit_success': fit_success,
        
        # Additional information
        'score_hhvsboth': selected_score_hhvsboth
    }
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    return results

def save_results(results, output_file):
    """
    Save results to a numpy file
    """
    np.savez(output_file, **results)
    print(f"Results saved to {output_file}")

def count_events_in_file(file_path, cut_value=0.8):
    """
    Count the number of events in a ROOT file that pass all selections
    """
    try:
        with uproot.open(file_path) as file:
            tree = file["Events"]
            
            # Load selection variables
            selection_branches = ["pass_selection", "pass_4j3b_selection"] + [f"score_{j}" for j in range(138)]
            data = tree.arrays(selection_branches, library="np")
            
            # Apply initial selection
            selection_mask = (data["pass_selection"] == 1) & (data["pass_4j3b_selection"] == 1)
            selected_indices = np.where(selection_mask)[0]
            
            if len(selected_indices) == 0:
                print("No events pass the initial selection")
                return 0
            
            # Calculate score_hhvsboth
            scores_ALLHH4b = np.zeros_like(data["score_0"][selection_mask])
            
            for j in range(136):
                branch_name = f"score_{j}"
                scores_ALLHH4b += data[branch_name][selection_mask]
            
            score_hhvsboth = scores_ALLHH4b / (scores_ALLHH4b + 
                                             data["score_136"][selection_mask] + 
                                             data["score_137"][selection_mask] )
            
            # Apply score cut
            score_cut_mask = (score_hhvsboth > cut_value)
            final_indices = selected_indices[score_cut_mask]
            
            print(f"Rate for hhvsboth > {cut_value}: {np.sum(score_cut_mask)/len(score_hhvsboth):.4f}")
            print(f"Number of events passing all selections: {len(final_indices)}")
            
            return len(final_indices)
    
    except Exception as e:
        print(f"Error processing ROOT file: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    """Main function to handle command line arguments and process single job"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Single job ROOT file processing for DCB+New Background fitting')
    parser.add_argument('--input', '-i', required=True, help='Input ROOT file path')
    parser.add_argument('--output-dir', '-o', default='dcb_fit_results_new_bg', help='Output directory')
    parser.add_argument('--start-event', '-s', type=int, default=0, help='Starting event index (inclusive)')
    parser.add_argument('--end-event', '-e', type=int, default=1, help='Ending event index (exclusive)')
    parser.add_argument('--cut-value', '-c', type=float, default=0.997, help='Cut value for score_hhvsboth')
    parser.add_argument('--job-id', '-j', type=str, default='', help='Job ID for output filename')
    parser.add_argument('--count-only', action='store_true', help='Only count events, do not process')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_event < 0:
        print("Error: start-event must be non-negative")
        sys.exit(1)
    
    if args.end_event <= args.start_event:
        print("Error: end-event must be greater than start-event")
        sys.exit(1)
    
    # If only counting events, do that and exit
    if args.count_only:
        total_events = count_events_in_file(args.input, args.cut_value)
        print(f"Total events available for processing: {total_events}")
        return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract filename as prefix
    file_name = os.path.basename(args.input).replace('.root', '')
    
    # Process the specified event range
    print(f"Processing file: {args.input}")
    print(f"Event range: {args.start_event} to {args.end_event-1}")
    print(f"Cut value: {args.cut_value}")
    print("Using DCB + New Flexible Background fitting strategy")
    
    start_time = time.time()
    results = process_events_range(args.input, args.start_event, args.end_event, args.cut_value)
    end_time = time.time()
    
    if results is not None:
        # Get actual processed event indices
        actual_indices = results['event_indices']
        
        # Determine the range of indices that were actually processed
        if len(actual_indices) > 0:
            min_idx = actual_indices.min()
            max_idx = actual_indices.max()
            idx_range = f"{min_idx}-{max_idx}"
        else:
            idx_range = "empty"
        
        # Create output filename
        cut_safename = 'cut'+str(args.cut_value).replace('.', 'p')
        
        # Include job ID in filename if provided
        job_suffix = f"_job{args.job_id}" if args.job_id else ""
        
        # Save as numpy file with new background suffix
        output_file_np = os.path.join(args.output_dir, 
                                     f"{cut_safename}_{file_name}_idx{idx_range}{job_suffix}_fit_results.npz")
        save_results(results, output_file_np)
        
        # # print summary statistics
        n_processed = len(actual_indices)
        n_successful = np.sum(results['fit_success'])
        success_rate = n_successful / n_processed if n_processed > 0 else 0
        
        # Calculate background statistics (new format)
        successful_fits = results['fit_success']
        bg_amplitudes = results['bg_amp'][successful_fits]
        bg_x_slopes = results['bg_x_slope'][successful_fits]
        bg_y_slopes = results['bg_y_slope'][successful_fits]
        bg_x_logs = results['bg_x_log'][successful_fits]
        bg_y_logs = results['bg_y_log'][successful_fits]
        
        avg_bg_amp = np.mean(bg_amplitudes) if len(bg_amplitudes) > 0 else 0
        avg_x_slope = np.mean(bg_x_slopes) if len(bg_x_slopes) > 0 else 0
        avg_y_slope = np.mean(bg_y_slopes) if len(bg_y_slopes) > 0 else 0
        avg_x_log = np.mean(bg_x_logs) if len(bg_x_logs) > 0 else 0
        avg_y_log = np.mean(bg_y_logs) if len(bg_y_logs) > 0 else 0
        
        print(f"\n===== Processing Summary =====")
        print(f"Input file: {args.input}")
        print(f"Requested range: {args.start_event} to {args.end_event-1}")
        print(f"Events processed: {n_processed}")
        print(f"Successful fits: {n_successful} ({success_rate*100:.1f}%)")
        print(f"New Background Function Statistics:")
        print(f"  - Average amplitude: {avg_bg_amp:.4f}")
        print(f"  - Average X slope: {avg_x_slope:.6f}")
        print(f"  - Average Y slope: {avg_y_slope:.6f}")
        print(f"  - Average X log term: {avg_x_log:.4f}")
        print(f"  - Average Y log term: {avg_y_log:.4f}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Output files:")
        print(f"  - {output_file_np}")
        print("==============================")
        
        # Additional analysis for new background function
        if n_successful > 0:
            # print(f"\n===== New Background Function Analysis =====")
            
            # Analyze slope directions
            positive_x_slope = np.sum(bg_x_slopes > 0.001)
            negative_x_slope = np.sum(bg_x_slopes < -0.001)
            flat_x = n_successful - positive_x_slope - negative_x_slope
            
            positive_y_slope = np.sum(bg_y_slopes > 0.001)
            negative_y_slope = np.sum(bg_y_slopes < -0.001)
            flat_y = n_successful - positive_y_slope - negative_y_slope
            
            # print(f"X-direction slopes:")
            # print(f"  - Positive (increasing): {positive_x_slope} ({positive_x_slope/n_successful*100:.1f}%)")
            # print(f"  - Negative (decreasing): {negative_x_slope} ({negative_x_slope/n_successful*100:.1f}%)")
            # print(f"  - Flat: {flat_x} ({flat_x/n_successful*100:.1f}%)")
            
            # print(f"Y-direction slopes:")
            # print(f"  - Positive (increasing): {positive_y_slope} ({positive_y_slope/n_successful*100:.1f}%)")
            # print(f"  - Negative (decreasing): {negative_y_slope} ({negative_y_slope/n_successful*100:.1f}%)")
            # print(f"  - Flat: {flat_y} ({flat_y/n_successful*100:.1f}%)")
            
            # Analyze slope patterns
            diagonal_pos = np.sum((bg_x_slopes > 0.001) & (bg_y_slopes > 0.001))
            diagonal_neg = np.sum((bg_x_slopes < -0.001) & (bg_y_slopes < -0.001))
            anti_diagonal = np.sum(((bg_x_slopes > 0.001) & (bg_y_slopes < -0.001)) | 
                                 ((bg_x_slopes < -0.001) & (bg_y_slopes > 0.001)))
            
            # print(f"Slope patterns:")
            # print(f"  - Diagonal (both positive): {diagonal_pos} ({diagonal_pos/n_successful*100:.1f}%)")
            # print(f"  - Diagonal (both negative): {diagonal_neg} ({diagonal_neg/n_successful*100:.1f}%)")
            # print(f"  - Anti-diagonal: {anti_diagonal} ({anti_diagonal/n_successful*100:.1f}%)")
            
            # Analyze log term usage
            significant_x_log = np.sum(np.abs(bg_x_logs) > 0.01)
            significant_y_log = np.sum(np.abs(bg_y_logs) > 0.01)
            
            # print(f"Logarithmic terms:")
            # print(f"  - Significant X log term: {significant_x_log} ({significant_x_log/n_successful*100:.1f}%)")
            # print(f"  - Significant Y log term: {significant_y_log} ({significant_y_log/n_successful*100:.1f}%)")
            
            # print("=============================================")
        
    else:
        print("No events were processed")
        sys.exit(1)

if __name__ == "__main__":
    main()
