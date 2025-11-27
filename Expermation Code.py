"""
Walsh-Hadamard Transform Image Compression and Encryption
=========================================================

This module implements image compression and encryption using 
the Walsh-Hadamard Transform (WHT) method.

Based on the master's thesis: "Image Compression and Encryption 
using Walsh-Hadamard Method"
"""

import numpy as np


# =============================================================================
# Section 5.1: 1D and 2D Walsh-Hadamard Transform
# =============================================================================

def fwht_1d(a):
    """
    In-place Fast Walsh-Hadamard Transform for 1D array.
    Length of a must be power of 2.
    """
    a = a.astype(float).copy()
    h = 1
    n = len(a)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    # optional normalization
    a /= np.sqrt(n)
    return a


def ifwht_1d(a):
    """
    Inverse FWHT for 1D array with orthonormal scaling.
    For orthonormal version, forward and inverse are the same.
    """
    # for orthonormal transform, inverse == forward
    return fwht_1d(a)


def fwht_2d(block):
    """
    2D Walsh-Hadamard transform for square block (N x N).
    N must be power of 2.
    """
    N = block.shape[0]
    X = block.astype(float).copy()
    
    # transform rows
    for i in range(N):
        X[i, :] = fwht_1d(X[i, :])
    
    # transform columns
    for j in range(N):
        X[:, j] = fwht_1d(X[:, j])
    
    return X


def ifwht_2d(coeffs):
    """
    Inverse 2D Walsh-Hadamard transform for square block (N x N).
    For orthonormal version, inverse == forward.
    """
    N = coeffs.shape[0]
    X = coeffs.astype(float).copy()
    
    # inverse columns
    for j in range(N):
        X[:, j] = ifwht_1d(X[:, j])
    
    # inverse rows
    for i in range(N):
        X[i, :] = ifwht_1d(X[i, :])
    
    return X


# =============================================================================
# Section 5.2: Image Block Division
# =============================================================================

def image_to_blocks(img, B):
    """
    Split image (H x W) into non-overlapping blocks B x B.
    Assumes H and W are divisible by B.
    Returns 4D array of shape (num_blocks_h, num_blocks_w, B, B).
    """
    H, W = img.shape
    assert H % B == 0 and W % B == 0
    blocks_h = H // B
    blocks_w = W // B
    blocks = np.zeros((blocks_h, blocks_w, B, B), dtype=float)
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            blocks[i, j] = img[i*B:(i+1)*B, j*B:(j+1)*B]
    return blocks


def blocks_to_image(blocks):
    """
    Reconstruct image from blocks.
    blocks: 4D array (blocks_h, blocks_w, B, B)
    """
    blocks_h, blocks_w, B, _ = blocks.shape
    H = blocks_h * B
    W = blocks_w * B
    img = np.zeros((H, W), dtype=float)
    for i in range(blocks_h):
        for j in range(blocks_w):
            img[i*B:(i+1)*B, j*B:(j+1)*B] = blocks[i, j]
    return img


# =============================================================================
# Section 5.3: Compression in Walsh Domain
# =============================================================================

def compress_block_wht(block, threshold):
    """
    Apply 2D WHT to block and zero out small coefficients.
    Returns compressed coefficients.
    """
    F = fwht_2d(block)
    mask = np.abs(F) >= threshold
    F_compressed = F * mask
    return F_compressed, mask


def decompress_block_wht(F_compressed):
    """
    Reconstruct block from compressed WHT coefficients.
    """
    block_rec = ifwht_2d(F_compressed)
    return block_rec


def compress_image_wht(img, B=8, threshold=5.0):
    """
    Compress entire grayscale image using block-wise WHT.
    Returns compressed coefficient blocks and mask blocks.
    """
    blocks = image_to_blocks(img, B)
    blocks_h, blocks_w, _, _ = blocks.shape
    
    coeff_blocks = np.zeros_like(blocks)
    mask_blocks = np.zeros_like(blocks, dtype=bool)
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            F_comp, mask = compress_block_wht(blocks[i, j], threshold)
            coeff_blocks[i, j] = F_comp
            mask_blocks[i, j] = mask
    return coeff_blocks, mask_blocks


def decompress_image_wht(coeff_blocks, B=8):
    """
    Decompress grayscale image from WHT coefficient blocks.
    """
    blocks_h, blocks_w, _, _ = coeff_blocks.shape
    rec_blocks = np.zeros_like(coeff_blocks)
    for i in range(blocks_h):
        for j in range(blocks_w):
            rec_blocks[i, j] = decompress_block_wht(coeff_blocks[i, j])
    img_rec = blocks_to_image(rec_blocks)
    return img_rec


# =============================================================================
# Section 5.4: Coefficient Encryption
# =============================================================================

def encrypt_block_coeffs(F_block, key):
    """
    Encrypt WHT coefficients in block using permutation and sign mask.
    key: int seed for RNG (for demonstration purposes).
    Returns encrypted coefficients and permutation/sign info (for debug).
    """
    B = F_block.shape[0]
    N = B * B
    flat = F_block.reshape(N)
    
    rng = np.random.RandomState(key)
    perm = rng.permutation(N)
    sign_mask = rng.choice([-1.0, 1.0], size=N)
    
    encrypted_flat = sign_mask * flat[perm]
    encrypted_block = encrypted_flat.reshape(B, B)
    return encrypted_block, perm, sign_mask


def decrypt_block_coeffs(F_enc_block, key):
    """
    Decrypt WHT coefficients in block using same key.
    """
    B = F_enc_block.shape[0]
    N = B * B
    flat_enc = F_enc_block.reshape(N)
    
    rng = np.random.RandomState(key)
    perm = rng.permutation(N)
    sign_mask = rng.choice([-1.0, 1.0], size=N)
    
    flat_dec = np.zeros(N, dtype=float)
    # reverse permutation and sign mask
    for idx_enc, p in enumerate(perm):
        # flat[p] was transformed to enc[idx_enc] * sign_mask[idx_enc]
        flat_dec[p] = flat_enc[idx_enc] * sign_mask[idx_enc]
    
    dec_block = flat_dec.reshape(B, B)
    return dec_block


def encrypt_image_coeffs(coeff_blocks, base_key=1234):
    """
    Encrypt all WHT coefficient blocks with block-wise keys.
    base_key: base integer; each block uses base_key + offset.
    """
    blocks_h, blocks_w, B, _ = coeff_blocks.shape
    enc_blocks = np.zeros_like(coeff_blocks)
    for i in range(blocks_h):
        for j in range(blocks_w):
            key = base_key + i * 1000 + j
            enc_blocks[i, j], _, _ = encrypt_block_coeffs(
                coeff_blocks[i, j], key)
    return enc_blocks


def decrypt_image_coeffs(enc_blocks, base_key=1234):
    """
    Decrypt all WHT coefficient blocks with block-wise keys.
    """
    blocks_h, blocks_w, B, _ = enc_blocks.shape
    dec_blocks = np.zeros_like(enc_blocks)
    for i in range(blocks_h):
        for j in range(blocks_w):
            key = base_key + i * 1000 + j
            dec_blocks[i, j] = decrypt_block_coeffs(enc_blocks[i, j], key)
    return dec_blocks


# =============================================================================
# Section 5.5: Complete Pipeline: Compression + Encryption
# =============================================================================

def compress_encrypt_image(img, B=8, threshold=5.0, base_key=1234):
    """
    Full pipeline: block-wise WHT compression + encryption.
    Returns encrypted coefficient blocks and additional info if needed.
    """
    coeff_blocks, mask_blocks = compress_image_wht(img, B=B, 
                                                    threshold=threshold)
    enc_blocks = encrypt_image_coeffs(coeff_blocks, base_key=base_key)
    return enc_blocks, mask_blocks


def decrypt_decompress_image(enc_blocks, mask_blocks, B=8, base_key=1234):
    """
    Full pipeline: decryption + WHT-based decompression.
    """
    dec_coeff_blocks = decrypt_image_coeffs(enc_blocks, base_key=base_key)
    # Here we could re-apply mask, but mask is already encoded in zeros.
    img_rec = decompress_image_wht(dec_coeff_blocks, B=B)
    return img_rec


# =============================================================================
# Section 5.6: Quality Assessment: MSE and PSNR
# =============================================================================

def mse(original, reconstructed):
    """
    Mean Squared Error between two images.
    """
    diff = original.astype(float) - reconstructed.astype(float)
    return np.mean(diff ** 2)


def psnr(original, reconstructed, max_val=255.0):
    """
    Peak Signal-to-Noise Ratio in dB.
    """
    m = mse(original, reconstructed)
    if m == 0:
        return np.inf
    return 10.0 * np.log10((max_val ** 2) / m)


# =============================================================================
# Utility Functions
# =============================================================================

def pad_image_to_block_size(img, B):
    """
    Pad image so that dimensions are divisible by block size B.
    """
    H, W = img.shape
    new_H = ((H + B - 1) // B) * B
    new_W = ((W + B - 1) // B) * B
    
    padded = np.zeros((new_H, new_W), dtype=img.dtype)
    padded[:H, :W] = img
    return padded, (H, W)


def unpad_image(img, original_shape):
    """
    Remove padding from image.
    """
    H, W = original_shape
    return img[:H, :W]


def get_compression_ratio(mask_blocks):
    """
    Calculate compression ratio based on non-zero coefficients.
    """
    total = mask_blocks.size
    non_zero = np.sum(mask_blocks)
    if non_zero == 0:
        return np.inf
    return total / non_zero


if __name__ == "__main__":
    # Simple test
    print("Walsh-Hadamard Transform Image Compression and Encryption")
    print("=" * 60)
    
    # Create a simple test image
    np.random.seed(42)
    test_img = np.random.randint(0, 256, size=(64, 64)).astype(float)
    
    print(f"Original image shape: {test_img.shape}")
    
    # Test compression and encryption
    B = 8
    threshold = 10.0
    base_key = 12345
    
    # Compress and encrypt
    enc_blocks, mask_blocks = compress_encrypt_image(
        test_img, B=B, threshold=threshold, base_key=base_key)
    
    print(f"Encrypted blocks shape: {enc_blocks.shape}")
    print(f"Compression ratio: {get_compression_ratio(mask_blocks):.2f}")
    
    # Decrypt and decompress
    reconstructed = decrypt_decompress_image(
        enc_blocks, mask_blocks, B=B, base_key=base_key)
    
    # Calculate quality metrics
    mse_val = mse(test_img, reconstructed)
    psnr_val = psnr(test_img, reconstructed)
    
    print(f"MSE: {mse_val:.4f}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print("\nTest completed successfully!")