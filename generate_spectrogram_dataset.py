import os
import random
import argparse
import h5py
import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

DEFAULT_CLASSES = [
    "T0000",
    "T0001",
    "T0010",
    "T0011",
    "T0100",
    "T0101",
    "T0110",
    "T0111",
    "T1000",
    "T1001",
    "T1010",
    "T1100",
    "T1101",
    "T1110",
    "T1111",
    "T10000",
    "T10001",
    "T10010",
    "T10011",
    "T10100",
    "T10101",
    "T10110",
    "T10111",
    "T11000",
]


def discover_classes(data_root):
    if not os.path.exists(data_root):
        return []
    classes = []
    for name in os.listdir(data_root):
        full = os.path.join(data_root, name)
        if os.path.isdir(full) and name.startswith("T"):
            classes.append(name)

    def class_order_key(name):
        bits = name[1:]
        if bits.isdigit() and set(bits).issubset({"0", "1"}):
            # Keep 4-bit classes before 5-bit classes, then binary ascending.
            return (len(bits), int(bits, 2), name)
        return (99, 0, name)

    return sorted(classes, key=class_order_key)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_mat_files(class_dir):
    if not os.path.exists(class_dir):
        return []
    return [
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(".mat")
    ]


def read_iq_segment(mat_path, n_samples, rng):
    """
    Read a segment of IQ data from a .mat file without loading the whole dataset.
    Returns complex signal with length n_samples.
    """
    with h5py.File(mat_path, "r", libver="latest", swmr=True) as f:
        if "RF0_I" in f and "RF0_Q" in f:
            i_ds, q_ds = f["RF0_I"], f["RF0_Q"]
        else:
            keys = list(f.keys())
            i_ds, q_ds = f[keys[0]], f[keys[1]]

        shape = i_ds.shape
        if len(shape) == 2 and shape[0] == 1:
            total_len = shape[1]
            max_offset = max(total_len - n_samples, 0)
            offset = rng.randint(0, max_offset + 1) if max_offset > 0 else 0
            end_pos = min(offset + n_samples, total_len)
            i_seg = i_ds[0, offset:end_pos]
            q_seg = q_ds[0, offset:end_pos]
        elif len(shape) == 2 and shape[1] == 1:
            total_len = shape[0]
            max_offset = max(total_len - n_samples, 0)
            offset = rng.randint(0, max_offset + 1) if max_offset > 0 else 0
            end_pos = min(offset + n_samples, total_len)
            i_seg = i_ds[offset:end_pos, 0]
            q_seg = q_ds[offset:end_pos, 0]
        else:
            total_len = shape[0]
            max_offset = max(total_len - n_samples, 0)
            offset = rng.randint(0, max_offset + 1) if max_offset > 0 else 0
            end_pos = min(offset + n_samples, total_len)
            i_seg = i_ds[offset:end_pos]
            q_seg = q_ds[offset:end_pos]

    i_seg = np.array(i_seg, dtype=np.float32)
    q_seg = np.array(q_seg, dtype=np.float32)

    if len(i_seg) < n_samples:
        pad_len = n_samples - len(i_seg)
        i_seg = np.pad(i_seg, (0, pad_len), "constant")
        q_seg = np.pad(q_seg, (0, pad_len), "constant")

    return (i_seg + 1j * q_seg).astype(np.complex64)


def augment_signal(
    signal,
    fs_hz,
    rng,
    noise_snr_db_min,
    noise_snr_db_max,
    freq_shift_hz,
    noise_prob,
    freq_shift_prob,
    phase_prob,
    phase_range_deg,
    gain_prob,
    gain_db_range,
    time_roll_prob,
    time_roll_max_ratio,
    mask_prob,
    mask_max_ratio,
    multipath_prob,
    multipath_delay_max,
    multipath_atten_db,
):
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    out = np.array(signal, dtype=np.complex64, copy=True)

    if phase_prob > 0 and rng.random() < phase_prob:
        phase_rad = np.deg2rad(rng.uniform(-phase_range_deg, phase_range_deg))
        out = (out * np.exp(1j * phase_rad)).astype(np.complex64)

    if gain_prob > 0 and rng.random() < gain_prob:
        gain_db = rng.uniform(-gain_db_range, gain_db_range)
        gain = 10 ** (gain_db / 20.0)
        out = (out * gain).astype(np.complex64)

    if time_roll_prob > 0 and rng.random() < time_roll_prob:
        max_shift = int(len(out) * max(0.0, time_roll_max_ratio))
        if max_shift > 0:
            shift = rng.randint(-max_shift, max_shift + 1)
            out = np.roll(out, shift).astype(np.complex64)

    if multipath_prob > 0 and rng.random() < multipath_prob:
        max_delay = max(1, int(multipath_delay_max))
        delay = rng.randint(1, max_delay + 1)
        atten = 10 ** (-abs(multipath_atten_db) / 20.0)
        delayed = np.zeros_like(out)
        delayed[delay:] = out[:-delay]
        out = (out + atten * delayed).astype(np.complex64)

    if mask_prob > 0 and rng.random() < mask_prob:
        max_mask = int(len(out) * max(0.0, mask_max_ratio))
        if max_mask > 1:
            mask_len = rng.randint(1, max_mask + 1)
            start = rng.randint(0, max(1, len(out) - mask_len + 1))
            out[start : start + mask_len] = 0

    if noise_prob > 0 and rng.random() < noise_prob:
        power = np.mean(np.abs(out) ** 2)
        if power <= 0:
            power = 1e-12
        snr_db = rng.uniform(noise_snr_db_min, noise_snr_db_max)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = power / max(snr_linear, 1e-12)
        noise = (
            np_rng.randn(len(out)).astype(np.float32)
            + 1j * np_rng.randn(len(out)).astype(np.float32)
        )
        noise = noise * np.sqrt(noise_power / 2.0)
        out = (out + noise).astype(np.complex64)

    if freq_shift_hz > 0 and rng.random() < freq_shift_prob:
        shift = rng.uniform(-freq_shift_hz, freq_shift_hz)
        t = np.arange(len(out), dtype=np.float32) / float(fs_hz)
        out = (out * np.exp(1j * 2.0 * np.pi * shift * t)).astype(np.complex64)

    return out


def compute_spectrogram_image(signal, fs_mhz, nperseg, noverlap, center_freq_mhz):
    """
    Compute spectrogram in dB and return a 2D array.
    """
    _, _, sxx = spectrogram(
        signal,
        fs=fs_mhz,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
    )

    sxx = np.fft.fftshift(sxx, axes=0)
    sxx_db = 10 * np.log10(sxx + 1e-12)

    # Shift frequency axis to center frequency (kept for consistency)
    # We do not use f/t axes when saving images, but keep the shift logic aligned.
    _ = center_freq_mhz

    return sxx_db


def save_spectrogram_image(sxx_db, out_path, image_size, image_format, jpg_quality):
    """
    Save spectrogram as an image file using a fixed colormap.
    Prefer a faster image writer when available.
    """
    if imageio is None:
        plt.imsave(out_path, sxx_db, cmap="jet", origin="lower")
        return

    # Normalize to 0-255 and apply colormap via matplotlib, then save with imageio
    vmin = np.min(sxx_db)
    vmax = np.max(sxx_db)
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = (sxx_db - vmin) / (vmax - vmin)
    rgba = plt.get_cmap("jet")(scaled, bytes=True)[:, :, :3]
    if image_size and (rgba.shape[0] != image_size or rgba.shape[1] != image_size):
        zoom_y = image_size / rgba.shape[0]
        zoom_x = image_size / rgba.shape[1]
        rgba = zoom(rgba, (zoom_y, zoom_x, 1), order=1)

    if image_format == "jpg":
        imageio.imwrite(out_path, rgba, quality=jpg_quality)
    else:
        imageio.imwrite(out_path, rgba)


def split_files_no_leak(mat_files, train_ratio, rng):
    if not mat_files:
        return [], []

    files = mat_files[:]
    rng.shuffle(files)
    if len(files) == 1:
        return files, []

    n_train = int(round(len(files) * train_ratio))
    n_train = max(1, min(n_train, len(files) - 1))
    train_files = files[:n_train]
    test_files = files[n_train:]
    return train_files, test_files


def generate_for_class(
    class_name,
    class_dir,
    out_train_dir,
    out_test_dir,
    train_count,
    test_count,
    n_samples,
    fs_hz,
    fs_mhz,
    nperseg,
    noverlap,
    center_freq_mhz,
    seed,
    train_ratio,
    augment,
    noise_snr_db_min,
    noise_snr_db_max,
    freq_shift_hz,
    noise_prob,
    freq_shift_prob,
    phase_prob,
    phase_range_deg,
    gain_prob,
    gain_db_range,
    time_roll_prob,
    time_roll_max_ratio,
    mask_prob,
    mask_max_ratio,
    multipath_prob,
    multipath_delay_max,
    multipath_atten_db,
    augment_test,
    image_size,
    image_format,
    jpg_quality,
):
    rng = random.Random(seed)
    mat_files = list_mat_files(class_dir)
    if not mat_files:
        print(f"[WARN] No .mat files found for {class_name} in {class_dir}")
        return

    train_files, test_files = split_files_no_leak(mat_files, train_ratio, rng)
    if not train_files:
        print(f"[WARN] No train files for {class_name}. Skipping.")
        return
    if not test_files:
        print(f"[WARN] No test files for {class_name}. All images will use train files only.")

    for i in range(train_count):
        mat_path = train_files[i % len(train_files)]
        try:
            signal = read_iq_segment(mat_path, n_samples, rng)
            if augment:
                signal = augment_signal(
                    signal,
                    fs_hz,
                    rng,
                    noise_snr_db_min,
                    noise_snr_db_max,
                    freq_shift_hz,
                    noise_prob,
                    freq_shift_prob,
                    phase_prob,
                    phase_range_deg,
                    gain_prob,
                    gain_db_range,
                    time_roll_prob,
                    time_roll_max_ratio,
                    mask_prob,
                    mask_max_ratio,
                    multipath_prob,
                    multipath_delay_max,
                    multipath_atten_db,
                )
            sxx_db = compute_spectrogram_image(signal, fs_mhz, nperseg, noverlap, center_freq_mhz)
            out_path = os.path.join(out_train_dir, f"{class_name}_{i:04d}.{image_format}")
            save_spectrogram_image(sxx_db, out_path, image_size, image_format, jpg_quality)
        except Exception as e:
            print(f"[ERROR] {class_name} train file {mat_path} failed: {e}")

    for i in range(test_count):
        mat_path = test_files[i % len(test_files)] if test_files else train_files[i % len(train_files)]
        try:
            signal = read_iq_segment(mat_path, n_samples, rng)
            if augment and augment_test:
                signal = augment_signal(
                    signal,
                    fs_hz,
                    rng,
                    noise_snr_db_min,
                    noise_snr_db_max,
                    freq_shift_hz,
                    noise_prob,
                    freq_shift_prob,
                    phase_prob,
                    phase_range_deg,
                    gain_prob,
                    gain_db_range,
                    time_roll_prob,
                    time_roll_max_ratio,
                    mask_prob,
                    mask_max_ratio,
                    multipath_prob,
                    multipath_delay_max,
                    multipath_atten_db,
                )
            sxx_db = compute_spectrogram_image(signal, fs_mhz, nperseg, noverlap, center_freq_mhz)
            out_path = os.path.join(out_test_dir, f"{class_name}_{i:04d}.{image_format}")
            save_spectrogram_image(sxx_db, out_path, image_size, image_format, jpg_quality)
        except Exception as e:
            print(f"[ERROR] {class_name} test file {mat_path} failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate spectrogram image dataset from IQ .mat files")
    parser.add_argument("--data-root", default="./Dataset", help="Root folder containing class subfolders")
    parser.add_argument("--out-root", default="./spectrogram_dataset", help="Output dataset root")
    parser.add_argument("--train-count", type=int, default=200, help="Images per class in train split")
    parser.add_argument("--test-count", type=int, default=50, help="Images per class in test split")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="File-level train ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--fs-hz", type=float, default=100e6, help="Sampling rate (Hz)")
    parser.add_argument("--duration", type=float, default=0.1, help="Signal duration (seconds)")
    parser.add_argument("--nperseg", type=int, default=2048, help="STFT window size")
    parser.add_argument("--noverlap", type=int, default=1024, help="STFT overlap")
    parser.add_argument("--fs-mhz", type=float, default=100, help="Spectrogram fs in MHz for display")
    parser.add_argument("--center-freq-mhz", type=float, default=2440, help="Center frequency in MHz")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--noise-snr-db-min", type=float, default=2.0, help="Min SNR for noise augmentation (dB)")
    parser.add_argument("--noise-snr-db-max", type=float, default=12.0, help="Max SNR for noise augmentation (dB)")
    parser.add_argument("--freq-shift-hz", type=float, default=12000.0, help="Max absolute frequency shift (Hz)")
    parser.add_argument("--noise-prob", type=float, default=0.9, help="Probability of adding noise")
    parser.add_argument("--freq-shift-prob", type=float, default=0.9, help="Probability of frequency shift")
    parser.add_argument("--phase-prob", type=float, default=0.8, help="Probability of random phase rotation")
    parser.add_argument("--phase-range-deg", type=float, default=45.0, help="Max phase rotation angle (deg)")
    parser.add_argument("--gain-prob", type=float, default=0.8, help="Probability of random gain scaling")
    parser.add_argument("--gain-db-range", type=float, default=8.0, help="Max gain variation (dB)")
    parser.add_argument("--time-roll-prob", type=float, default=0.7, help="Probability of circular time shift")
    parser.add_argument("--time-roll-max-ratio", type=float, default=0.12, help="Max circular shift ratio")
    parser.add_argument("--mask-prob", type=float, default=0.45, help="Probability of temporal masking")
    parser.add_argument("--mask-max-ratio", type=float, default=0.08, help="Max temporal mask ratio")
    parser.add_argument("--multipath-prob", type=float, default=0.5, help="Probability of simple multipath")
    parser.add_argument("--multipath-delay-max", type=int, default=64, help="Max delay samples for multipath")
    parser.add_argument("--multipath-atten-db", type=float, default=10.0, help="Multipath attenuation (dB)")
    parser.add_argument("--augment-test", action="store_true", help="Also augment test split")
    parser.add_argument("--image-size", type=int, default=256, help="Output image size (square). 0 keeps original")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg", help="Output image format")
    parser.add_argument("--jpg-quality", type=int, default=85, help="JPEG quality (1-95)")

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class folder names. If omitted, auto-discover all class folders in data root",
    )

    args = parser.parse_args()

    n_samples = int(args.fs_hz * args.duration)
    out_train_root = os.path.join(args.out_root, "train")
    out_test_root = os.path.join(args.out_root, "test")

    for split_root in [out_train_root, out_test_root]:
        ensure_dir(split_root)

    if args.noise_snr_db_max < args.noise_snr_db_min:
        raise ValueError("--noise-snr-db-max must be >= --noise-snr-db-min")

    classes = args.classes if args.classes else discover_classes(args.data_root)
    if not classes:
        classes = DEFAULT_CLASSES
        print("[WARN] Auto class discovery found nothing. Falling back to DEFAULT_CLASSES.")

    print(f"[INFO] Using {len(classes)} classes: {classes}")

    for class_name in classes:
        class_dir = os.path.join(args.data_root, class_name)
        out_train_dir = os.path.join(out_train_root, class_name)
        out_test_dir = os.path.join(out_test_root, class_name)
        ensure_dir(out_train_dir)
        ensure_dir(out_test_dir)

        print(f"[INFO] Generating {class_name}: train={args.train_count}, test={args.test_count}")
        generate_for_class(
            class_name=class_name,
            class_dir=class_dir,
            out_train_dir=out_train_dir,
            out_test_dir=out_test_dir,
            train_count=args.train_count,
            test_count=args.test_count,
            n_samples=n_samples,
            fs_hz=args.fs_hz,
            fs_mhz=args.fs_mhz,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            center_freq_mhz=args.center_freq_mhz,
            seed=args.seed,
            train_ratio=args.train_ratio,
            augment=not args.no_augment,
            noise_snr_db_min=args.noise_snr_db_min,
            noise_snr_db_max=args.noise_snr_db_max,
            freq_shift_hz=args.freq_shift_hz,
            noise_prob=args.noise_prob,
            freq_shift_prob=args.freq_shift_prob,
            phase_prob=args.phase_prob,
            phase_range_deg=args.phase_range_deg,
            gain_prob=args.gain_prob,
            gain_db_range=args.gain_db_range,
            time_roll_prob=args.time_roll_prob,
            time_roll_max_ratio=args.time_roll_max_ratio,
            mask_prob=args.mask_prob,
            mask_max_ratio=args.mask_max_ratio,
            multipath_prob=args.multipath_prob,
            multipath_delay_max=args.multipath_delay_max,
            multipath_atten_db=args.multipath_atten_db,
            augment_test=args.augment_test,
            image_size=args.image_size,
            image_format=args.image_format,
            jpg_quality=args.jpg_quality,
        )

    print(f"[DONE] Output saved to: {args.out_root}")


if __name__ == "__main__":
    main()
