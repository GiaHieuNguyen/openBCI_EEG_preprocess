import numpy as np
from scipy.signal import butter, lfilter, iirnotch

# ==========================================
# 1. EEG FILTERING & PREPROCESSING STAGE
# ==========================================

def apply_notch_filter(data, fs=250.0, notch_freq=50.0, Q=30.0):
    """Applies a Notch Filter to remove powerline noise (50Hz/60Hz)."""
    b, a = iirnotch(notch_freq, Q, fs)
    # Using lfilter instead of filtfilt for compatibility with real-time/short buffers
    return lfilter(b, a, data)

def apply_bandpass_filter(data, fs=250.0, lowcut=1.0, highcut=50.0, order=4):
    """Applies a Butterworth Bandpass Filter to isolate EEG bands and remove DC offset."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def preprocess_eeg(data, fs=250.0):
    """
    Standard EEG preprocessing pipeline.
    fs: Sampling frequency in Hz (Default 250Hz for OpenBCI Cyton).
    """
    # 1. Mean subtraction (Quick baseline correction)
    data = np.array(data) - np.mean(data)
    
    # 2. Apply 50Hz Notch Filter (Change to 60.0 if in US/Americas)
    data_notched = apply_notch_filter(data, fs=fs, notch_freq=50.0)
    
    # 3. Apply 1-50Hz Bandpass Filter
    data_filtered = apply_bandpass_filter(data_notched, fs=fs, lowcut=1.0, highcut=50.0)
    
    return data_filtered

# ==========================================
# 2. HEX CONVERSION STAGE (Your Code)
# ==========================================

def float_to_configurable_hex(eeg_stream_data, bit_width=32, fractional_bits=13, filename="eeg_init.hex"):
    """
    Converts floating-point EEG data into a parameterized fixed-point hex file.
    Includes saturation arithmetic to prevent catastrophic integer overflow.
    """
    scale_factor = 1 << fractional_bits
    max_val = (1 << (bit_width - 1)) - 1
    min_val = -(1 << (bit_width - 1))
    
    hex_chars = bit_width // 4
    format_str = f"0{hex_chars}X"
    bit_mask = (1 << bit_width) - 1

    with open(filename, 'w') as f:
        for val in eeg_stream_data:
            scaled_val = val * scale_factor
            int_val = int(round(scaled_val))
            
            if int_val > max_val:
                int_val = max_val
            elif int_val < min_val:
                int_val = min_val
                
            if int_val < 0:
                int_val = (1 << bit_width) + int_val
                
            int_val = int_val & bit_mask
            hex_str = f"{format(int_val, format_str)}\n"
            
            f.write(hex_str)
            
    print(f"[{bit_width}-bit, Q{bit_width-fractional_bits}.{fractional_bits}] "
          f"Successfully wrote {len(eeg_stream_data)} values to {filename}")

# ==========================================
# 3. EXECUTION AND TESTING
# ==========================================

if __name__ == "__main__":
    # Your raw data (Contains massive DC offset)
    latest_eeg_reading = [
        -25718.94535052, -21957.63849707, -21429.66794129, -12771.22798815,
        -121438.25697163, -70510.43307906, -103836.79465494, -122210.73326,
        -8716.88976489, -29472.09381725, -8961.37314574, -8856.18583634,
        -77765.22818389, -74019.07581318, -107278.02452779, -146251.66609903
    ]

    # Note: Digital filters need more than 16 samples to work properly without transient artifacts.
    # For demonstration, we will filter your 16 samples, but in reality, you want to pass
    # a larger buffer (e.g., 250+ samples representing at least 1 second of data).

    print("--- Applying Preprocessing ---")
    # Preprocess the data (Assuming 250Hz sampling rate)
    clean_eeg_reading = preprocess_eeg(latest_eeg_reading, fs=250.0)

    print(f"Sample Before: {latest_eeg_reading[4]:.2f} (Massive Offset)")
    print(f"Sample After : {clean_eeg_reading[4]:.2f} (Centered near zero)")
    print("------------------------------")

    # Example 1: Generate the 32-bit Q19.13 file
    float_to_configurable_hex(clean_eeg_reading, bit_width=32, fractional_bits=13, filename="eeg_32b_Q19_13.hex")

    # Example 2: Generate a 16-bit Q16.0 file
    float_to_configurable_hex(clean_eeg_reading, bit_width=16, fractional_bits=0, filename="eeg_16b_Q16_0.hex")

    # Example 3: Generate the 16-bit Q11.5 file
    # NOTE: Because we removed the DC offset, this will no longer aggressively hit the saturation limits!
    float_to_configurable_hex(clean_eeg_reading, bit_width=16, fractional_bits=5, filename="eeg_16b_Q11_5.hex")