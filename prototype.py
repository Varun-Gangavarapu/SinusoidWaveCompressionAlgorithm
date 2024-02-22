import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
import sys

def grayscale_image_to_array(image_path):
    """
    Convert a grayscale image to a 2D array of brightness values.
    
    :param image_path: Path to the grayscale image.
    :return: 2D array of brightness values.
    """
    # Open the image
    with Image.open(image_path) as img:
        # Convert image to grayscale if it's not already
        grayscale_img = img.convert('L')
        # Convert the image data to a 2D numpy array
        brightness_array = np.array(grayscale_img)
    
    return brightness_array

def create_composite_wave(waves):
    """
    Create a composite wave from a list of sine waves.
    :param waves: List of sine waves.
    :return: Composite wave.
    """
    return np.sum(waves, axis=0)

def reconstruct_original_waves(composite_wave, t):
    """
    Attempt to reconstruct the original sine waves from a composite wave.
    :param composite_wave: The composite wave.
    :param t: Time axis.
    :return: List of reconstructed sine waves.
    """
    fft_result = np.fft.fft(composite_wave)
    frequencies = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
    positive_frequencies = frequencies[:len(frequencies)//2]
    magnitudes = np.abs(fft_result)[:len(frequencies)//2]

    # Finding peaks to guess significant frequencies
    peaks, _ = find_peaks(magnitudes, height=0.4)  # height threshold is arbitrary
    significant_freqs = positive_frequencies[peaks]
    significant_magnitudes = magnitudes[peaks]
    significant_phases = np.angle(fft_result[peaks])

    # Reconstructing waves from significant frequencies
    reconstructed_waves = []
    for i in range(len(significant_freqs)):
        wave = significant_magnitudes[i] * np.sin(2 * np.pi * significant_freqs[i] * t + significant_phases[i])
        reconstructed_waves.append(wave)

    return reconstructed_waves

def create_composite_wave_from_params(params, t):
    """
    Create a composite wave from an array of tuples (Magnitude, Frequency, Phase Shift).
    :param params: Array of tuples (Magnitude, Frequency, Phase Shift as a multiple of pi).
    :param t: Time axis.
    :return: Composite wave.
    """
    waves = [magnitude * np.sin(frequency * t + phase_shift * np.pi) for magnitude, frequency, phase_shift in params]
    return np.sum(waves, axis=0)


def reconstruct_wave_params(composite_wave, t):
    fft_result = np.fft.fft(composite_wave)
    time_step = t[1] - t[0]  # Calculate time step
    frequencies = np.fft.fftfreq(len(t), d=time_step)
    positive_frequencies = frequencies[:len(frequencies)//2]  # Frequency in Hertz
    magnitudes = np.abs(fft_result)[:len(frequencies)//2] * 2 / len(t)  # Scaling magnitudes

    # Finding peaks for significant frequencies
    peaks, _ = find_peaks(magnitudes, height=0.1)  # Adjust height as needed
    significant_freqs = positive_frequencies[peaks]
    significant_magnitudes = magnitudes[peaks]
    significant_phases = np.angle(fft_result[peaks])

    # Convert frequencies and phases to fractions of pi
    converted_freqs = [freq * 2 * np.pi for freq in significant_freqs]  # Frequency as fraction of pi
    converted_phases = [phase / np.pi for phase in significant_phases]  # Phase shift as fraction of pi

    return [(int(round(mag, 3)), int(round(freq, 3)), round(phase + .5, 3) if round(phase + .5, 3) != -0.0 else 0.0 ) for mag, freq, phase in zip(significant_magnitudes, converted_freqs, converted_phases)]

def create_image_from_brightness_array(brightness_array, output_path):
    """
    Create an image from a 2D array of brightness values.

    :param brightness_array: 2D array of brightness values.
    :param output_path: Path to save the output image.
    """
    # Convert the 2D array to a numpy array with the 'uint8' data type
    image_data = np.array(brightness_array, dtype=np.uint8)

    # Create an image from the numpy array
    image = Image.fromarray(image_data, 'L')

    # Save or show the image
    image.save(output_path)
    # image.show()  # Uncomment to display the image

def example2():
    # Example usage
    # Parameters for creating waves: (Magnitude, Frequency, Phase Shift)
    t = np.linspace(0, 2 * np.pi, 5000, endpoint=False)

    image_path = 'images/Image2.jpg'  # Replace with your image path
    brightness_values = grayscale_image_to_array(image_path)
    wave_params = []
    image = []
    for i in range(0, len(brightness_values)):
        for j in range(0, len(brightness_values)):
            wave_params.append((i+1,(j+1)*10, brightness_values[i][j]/1000))
        composite_wave = create_composite_wave_from_params(wave_params, t)
        # print(sys.getsizeof(composite_wave))
        # Reconstructing wave parameters
        reconstructed_params = reconstruct_wave_params(composite_wave, t)
        # print(sys.getsizeof(reconstructed_params))
        last_elements = [int(tup[-1]*1000) for tup in reconstructed_params]
        image.append(last_elements)
        # print("Origional Parameters: ", wave_params)
        # print("Reconstructed Parameters: ", reconstructed_params)
        
    
    print(image)
    print(sys.getsizeof(brightness_values))
    print(sys.getsizeof(image))
    print(len(brightness_values))
    print(len(image))
    create_image_from_brightness_array(image, "images/output.png")
    # wave_params = [
    #     (1, 200, 0.0),
    #     (2, 300, 0.0),
    #     (3, 400, 0.2)
    # ]
    # Creating the composite wave
    # composite_wave = create_composite_wave_from_params(wave_params, t)
    # print("Done")
    # # Reconstructing wave parameters
    # reconstructed_params = reconstruct_wave_params(composite_wave, t)
    # print("Origional Parameters: ", wave_params)
    # print("Reconstructed Parameters: ", reconstructed_params)


def example1():
    # Example usage:
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)

    # Creating original waves
    original_waves = [
        1 * np.sin(2 * 2 * np.pi * t),
        1.5 * np.sin(3 * 2 * np.pi * t + np.pi/4),
        0.5 * np.sin(4 * 2 * np.pi * t + np.pi/2)
    ]

    # Creating the composite wave
    composite_wave = create_composite_wave(original_waves)

    # Reconstructing the original waves
    reconstructed_waves = reconstruct_original_waves(composite_wave, t)

    # Plotting for demonstration
    plt.figure(figsize=(12, 8))

    # Plotting the composite wave
    plt.subplot(len(reconstructed_waves) + 1, 1, 1)
    plt.plot(t, composite_wave)
    plt.title('Composite Wave')

    # Plotting the reconstructed waves
    for i, wave in enumerate(reconstructed_waves, start=2):
        plt.subplot(len(reconstructed_waves) + 1, 1, i)
        plt.plot(t, wave)
        plt.title(f'Reconstructed Wave {i-1}')

    plt.tight_layout()
    plt.show()

def imageArr():
    image_path = 'images/Image2.jpg'  # Replace with your image path
    brightness_values = grayscale_image_to_array(image_path)
    with open("output.txt", 'w') as file:
            for row in brightness_values:
                file.write(' '.join(map(str, row)) + '\n')
    print(brightness_values)

example2()

