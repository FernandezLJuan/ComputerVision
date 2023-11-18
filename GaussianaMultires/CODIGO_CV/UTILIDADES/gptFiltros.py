import cv2
import numpy as np
import matplotlib.pyplot as plt

def highPassFilter(sigmaF, nscale, minlonguraOnda, imaxe, mult):
    rows, cols = imaxe.shape

    IM = np.fft.fft2(imaxe)

    if cols % 2:
        xvals = np.fft.fftshift(np.fft.fftfreq(cols))
    else:
        xvals = np.fft.fftfreq(cols)

    if rows % 2:
        yvals = np.fft.fftshift(np.fft.fftfreq(rows))
    else:
        yvals = np.fft.fftfreq(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radio = np.sqrt(x**2 + y**2)
    radio = np.fft.ifftshift(radio)
    radio[0, 0] = 1.0

    convolved = []

    for ss in range(nscale):
        f0 = 0.1
        NsigmaF = sigmaF * mult**ss  # Adjust as needed

        print("For scale {}, sigmaF {}".format(ss, NsigmaF))

        # Calculate the filter component for a high-pass filter
        low_pass_component = np.exp(-(radio**2) / (2 * (NsigmaF / f0)**2))
        high_pass_component = 1 - low_pass_component  # Complement for high-pass

        high_pass_component[0, 0] = 0.0

        # Apply the filter in the Fourier domain
        convolved_result = np.fft.ifft2(IM * high_pass_component)
        convolved.append(convolved_result)

        realPart = np.real(convolved_result)

        BW = 2 * np.sqrt(2 / np.log(2)) * np.abs(np.log(NsigmaF / f0))
        print("Bandwidth at scale {}: {}".format(ss, BW))

        plt.subplot(1, nscale, ss + 1)
        plt.imshow(realPart, cmap="gray")
        plt.title(f'Scale {ss + 1} - High Pass')

    plt.show()

    plt.imshow(np.sum(np.real(convolved), axis=0), cmap='gray')
    plt.title('High Pass Sum')
    plt.show()

# Load the image using OpenCV
image_path = "../DATA/orange.png"
imaxe = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Example usage
sigmaF = 0.3  # Adjust as needed
nscale = 4
minlonguraOnda = 1
mult = 1.6

highPassFilter(sigmaF, nscale, minlonguraOnda, imaxe, mult)