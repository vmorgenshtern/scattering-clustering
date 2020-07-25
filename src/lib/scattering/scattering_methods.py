############################################################
# Layers and Utils of scattering transform
# Patched-Imagenet/Scattering_space/lib/models/scattering_method
############################################################

import kymatio as km


def scattering_layer(J=3, shape=(32, 32), max_order=2, L=6, debug=False):
    """
    Creating a scattering transform "network"

    Args:
    ----
    J: Integer
        logscale of the scattering (2^J)
    shape: tuple (X,Y)
        size of the input images
    max_order: Integer
        number of scattering layers
    L: Integer
        number of angles used for the Morlet Wavelet
    scattering_layer: Kymatio Scattering 2D layer
        Kymatio layer that performs a 2D scattering transform
    total_scat_coeffs: Integer
        Total number of scattering coefficients
    """

    total_scat_coeffs = get_num_scattering_coefficients(J=J, shape=shape, max_order=max_order, L=L, debug=debug)
    scattering_layer = km.Scattering2D(J=J, shape=shape, max_order=max_order, L=L)

    return scattering_layer, total_scat_coeffs


def get_num_scattering_coefficients(J=3, shape=(32, 32), max_order=2, L=3, debug=False):
    """
    Computing the total number of scattering coefficients

    Args:
    -----
    J: Integer
        logscale of the scattering (2^J)
    shape: tuple (X,Y)
        size of the input images
    max_order: Integer
        number of scattering layers
    L: Integer
        number of angles used for the Morlet Wavelet
    total_scat_coeffs: Integer
        Total number of scattering coefficients
    """

    height = shape[0]//(2**J)
    width = shape[0]//(2**J)
    channels = 1 + L*J
    if(max_order==2):
        channels+= L**2*J*(J-1)//2
    total_scat_coeffs = channels*height*width

    if(debug):
        print("Scattering Feature stats given parameters:")
        print(f"   Total_scat_coeffs: {total_scat_coeffs}")
        print(f"   Shape: {(channels, height, width)}")
        print(f"       Height: {height}")
        print(f"       Hidth: {width}")
        print(f"       Channels: {channels}")

    return total_scat_coeffs


def get_scat_features_per_layer(J=3, L=3):
    """
    Computing the number of scattering features corresponding to each order

    Args:
    -----
    J: Integer
        logscale of the scattering (2^J)
    L: Integer
        number of angles used for the Morlet Wavelet
    """

    zero_order = 1
    first_order = L*J
    second_order =  L**2*J*(J-1)//2
    total_coeffs = zero_order + first_order + second_order

    return total_coeffs


def reshape_scat_coeffs(scat_coeffs, method="channelwise"):
    """
    Reshaping the scattering coefficients for channelwise or batchwise processing

    Args:
    -----
    input_coeffs: torch Tensor
        5-dim Scattering coefficients with shape
        (Batch_size, num_patches, n_scat_filters, height scat filter, width scat filter)
    method: string
        Method used for processing the patches
    """

    # reshaping the features for batchwise or channelwise patch processing
    if(method=="channelwise"):
        scat_coeffs = scat_coeffs.view(scat_coeffs.shape[0],-1,scat_coeffs.shape[-2],scat_coeffs.shape[-1])
    elif(method=="batchwise"):
        scat_coeffs = scat_coeffs.view(-1,scat_coeffs.shape[-3],scat_coeffs.shape[-2],scat_coeffs.shape[-1])
    else:
        print(f"Patch processing method {method} is not recognized. It must be one of the following [batchwise, channelwise]")
        exit()

    return scat_coeffs

#
