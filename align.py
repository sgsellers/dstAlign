import numpy as np
import astropy.io.fits as fits
import scipy.ndimage as scind
import scipy.interpolate as scinterp
import scipy.signal as scig
import sunpy.map as smap
import sunpy.coordinates.frames as frames
import astropy.units as u
from astropy.coordinates import SkyCoord
import warnings
import cv2
from sunpy.net import Fido, attrs as a


class PointingError(Exception):
    """
    Exception raised when pointing information is not minimally complete
    """
    def __init__(self, message="Pointing information is incomplete"):
        self.message = message
        super().__init__(self.message)


def _find_nearest(array, value):
    """Determines index of closest array value to specified value
    """
    return np.abs(array-value).argmin()


# Modified from hsgPy to only consider hairlines
def detect_hairlines(image, starting_threshold=0.5, hairline_width=5, nhairlines=2, nloops=10):
    """
    Detects hairlines from a raster image (as opposed to a spectral image).
    Typically, reduced FIRS/HSG hairlines show up as a bright+dark inversion.
    We select hairlines from the derivative intensity averaged along the x-axis.
    It will attempt to find nhairlines, changing the threshold for peaks to try and match the number expected.
    If it cannot find nhairlines within nloops, it will default to the middle 60% of the image in Y.

    :param image: numpy.ndarray
        2D image (typically an averaged flat field) for beam detection.
    :param starting_threshold: float
        Threshold for derivative profile separation.
    :param hairline_width: int
        Maximum width of hairlines in pixels
    :param nhairlines: int
        Expected number of hairlines. If there are too few, repeat the code with a higher threshold.
        Too many and it lowers the threshold. Threshold moves in steps of 0.05
    :param nloops: int
        Max number of loops to perform before giving up
    :return hairline_centers: numpy.ndarray
        Locations of hairlines. Shape is 1D, with length nhairlines, and the entries are the center of the hairline
    """
    gradient_profile = np.gradient(np.nanmean(image, axis=1))
    for j in range(nloops):
        pos_peaks, _ = scig.find_peaks(gradient_profile, height=starting_threshold*np.nanmax(gradient_profile))
        neg_peaks, _ = scig.find_peaks(-gradient_profile, height=starting_threshold*np.nanmax(-gradient_profile))
        # Hairlines must have positive and negative peak within hairline_width
        # Can also be two positive peaks per negative within hairline_width.
        hairline_starts = []
        for peak in pos_peaks:
            if len(neg_peaks[(neg_peaks >= peak - hairline_width) & (neg_peaks <= peak + hairline_width)]) > 0:
                hairline_starts.append(peak)
        hairline_ends = []
        for peak in neg_peaks:
            if len(pos_peaks[(pos_peaks >= peak - hairline_width) & (pos_peaks <= peak + hairline_width)]) > 0:
                hairline_ends.append(peak)
        hairline_centers = []
        for i in range(len(hairline_starts)):
            hairline_centers.append((hairline_starts[i] + hairline_ends[i]) / 2)
        if len(hairline_centers) == nhairlines:
            break
        elif len(hairline_centers) < nhairlines:
            starting_threshold -= 0.1
        else:
            starting_threshold += 0.1
    # Couldn't find 'em. Grab the innermost 60%
    if len(hairline_centers) != nhairlines:
        startidx = int(image.shape[1]/2 - 0.3*image.shape[1])
        endidx = int(image.shape[1]/2 + 0.3*image.shape[1])
        hairline_centers = [startidx, endidx]

    hairline_centers = np.array(hairline_centers)
    return hairline_centers


def read_firs_raster(fits_file, hecore=False):
    """
    Reads a FIRS FITS file that will require alignment, and return a dictionary of relevant header information
    and the datacube
    :param fits_file: str
        Path to Level-1.5 FIRS FITS File
    :param hecore: bool
        If True, returns a He I 10830 core image
        If False, returns a continuum image
    :return image: numpy.ndarray
        2D numpy array containing requested image, cropped to avoid hairlines
    :return header_dict: dictionary
        Python dictionary of header information, with reference pixels updated for hairline cropping
    """
    with fits.open(fits_file) as f:
        if hecore:
            he_index = _find_nearest(f['lambda-coordinate'].data, 10830.3)
            image = np.nanmean(f[1].data[:, :, he_index-1:he_index+1], axis=2)
        else:
            cont_edge = _find_nearest(f['lambda-coordinate'].data, 10825)
            image = np.nanmean(f[1].data[:, :, :cont_edge], axis=2)
        date_avg = np.datetime64(f[0].header['STARTOBS']) + (
            np.datetime64(f[0].header['ENDOBS']) - np.datetime64(f[0].header['STARTOBS'])
        ) / 2
        header_dict = {
            "CDELT1": f[1].header['CDELT1'],
            "CDELT2": f[1].header['CDELT2'],
            "CTYPE1": f[1].header["CTYPE1"],
            "CTYPE2": f[1].header["CTYPE2"],
            "CUNIT1": f[1].header["CUNIT1"],
            "CUNIT2": f[1].header["CUNIT2"],
            "CRVAL1": f[1].header["CRVAL1"],
            "CRVAL2": f[1].header["CRVAL2"],
            "CRPIX1": f[1].header["CRPIX1"],
            "CRPIX2": f[1].header["CRPIX2"],
            "CROTA2": f[1].header["CROTA2"],
            "DATE-AVG": date_avg.astype(str)
        }
    return image, header_dict


def read_rosa_zyla_image(filename, slat=None, slon=None, dx=None, dy=None, rotation=None, obstime=None, translation=[]):
    """
    Reads a ROSA or HARDCam Level-1 or 1.5 FITS file and preps it for alignment. If the FITS file is from an
    older version of SSOSoft without useful metadata, these must be provided.

    Likewise, these cameras can be afflicted by an unknown series of flips and rotations induced by the optical path.
    If these are not corrected for (as in older versions of SSOSoft), these translations must be provided.
    See the SSOSoft github repository (https://github.com/sgsellers/SSOsoft) for an example of the correct target
    image orientation.

    :param filename: str
        Path to file containing image to be used for alignment.
    :param slat: float, or None-type
        Stonyhurst Latitude. Required if not in the FITS header
    :param slon: float, or None-type
        Stonyhurst Longitude. Required if not in the FITS header
    :param dx: float
        Pixel Scale in X. Required if not in the FITS header
    :param dy: float
        Pixel Scale in Y. Required if not in the FITS header
    :param rotation: float
        Rotation relative to solar-north. Note that this is NOT the same as the obs series GDRAN for the DST.
        The DST guider is rotated such that 13.3 degrees is solar north, so subtract 13.3 from the GDRAN.
        Note also that the DST GDRAN is set up such that positive is clockwise, so for scipy and other functions,
        the sign of the GDRAN must be reversed. This is done in the code. You just subtract off that 13.3, and we'll
        call it even, okay?
    :param obstime: str
        String containing obstime in format YYYY-MM-DDTHH:MM:SS.mmm. Required if not in FITS header.
    :param translation: list
        List of bulk translations. Currently accepts rot90, fliplr, flipud, flip, corresponding to numpy
        array manipulation functions of the same name. If you are unsure whether the data have already undergone
        bulk translation to match the telescope's orientation, check the PRSTEP keywords in the FITS header.
        If there are no PRSTEP keywords, or if the PRSTEP keywords do not indicate alignment to Solar-North, this
        must be provided.
    :return image: numpy.ndarray
        Image array
    :return header_dict: dictionary
        Python Dictionary containing minimal header keywords.
    """
    with fits.open(filename) as f:
        required_keywords = [
            "CDELT1", "CDELT2", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2", "CROTA2", "STARTOBS"
        ]
        # Check to see if all required keywords are in FITS header. If they aren't all there, check if
        # kwargs are provided
        if not all([i in list(f[0].header.keys()) for i in required_keywords]):
            if not all([slat, slon, dx, dy, rotation]):
                raise PointingError()
            # If they are, populate the metadata dictionary from kwargs
            else:
                coord = SkyCoord(
                    slon*u.deg, slat*u.deg,
                    obstime=obstime, observer="earth",
                    frame=frames.HeliographicStonyhurst
                ).transform_to(frames.Helioprojective)
                header_dict = {
                    "CDELT1": dx,
                    "CDELT2": dy,
                    "CTYPE1": "HPLN-TAN",
                    "CTYPE2": "HPLT-TAN",
                    "CUNIT1": "arcsec",
                    "CUNIT2": "arcsec",
                    "CRVAL1": coord.Tx.value,
                    "CRVAL2": coord.Ty.value,
                    "CRPIX1": f[0].data.shape[1],
                    "CRPIX2": f[0].data.shape[0],
                    "CROTA2": rotation,
                    "DATE-AVG": obstime
                }
        else:
            header_dict = {
                "CDELT1": f[0].header['CDELT1'],
                "CDELT2": f[0].header['CDELT2'],
                "CTYPE1": f[0].header["CTYPE1"],
                "CTYPE2": f[0].header["CTYPE2"],
                "CUNIT1": f[0].header["CUNIT1"],
                "CUNIT2": f[0].header["CUNIT2"],
                "CRVAL1": f[0].header["CRVAL1"],
                "CRVAL2": f[0].header["CRVAL2"],
                "CRPIX1": f[0].header["CRPIX1"],
                "CRPIX2": f[0].header["CRPIX2"],
                "CROTA2": f[0].header["CROTA2"],
                "DATE-AVG": f[0].header['STARTOBS']
            }
        image = f[0].data
        for i in translation:
            if "rot90" in i.lower():
                image = np.rot90(image)
            if "fliplr" in i.lower():
                image = np.fliplr(image)
            if "flipud" in i.lower():
                image = np.flipud(image)
        return image, header_dict


def align_images(data_image, data_dict, reference_smap, niter=3, rotation_correct=False):
    """
    Performs iterative bulk and fine alignment of reference and data image.
    General flow is:
        1.) Degrade data image to reference image scale using parameters in relevant dictionaries
        2.) Align reference image to data image
        3.) Update centering of reference image
        4.) Repeat niter times
        5.) Interpolate reference image to data image scale
        6.) Align again...
    :param data_image: numpy.ndarray
        2D array containing image to align. Expected to be rotated to solar north (or nearly so),
        and cropped to avoid hairlines.
    :param data_dict: dictionary
        Python dictionary of alignment-focused FITS keywords. This should have:
            a.) DATE-AVG
            b.) CDELT1/2
            c.) CRPIX1/2 NOTE: MUST Be altered for new yrange if hairlines are clipped
            d.) CTYPE1/2
            e.) CUNIT1/2
            f.) CRVAL1/2
        Provided that these are referenced to the center, cropping the hairlines shouldn't matter.
    :param reference_smap: sunpy.map.Map
        Full-disk Sunpy Map. Generally HMI
    :param niter: int
        Number of sequential alignments to perform. Equal for coarse and fine alignments
    :param rotation_correct: bool
        If true, after fine-alignment, attempts to determine relative rotation via linear correlation
    :return data_dict: dictionary
        Python dictionary of correct alignment keywords, modified from data_dict.
    """
    xgrid = np.linspace(
        0,
        data_dict['CDELT1'] * (data_image.shape[1] - 1),
        num=data_image.shape[1]
    )
    ygrid = np.linspace(
        0,
        data_dict['CDELT2'] * (data_image.shape[0] - 1),
        num=data_image.shape[0]
    )
    interpolator = scinterp.RegularGridInterpolator(
        (ygrid, xgrid), data_image
    )
    new_xgrid = np.arange(0, xgrid[-1], reference_smap.scale[0].value)
    new_ygrid = np.arange(0, ygrid[-1], reference_smap.scale[1].value)
    ut, vt = np.meshgrid(new_ygrid, new_xgrid, indexing='ij')
    interpoints = np.array([ut.ravel(), vt.ravel()]).T
    interpolated_data = interpolator(interpoints, method='linear').reshape(len(new_ygrid), len(new_xgrid))
    # Sequential Alignment. Sometimes DST coordinates are far off, so repeat alignment with increasingly-accurate
    # Reference image submaps.
    for i in range(niter):
        data_map = smap.Map(data_image, data_dict)
        # Near or beyond the limb, HMI and AIA pixel coordinates break down.
        # To be safe, we're going to do some fuckery with coordinates instead of a simple submap.
        # We'll want to pad the edges, just in case the coordinates are very far off.
        # Pad by FOV/2 (i.e., and extra FOV/4 on each side)
        with frames.Helioprojective.assume_spherical_screen(data_map.observer_coordinate):
            reference_submap = reference_smap.reproject_to(data_map.wcs)
        # reference_submap = reference_smap.submap(
        #     bottom_left=data_map.bottom_left_coord,
        #     top_right=data_map.top_right_coord
        # )
        correlation_map = scig.correlate2d(interpolated_data, reference_submap.data, mode='same', boundary='symm')
        y, x = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        y0 = interpolated_data.shape[0]/2
        x0 = interpolated_data.shape[1]/2
        y0ref = reference_submap.data.shape[0]/2
        x0ref = reference_submap.data.shape[1]/2
        yshift = y0 - y - (y0 - y0ref)
        xshift = x0 - x - (x0 - x0ref)
        data_dict['CRVAL1'] = data_map.center.Tx.value + (xshift * data_map.scale[0].value)
        data_dict['CRVAL2'] = data_map.center.Ty.value + (yshift * data_map.scale[1].value)
    # Now interpolate both the original and reference map to the finest grid scale
    fine_scale = np.nanmin([data_dict['CDELT1'], data_dict['CDELT2']])
    fine_xgrid = np.arange(0, xgrid[-1], fine_scale)
    fine_ygrid = np.arange(0, ygrid[-1], fine_scale)
    ut, vt = np.meshgrid(fine_ygrid, fine_xgrid, indexing='ij')
    interpoints = np.array([ut.ravel(), vt.ravel()]).T
    interpolated_data = interpolator(interpoints, method='linear').reshape(len(fine_ygrid), len(fine_xgrid))
    # Sequential fine alignment
    for i in range(niter):
        data_map = smap.Map(data_image, data_dict)
        reference_submap = reference_smap.submap(
            bottom_left=data_map.bottom_left_coord,
            top_right=data_map.top_right_coord
        )
        reference_xgrid = np.linspace(
            0,
            reference_submap.scale[0].value * (reference_submap.data.shape[1] - 1),
            num=reference_submap.data.shape[1]
        )
        reference_ygrid = np.linspace(
            0,
            reference_submap.scale[1].value * (reference_submap.data.shape[0] - 1),
            num=reference_submap.data.shape[0]
        )
        reference_interpolator = scinterp.RegularGridInterpolator(
            (reference_ygrid, reference_xgrid), reference_submap.data
        )
        reference_fine_xgrid = np.arange(0, reference_xgrid[-1], fine_scale)
        reference_fine_ygrid = np.arange(0, reference_ygrid[-1], fine_scale)
        rut, rvt = np.meshgrid(reference_fine_ygrid, reference_fine_xgrid, indexing='ij')
        refinterpoints = np.array([rut.ravel(), rvt.ravel()]).T
        refinterp = reference_interpolator(refinterpoints, method='linear').reshape(
            len(reference_fine_ygrid), len(reference_fine_xgrid)
        )
        correlation_map = scind.correlate(interpolated_data, refinterp)
        y, x = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        y0 = interpolated_data.shape[0] / 2
        x0 = interpolated_data.shape[1] / 2
        y0ref = refinterp.shape[0] / 2
        x0ref = refinterp.shape[1] / 2
        yshift = y0 - y - (y0 - y0ref)
        xshift = x0 - x - (x0 - x0ref)
        data_dict['CRVAL1'] = data_map.center.Tx.value + (xshift * fine_scale)
        data_dict['CRVAL2'] = data_map.center.Ty.value + (yshift * fine_scale)
    if rotation_correct:
        data_map = smap.Map(data_image, data_dict)
        reference_submap = reference_smap.submap(
            bottom_left=data_map.bottom_left_coord,
            top_right=data_map.top_right_coord
        )
        reference_xgrid = np.linspace(
            0,
            reference_submap.scale[0].value * (reference_submap.data.shape[1] - 1),
            num=reference_submap.data.shape[1]
        )
        reference_ygrid = np.linspace(
            0,
            reference_submap.scale[1].value * (reference_submap.data.shape[0] - 1),
            num=reference_submap.data.shape[0]
        )
        reference_interpolator = scinterp.RegularGridInterpolator(
            (reference_ygrid, reference_xgrid), reference_submap.data
        )
        reference_fine_xgrid = np.arange(0, reference_xgrid[-1], fine_scale)
        reference_fine_ygrid = np.arange(0, reference_ygrid[-1], fine_scale)
        rut, rvt = np.meshgrid(reference_fine_ygrid, reference_fine_xgrid, indexing='ij')
        refinterpoints = np.array([rut.ravel(), rvt.ravel()]).T
        refinterp = reference_interpolator(refinterpoints, method='linear').reshape(
            len(reference_fine_ygrid), len(reference_fine_xgrid)
        )
        rotangle = determine_relative_rotation(interpolated_data, refinterp)
        data_dict['CROTA2'] = rotangle

    return data_dict


def determine_relative_rotation(data_image, reference_image):
    """
    Iteratively determines relative rotation of two images.
    Assumes both images are scaled the same, and are co-aligned within tolerances.
    Will crop larger array to smaller to account for single-pixel discrepancies

    :param data_image: numpy.ndarray
        2D numpy array containing image data
    :param reference_image: numpy.ndarray
        2D numpy array containing reference image data
    :return rotation: float
        Relative angle between the two images
    """
    if data_image.shape != reference_image.shape:
        if len(data_image.flatten()) > len(reference_image.flatten()):
            data_image = data_image[:reference_image.shape[0], :reference_image.shape[1]]
        else:
            reference_image = reference_image[:data_image.shape[0], :data_image.shape[1]]

    # Rotate between +/- 5 degrees
    rotation_angles = np.linspace(-5, 5, num=50)
    correlation_vals = np.zeros(len(rotation_angles))
    for i in range(len(rotation_angles)):
        rotated_image = scind.rotate(data_image, rotation_angles[i], reshape=False)
        correlation_vals[i] = np.nansum(
            rotated_image * reference_image
        ) / np.sqrt(
            np.nansum(rotated_image**2) * np.nansum(reference_image**2)
        )
    interp_range = np.linspace(rotation_angles[0], rotation_angles[-1], 1000)
    corr_interp = scinterp.interp1d(
        rotation_angles,
        correlation_vals,
        kind='quadratic'
    )(interp_range)
    rotation = interp_range[list(corr_interp).index(np.nanmax(corr_interp))]
    if (rotation == interp_range[0]) or (rotation == interp_range[-1]):
        warnings.warn("Rotation offset could not be determined by linear correlation. Defaulting to 0 degrees.")
        rotation = 0
    return rotation


def scale_image(arr, vmin=0, vmax=1):
    """Rescales image to be between vmin and vmax.
    Mostly for converting float images to int images where the original vmin/vmax are close"""
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


def determine_linegrid_spacing(linegrid_image, pixel_thresh=10):
    """Uses a Hough transform to determine line grid spacing"""
    rescaled_linegrid = scale_image(linegrid_image, vmin=0, vmax=255).astype(np.uint8)
    # Thresholds are currently a fudge.
    edge_image = cv2.Canny(rescaled_linegrid, 50, 120)
    # Only want vertical and horizontal lines
    theta_res = np.pi / 2
    # Pixel-by-pixel resolution. Smaller means fewer lines
    rho_res = 1
    # Fully don't know what this does.
    threshold = 300
    hough_lines = cv2.HoughLines(edge_image, rho_res, theta_res, threshold)
    # Theta should be near zero
    horizontal_lines = np.sort(np.array([i[0] for i in hough_lines[:, 0, :] if i[1] < 1]))
    horizontal_lines -= horizontal_lines[0]
    # Theta should be near pi/2
    vertical_lines = np.sort(np.array([i[0] for i in hough_lines[:, 0, :] if i[1] > 1]))
    vertical_lines -= vertical_lines[0]

    mean_horizontal_lines = []
    for i in range(1, len(horizontal_lines)):
        if horizontal_lines[i] - horizontal_lines[i - 1] < pixel_thresh:
            mean_horizontal_lines.append(0.5*(horizontal_lines[i] + horizontal_lines[i - 1]))
        else:
            mean_horizontal_lines.append(horizontal_lines[i])
    mean_horizontal_lines = np.sort(np.array(mean_horizontal_lines))
    subtractions = mean_horizontal_lines[1:] - mean_horizontal_lines[:-1]
    horizontal_spacing = np.nanmean(subtractions)
    return horizontal_spacing


def fetch_reference_image(metadata, savepath=".", channel="HMI"):
    """
    Fetches temporally-similar full-disk reference image for alignment.
    :param metadata: dict
        Python dictionary of metadata. For our purposes, we really only need a time.
    :param savepath: str
        Path to save reference image. Defaults to working directory
    :param channel: str
        Reference image channel. Default is HMI for whitelight.
        Otherwise, assumes that the string is an AIA filter wavelength, e.g., "171"
    :return refmap: sunpy.map.Map
        Sunpy map of reference image.
    """
    timerange = (
        metadata['DATE-AVG'],
        (np.datetime64(metadata["DATE-AVG"]) + np.timedelta64(5, "m")).astype(str)
    )
    if channel.lower() == "hmi":
        fido_search = Fido.search(
            a.Time(timerange[0], timerange[1]),
            a.Instrument(channel),
            a.Physobs.intensity
        )
    else:
        fido_search = Fido.search(
            a.Time(timerange[0], timerange[1]),
            a.Instrument("AIA"),
            a.Wavelength(int(channel) * u.Angstrom)
        )
    dl_file = Fido.fetch(fido_search[0, 0], path=savepath)
    refmap = smap.Map(dl_file)
    return refmap


def find_best_image_speckle_alpha(flist):
    """
    From a list of ROSA/Zyla Level-1 or 1.5 files, finds the best reconstruction, assuming the
    "SPKLALPH" FITS header keyword is present. If it is not, you can usually find the best
    :param flist: list
        List of filepaths for iteration
    :return best_file: str
        File in list with best alpha
    """
    spkl_alphs = np.array([fits.open(i)[0].header['SPKLALPH'] for i in flist])
    best_file = flist[spkl_alphs.argmax()]
    return best_file
