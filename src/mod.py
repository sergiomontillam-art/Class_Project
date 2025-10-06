import pandas as pd
from PIL import Image
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from IPython.display import display

N_IMAGES_PER_FOLDER = 1000
LOW_Q = 0.05
HIGH_Q = 0.95
IMG_BASE_OUT_DIR = Path("galaxy_color_cutouts")
N_THREADS = 10


def filtering(data_path: str) -> pd.DataFrame:
    """
    Filters and cleans a dataset of astronomical objects based on photometric error thresholds,
    probability criteria, and outlier removal in magnitude columns.

    Parameters:
        data_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the main columns of interest,
        with outliers removed from the magnitude columns and objects selected according to
        specified error and probability thresholds.

    Filtering steps:
        - Selects rows where model magnitude errors in 'u', 'g', 'r', 'i', 'z' bands are below
          specified thresholds.
        - Keeps rows where either 'p_cs_debiased' or 'p_el_debiased' is at least 0.9.
        - Removes outliers in 'modelMag_{band}' columns based on quantile thresholds (LOW_Q, HIGH_Q).
        - Returns only a subset of columns relevant for further analysis.

    """
    df = pd.read_csv(data_path)
    filter1 = (
        (df.modelMagErr_u < 0.5)
        & (df.modelMagErr_g < 0.05)
        & (df.modelMagErr_r < 0.05)
        & (df.modelMagErr_i < 0.05)
        & (df.modelMagErr_z < 0.05)
        & ((df.p_cs_debiased >= 0.9) | (df.p_el_debiased >= 0.9))
    )

    df_filtered = df[filter1]

    outliers_cols = [f"modelMag_{f}" for f in "ugriz"]
    for col in outliers_cols:
        q_low = df_filtered[col].quantile(LOW_Q)
        q_hi = df_filtered[col].quantile(HIGH_Q)
        df_filtered = df_filtered[
            (df_filtered[col] < q_hi) & (df_filtered[col] > q_low)
        ]

    main_cols = (
        [
            "dr7objid",
            "ra",
            "dec",
            "p_el_debiased",
            "p_cs_debiased",
            "spiral",
            "elliptical",
        ]
        + ["petroR50_r", "petroR90_r"]
        + [f"modelMag_{f}" for f in "ugriz"]
        + [f"extinction_{f}" for f in "ugriz"]
    )
    df_return = df_filtered[main_cols]
    return df_return


def sampling(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """
    Randomly samples `n` rows from the given DataFrame and resets the index.

    Parameters:
        df (pd.DataFrame): The input DataFrame to sample from.
        n (int): The number of rows to sample.
        random_state (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        pd.DataFrame: A new DataFrame containing `n` randomly sampled rows with a reset index.
    """
    sample = df.sample(n, random_state=random_state)
    sample = sample.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def download_cutout(
    index: int,
    ra: float,
    dec: float,
    width: int = 64,
    height: int = 64,
    format: str = "jpg",
    hips: str = "CDS/P/SDSS9/color",
    scale: float = 0.396,
):
    """
    Downloads a sky image cutout from the HiPS2FITS service and saves it to a structured directory.

    Args:
        index (int): Index of the image, used for naming and batching.
        ra (float): Right ascension of the image center (in degrees).
        dec (float): Declination of the image center (in degrees).
        width (int, optional): Width of the image in pixels. Defaults to 64.
        height (int, optional): Height of the image in pixels. Defaults to 64.
        format (str, optional): Image format (e.g., "jpg", "png"). Defaults to "jpg".
        hips (str, optional): HiPS survey identifier. Defaults to "CDS/P/SDSS9/color".
        scale (float, optional): Pixel scale in arcseconds/pixel. Defaults to 0.396.

    Returns:
        str: Status message indicating whether the image was downloaded, skipped, or if an error occurred.

    """
    IMG_BASE_OUT_DIR.mkdir(exist_ok=True)
    url = "https://alasky.u-strasbg.fr/hips-image-services/hips2fits"
    # Determine subfolder
    folder_idx = index // N_IMAGES_PER_FOLDER
    out_dir = IMG_BASE_OUT_DIR / f"batch_{folder_idx:02d}"
    out_dir.mkdir(exist_ok=True)

    filename = out_dir / f"galaxy_{index}.jpg"
    if filename.exists():  # skip if already downloaded
        return f"Skipped {index}"

    params = {
        "hips": hips,
        "ra": ra,
        "dec": dec,
        "scale": scale,
        "width": width,
        "height": height,
        "format": format,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return f"Downloaded {index}"
        else:
            return f"Failed {index}, status {response.status_code}"
    except Exception as e:
        return f"Error {index}: {e}"


def parallel_download(df: pd.DataFrame, max_workers: int = N_THREADS):
    """
    Downloads data in parallel using a thread pool executor.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'ra' and 'dec' columns for each row.
        max_workers (int, optional): Maximum number of threads to use for parallel downloads. Defaults to N_THREADS.

    Returns:
        None

    Side Effects:
        Prints the result of each download as it completes.
        Introduces a short delay (0.05 seconds) between processing each completed download.

    Notes:
        - Assumes the existence of a 'download_cutout' function that takes (i, ra, dec) as arguments.
        - Requires 'ThreadPoolExecutor', 'as_completed', and 'time' to be imported.

    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_cutout, i, row["ra"], row["dec"])
            for i, row in df.iterrows()
        ]

        for future in as_completed(futures):
            print(future.result())
            time.sleep(0.05)


def print_image(index: int):
    """
    Displays an image corresponding to the given index from a structured directory.

    Args:
        index (int): The index of the image to display.

    The function constructs the image path based on the index and a predefined number of images per folder.
    If the image exists at the constructed path, it is opened and displayed. Otherwise, a message is printed indicating that the image was not found.

    Requires:
        - IMG_BASE_OUT_DIR (Path): Base directory containing image folders.
        - N_IMAGES_PER_FOLDER (int): Number of images stored in each folder.
        - Image (from PIL): Used to open and display images.
    """
    folder_idx = index // N_IMAGES_PER_FOLDER
    img_path = IMG_BASE_OUT_DIR / f"batch_{folder_idx:02d}" / f"galaxy_{index}.jpg"
    if img_path.exists():
        img = Image.open(img_path)
        display(img)

    else:
        print(f"Image {index} not found.")
