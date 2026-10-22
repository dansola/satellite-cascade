"""GDAL settings for reading the public Sentinel archives over HTTP.

All imagery hrefs are plain https:// URLs, which GDAL never signs. Unsigned access is still set per read with
`rasterio.Env(AWS_NO_SIGN_REQUEST="YES")` so the process-wide AWS credentials stay intact for Flyte's own uploads.
"""
import os


def configure_gdal() -> None:
    os.environ.update(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif,.jp2",
        # Under load the archive occasionally cuts a range read short; let GDAL retry at the HTTP layer.
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="1",
        GDAL_HTTP_RETRY_CODES="ALL",
    )
