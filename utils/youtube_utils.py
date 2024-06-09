from copy import deepcopy
import http.client
import os.path as osp

from pytube import YouTube
import pytube
import pytube.exceptions
import http


def get_full_url(id):
    return 'http://youtube.com/watch?v=' + id


def authenticate_pytube(token_file_idx=0):
    """Authenticate pytube using a dummy youtube url to improve stability."""
    yt = YouTube("https://www.youtube.com/watch?v=RgKAFK5djSk", use_oauth=True, allow_oauth_cache=True, token_file_idx=token_file_idx)
    _ = yt.streams.filter(file_extension='mp4')


def _get_resolution(resolution):
    """From https://github.com/pytube/pytube/blob/a32fff39058a6f7e5e59ecd06a7467b71197ce35/pytube/itags.py#L4
    we know that the resolution is in the format of "{resolution}p".
    """
    if resolution[-1] != "p" or not resolution[:-1].isdigit():
        print(f"Unknown resolution format: {resolution}")
        return None
    return int(resolution[:-1])


def download_youtube_video_using_pytube(
        url, 
        save_path, 
        token_file_idx=0,
        max_retries=1, 
        timeout=None, 
        min_res=None, 
        res=None
    ):
    """Download a video from YouTube to local storage with highest possible quality. Note that we don't filter by
    progressive=True as in the example because we want the highest quality, which is often not progressive. Thus,
    the downloaded video is likely to have no sound at all.
        > As mentioned before, progressive streams have the video and audio in a single file, but typically do not
        > provide the highest quality media; meanwhile, adaptive streams split the video and audio tracks but can
        > provide much higher quality.
        > https://pytube.io/en/latest/user/streams.html#filtering-by-streaming-method

    Reference:
        https://github.com/pytube/pytube#using-pytube-in-a-python-script

    Parameters
    ----------
    min_res : int
        Minimum resolution of the video. For example, 720 means that the video must have at least 720p resolution.
        Otherwise, the video downloading will be skipped.
    """
    # Get the highest quality video
    save_dir, save_name = osp.split(save_path)
    yt = YouTube(url, use_oauth=True, allow_oauth_cache=True, token_file_idx=token_file_idx)
    try:
        if res is None:
            highest_quality_video = yt.streams\
                .filter(file_extension='mp4')\
                .order_by('resolution')\
                .desc()\
                .first()
        else:
            highest_quality_video = yt.streams\
                .filter(file_extension='mp4', res=res)\
                .first()
            if highest_quality_video is None:
                highest_quality_video = yt.streams\
                    .filter(file_extension='mp4')\
                    .order_by('resolution')\
                    .desc()\
                    .first()
                min_res = int(res.replace("p", ""))

    except pytube.exceptions.AgeRestrictedError as e:
        # this error is often because rate/number of downloads is too high
        print("AgeRestrictedError: ", e)
        return {"downloaded": False, "error": "Age Error", "remove": True}
    except pytube.exceptions.VideoUnavailable as e:
        print("Video unavailable: ", e)
        return {"downloaded": False, "error": e, "remove": True}
    except KeyError as e:
        print("KeyError: ", e)
        if "streamingData" in e.args[0]:
            return {"downloaded": False, "error": e, "remove": True}
        else:
            raise e
    except pytube.exceptions.RegexMatchError as e:
        print("RegexMatchError: ", e)
        return {"downloaded": False, "error": e, "remove": True}
    
    if highest_quality_video is None:
        print("No video found")
        return {"downloaded": False, "error": "No video found", "remove": True}

    # Get info
    try:
        video_info = deepcopy(vars(highest_quality_video))
        video_info["_monostate"] = deepcopy(vars(video_info["_monostate"]))
    except TypeError as e:
        print("TypeError: ", e)
        video_info = {"downloaded": False, "error": e, "remove": True}
        return video_info

    # Check if the video is too low resolution
    if min_res is not None:
        video_resolution = _get_resolution(video_info['resolution'])
        if video_resolution is not None and video_resolution < min_res:
            video_info["downloaded"] = False
            video_info["remove"] = True
            video_info["error"] = f"Video resolution is too low: {video_resolution} < {min_res}"
        else:
            video_info["downloaded"] = True
    else:
        video_info["downloaded"] = True

    if video_info["downloaded"]:
        try:
            highest_quality_video.download(
                output_path=save_dir, filename=save_name, max_retries=max_retries, timeout=timeout)
        except pytube.exceptions.VideoUnavailable as e:
            return {"downloaded": False, "error": e, "remove": True}
        except AttributeError as e:
            return {"downloaded": False, "error": e, "remove": True}
        except http.client.HTTPException as e:
            print("HTTPException: ", e)
            return {"downloaded": False, "error": e, "remove": True}

    return video_info
