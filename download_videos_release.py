"""
This is for the release version. The script will download the videos according to the given csv recording files
videos will be saved to the same directory(specified by input args). 
All the video names are `key` + .mp4
"""
import os
import shutil
import sys
import time
import json
import argparse
import subprocess
from functools import partial
import random
import urllib
from typing import Literal, List, Dict, Union

import requests

from copy import deepcopy

import pandas as pd
import cv2

from multiprocessing import Pool
from multiprocessing import get_context
import multiprocessing
import jsonlines
import json
import argparse
from tqdm import tqdm
from datetime import timedelta
from datetime import datetime
import pytz

import re


from utils.youtube_utils import authenticate_pytube, download_youtube_video_using_pytube
# from utils.ytb_utils.hdfs_utils import hdfs_upload_files
from pytube import innertube

ACCOUNT_NUM = 3


def remove_silent(video_path):
    try:
        os.remove(video_path)
    except FileNotFoundError:
        pass


def parse_time(timestr):
    """Parse a time string e.g. '00:00:23.279' into a timedelta object."""
    # Regular expression to match the time format
    hours, minutes, seconds = re.match(r'(\d+):(\d+):(\d+\.\d+)', timestr).groups()
    return timedelta(hours=int(hours), minutes=int(minutes), seconds=float(seconds))

def cal_span_time(span):
    """
    Calculate the duration between two times in 'HH:MM:SS.mmm' format.
    'span': ['00:00:17.759', '00:00:23.279']
    return the duration in seconds
    """
    start_time = parse_time(span[0])
    end_time = parse_time(span[1])
    duration = end_time - start_time
    # change the duration to seconds
    return duration.total_seconds()



def csv_to_jsonl(csv_file_path, out_file_name, re_generate=False, hdvg=False):
    """
    jsonl belike: {"video_id":..., "url":..., "clip": [{"clip_id":..., "span": [..., ...], "orig_caption":...}]}
    csv columns: 
    video_id url timestamp	caption key
    -2yHu5qgTzM	https://www.youtube.com/watch?v=-2yHu5qgTzM	[['0:01:20.880', '0:01:23.983'], ...]	['A neuron in a black background.', ...]
    """
    if not re_generate and os.path.exists(out_file_name):
        return
    df = pd.read_csv(csv_file_path)

    df = df[df["video_time"] < 20]
    # randomly sample 200 videos

    video_ids = df["video_id"].unique()

    # Create the output dict
    from ast import literal_eval

    if hdvg:
        df["orig_span"] = df["orig_span"].apply(literal_eval)
        df["scene_cut"] = df["scene_cut"].apply(literal_eval)
    else:
        df["span"] = df["span"].apply(literal_eval)
    out_dict = {
        video_id: {"video_id": video_id}
        for video_id in video_ids
    }

    # TODO: Add special case for HD-VG
    # Create a new JSONL file for each video
    if hdvg:
        for i in tqdm(range(len(df))):
            video_id = df.iloc[i]["video_id"]
            out_dict[video_id]["url"] = df.iloc[i]["url"]
            if "clip" not in out_dict[df.iloc[i]["video_id"]]:
                out_dict[video_id]["clip"] = []
            clip_item = {}
            # we give the clip a new id
            clip_item["clip_id"] = df.iloc[i]["key"]
            clip_item["orig_span"] = df.iloc[i]["orig_span"]
            clip_item["scene_cut"] = df.iloc[i]["scene_cut"]
            out_dict[video_id]["clip"].append(clip_item)
    else:
        for i in tqdm(range(len(df))):
            video_id = df.iloc[i]["video_id"]
            out_dict[video_id]["url"] = df.iloc[i]["url"]
            if "clip" not in out_dict[df.iloc[i]["video_id"]]:
                out_dict[video_id]["clip"] = []
            clip_item = {}
            # we give the clip a new id
            clip_item["clip_id"] = df.iloc[i]["key"]
            clip_item["span"] = df.iloc[i]["span"]
            out_dict[video_id]["clip"].append(clip_item)

    # out_dic to list
    items = list(out_dict.values())
    # Save the items to the output directory
    with jsonlines.open(out_file_name, 'w') as writer:
        for item in items:
            writer.write(item)



class DownloadDataset():
    """
    1. Support multinode downloading.
    2. More efficient downloading strategy: 
        - for video clips from the same youtube video, download once for all
        - achieved by reformatting csv files into jsonl files.
    """
    def __init__(
            self, 
            metafiles:list, 
            out_dir,
            work_dir,
            process_num,
            node_num,
            node_id,
            save_interval,
            resolution,
            record_dir="cache/download_cache/reformat_records",
            hdvg=False,
        ):
        """
        Args:
            metafile: the path to the metafile
            out_dir: the path to the output directory, the downloaded and splitted videos will be saved here
            work_dir: the path to the work directory, the downloaded videos will be saved here
            hdvg: whether to process hdvg records. HD-VG does not have the direct timestamps of the video clips.
        """
        self.metafiles = metafiles
        self.out_dir = out_dir
        self.workdir = work_dir
        self.process_num = process_num if process_num else multiprocessing.cpu_count()

        self.record_dir = record_dir

        self.save_interval = save_interval
        self.node_num = node_num
        self.node_id = node_id

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir, exist_ok=True)
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        
        self.downloaded_dir= os.path.join(self.workdir, "downloaded_videos")
        if not os.path.exists(self.downloaded_dir):
            os.makedirs(self.downloaded_dir, exist_ok=True)
        
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir, exist_ok=True)

        self.jsonl_files = []

        for metafile in self.metafiles:
            assert ".csv" in metafile, f"metafile must be a csv file, but got {metafile}"
            jsonl_file = os.path.join(self.record_dir, os.path.basename(metafile).replace(".csv", ".jsonl"))
            self.jsonl_files.append(jsonl_file)
            csv_to_jsonl(metafile, jsonl_file, re_generate=(self.node_id == 0), hdvg=hdvg)

        self.resolution = resolution
        self.hdvg = hdvg

        # get the token_file_idx which is the accound idx
        # the principle is to assign the accounts evenly to each node
        self.token_file_idx = node_id % ACCOUNT_NUM



    def read_items_jsonl(self, metafile):
        # read the josnl file as a list of dicts
        items = []
        with open(metafile, 'r') as f:
            for line in jsonlines.Reader(f):
                items.append(line)
        return items

    def downloadvideo(self, vurl):
        video_info = download_youtube_video_using_pytube(
            vurl, 
            os.path.join(self.downloaded_dir, f"{vurl.split('?v=')[-1]}.mp4"),
            res=self.resolution,
            token_file_idx=self.token_file_idx,
        )

        if video_info["downloaded"]:
            return 0
        elif video_info["error"] == "No video found":
            return -2
        elif video_info["error"] == "Age Error":
            return -1
        elif video_info["remove"] == True:
            return -2
        else:
            return 1

    
    def cal_hdvg_duration(self, record):
        if record["scene_cut"][-1] == -1:
            # we assume that hdvg only has [0, -1] scencut 
            # no [n, -1], n!=0
            return cal_span_time(record["orig_span"])
        
        return (record["scene_cut"][-1] - record["scene_cut"][0]) / record["fps"]
    

    # ============================ 
    # The entry main function
    # ============================
    def download(self, item):
        """
        Try to download the video and cut into clips, and then upload the clips onto hdfs
        """
        assert item["video_id"] == item["url"].split("?v=")[-1]

        try:
            result = self.downloadvideo(item['url'])
        except urllib.error.HTTPError as e:
            print("HTTPError: ", e)
            print("Video id: ", item["video_id"])

            remove_silent(os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"))

            return {"video_id": item["video_id"], "downloaded": -1}

        except urllib.error.URLError as e:
            print("URLError: ", e)
            print("Video id: ", item["video_id"])
            remove_silent(os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"))
            time.sleep(60)
            return {"video_id": item["video_id"], "downloaded": -1}
        except Exception as e:
            print("Exception: ", e)
            print("Video id: ", item["video_id"])
            remove_silent(os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"))
            # write into error file
            with open(os.path.join(self.workdir, f"error_{self.node_id}.txt"), 'a') as f:
                f.write(f"{item['video_id']}\n")
                # Write the UTC-0 time
                beijing_tz = pytz.timezone('Asia/Shanghai')
                # UTC time -> Beijing Time
                now_utc = datetime.now(pytz.utc)
                now_beijing = now_utc.astimezone(beijing_tz)
                f.write(f"date: {now_beijing.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{e}\n")
            time.sleep(10)
            return {"video_id": item["video_id"], "downloaded": -1}

        if result != 0:
            if result == -1:
                remove_silent(os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"))
                return {"video_id": item["video_id"], "downloaded": -1}
            else:
                remove_silent(os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"))
                return {"video_id": item["video_id"], "downloaded": -1}

        try: 
            splited_paths = self.split_video(
                                vfile=os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"), 
                                records=item["clip"]
                            )
        except AssertionError as e:
            remove_silent(os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"))
            return {"video_id": item["video_id"], "downloaded": -1}
        except Exception as e:
            remove_silent(os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"))
            return {"video_id": item["video_id"], "downloaded": -1}

        # move the splited clips to the out_dir
        for splited_path in splited_paths:
            try:
                shutil.copy(splited_path, self.out_dir)
            except FileNotFoundError as e:
                print("FileNotFoundError: ", e)
                return {"video_id": item["video_id"], "downloaded": -1}
            else:
                # remove the splited_path
                os.remove(splited_path)

        remove_silent(os.path.join(self.downloaded_dir, f"{item['video_id']}.mp4"))

        return {"video_id": item["video_id"], "downloaded": 1}

    
    def download_multinode(self, multiprocess=False):
        # load all the items
        items = []
        for jsonl_file in self.jsonl_files:
            items.extend(self.read_items_jsonl(jsonl_file))

        beijing_tz = pytz.timezone('Asia/Shanghai')
        # UTC time -> Beijing Time
        now_utc = datetime.now(pytz.utc)
        now_beijing = now_utc.astimezone(beijing_tz)
        # get the dd of dd-mm-yyyy
        date = now_beijing.strftime('%d-%m-%Y')
        dd = int(date.split("-")[0])
        seed = dd
        random.seed(seed)
        random.shuffle(items)
        # split before shuffle 
        items = items[self.node_id::self.node_num]
        to_process_items = []
        for i, item in tqdm(enumerate(items), desc="Checking items", total=len(items)):
                item["clip"] = [
                    record for record in item["clip"] if 
                    not os.path.exists(
                            os.path.join(
                                self.out_dir,
                                f"{record['clip_id']}.mp4"
                                )
                            )   
                    and record.get("has_trans_pysd", 2) != -1
                ]
                if len(item["clip"]) == 0:
                    continue
                # if item.get("downloaded", 0) == 0:
                to_process_items.append(item)

        print(f"{len(to_process_items)} items to process")

        # split the items into node_num parts
        # process the items in parallel
        batch_size = self.save_interval
        start_time = time.time()
        for b_i in tqdm(range(0, len(to_process_items), batch_size), desc=f"node: {self.node_id}", total=len(to_process_items)//batch_size):
            batch = to_process_items[b_i:b_i+batch_size]
            results = []
            processed_items = []
            if multiprocess:
                with multiprocessing.Pool(processes=self.process_num) as pool:
                    results = pool.map(self.download, batch)
            else:
                for item in batch:
                    results.append(self.download(item))
            
            # save the results
            num_failed_case = 0
            for item_i, result in enumerate(results):
                assert result["video_id"] == to_process_items[b_i + item_i]["video_id"]
                if result["downloaded"] == -1:
                    num_failed_case += 1
                processed_items.append(to_process_items[b_i + item_i])
        
            # # save the results to the temporary jsonl file
            # with jsonlines.open(self.record_path, 'a') as writer:
            #     for item in processed_items:
            #         writer.write(item)
            
            cur_time = time.time()
            # print the progress
            print(f"batch: {b_i}/{len(to_process_items)}")
            print(f"Failed to access: {num_failed_case}")
            print(f"time elapsed: {cur_time-start_time:.2f} s")

            # write into the log file
            with open(os.path.join(self.workdir, f"log_{self.node_id}.txt"), 'a') as f:
                f.write(f"batch: {b_i}/{len(to_process_items)}\n")
                f.write(f"Failed to access: {num_failed_case}\n")
                f.write(f"time elapsed: {cur_time-start_time:.2f} s\n")
                # write the current time
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")

            # calculate the eta
            eta = (cur_time-start_time) / (b_i+1) * (len(to_process_items)-(b_i+1))
            # print eta in hours
            print(f"eta: {eta/3600:.2f} h")

            
    # ============================
    # Functions for getting the video clips
    # ============================

    def split_video(
            self, 
            vfile, 
            records
        ):
        """
        split the vfile into clips, according to the records
        records: list of dicts
            fields:
                non-hdvg-src: clip_id, span, 
                hdvg-src: clip_id, orig_span, scene_cut
        """
        splited_paths = []
        if not os.path.exists(os.path.join(self.workdir, 'splited_clips')):
            os.makedirs(os.path.join(self.workdir, 'splited_clips'), exist_ok=True)
        
        for record in records:
            # use ffmpeg to split the video into clips
            base_name = record["clip_id"] + ".mp4" if ".mp4" not in record["clip_id"] else record["clip_id"]
            out_fn = os.path.join(self.workdir, 'splited_clips', base_name)

            if self.hdvg:
                start_time = record["orig_span"][0]
                end_time = record["orig_span"][1]
                start_idx = record["scene_cut"][0]
                end_idx = record["scene_cut"][1]
                out_fn = self.hdgv_split(vfile, start_time, end_time, start_idx, end_idx, record["clip_id"])
            else:
                start_time = record["span"][0]
                end_time = record["span"][1]
                out_fn = self.split_video_single(vfile, start_time, end_time, out_fn)
            splited_paths.append(out_fn)
    
        return splited_paths

    def get_duration(self, start_time, end_time):
        hh,mm,s = start_time.split(':')
        ss,ms = s.split('.')
        timems1 = 3600*1000*int((hh)) +  60*1000*int(mm) + 1000*int(ss) + int(ms)
        hh,mm,s = end_time.split(':')
        ss,ms = s.split('.')
        timems2 = 3600*1000*int((hh)) +  60*1000*int(mm) + 1000*int(ss) + int(ms)
        dur = (timems2 - timems1)/1000
        return str(dur)

    
    def split_video_single(self, vfile, start_time, end_time, out_fn):

        cmd = [
                'ffmpeg', '-ss', start_time, '-t', self.get_duration(start_time, end_time), '-accurate_seek', '-i', vfile,
                '-c', 'copy', '-avoid_negative_ts', '1', '-reset_timestamps', '1',
                '-y', '-hide_banner', '-loglevel', 'panic', '-map', '0', out_fn
            ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        return out_fn

    def hdgv_split(self, vfile, start_time, end_time, start_idx, end_idx, hdvg_clip_name):
        # cut hdvila clip
        out_fn = os.path.join(self.workdir, 'splited_clips', f"{hdvg_clip_name}_{start_idx}.mp4")

        self.split_video_single(vfile, start_time, end_time, out_fn)

        # cut hdvg clip
        clip_fn = hdvg_clip_name + ".mp4" if ".mp4" not in hdvg_clip_name else hdvg_clip_name
        if end_idx == -1:
            os.replace(out_fn, os.path.join(self.workdir, 'splited_clips', clip_fn))
            return os.path.join(self.workdir, 'splited_clips', clip_fn)
        else:
            oricap = cv2.VideoCapture(out_fn)
            save_path = os.path.join(self.workdir, 'splited_clips', clip_fn)
            h = oricap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            w = oricap.get(cv2.CAP_PROP_FRAME_WIDTH)
            fps = oricap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(w),int(h)))
            oricap.set(cv2.CAP_PROP_POS_FRAMES, start_idx+1)
            current = start_idx+1
            while current < end_idx:
                ret, frame = oricap.read()
                if ret:
                    writer.write(frame)
                current += 1
            
            oricap.release()
            writer.release()
            os.remove(out_fn)
            return save_path


def prepare_dataset_webvid(csv_file_path, out_csv_name, sample_num=50):
    df = pd.read_csv(csv_file_path)
    df = df[df['video_time'].apply(lambda x: 10<=x<=30)]
    # sample the dataset
    df = df.sample(n=sample_num*5, random_state=42, ignore_index=True)
    # change `contentUrl` to `url`
    df['url'] = df['contentUrl']
    df['valid'] = 0
    # request for each url, ensure the url is valid
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            response = requests.head(row['url'], timeout=10)
            if response.status_code == 200:
                df.iloc[i, df.columns.get_loc('valid')] = 1
        except Exception as e:
            print(e)
            df.iloc[i, df.columns.get_loc('valid')] = 0
        
    df = df[df['valid'] == 1]
    df = df.sample(n=sample_num, random_state=42)
    df.to_csv(out_csv_name, index=False)


def reset_auth(start_from_idx):
    for token_file_idx in range(start_from_idx, ACCOUNT_NUM):
        ## remove cache dirs
        print(f"removing cache file, reauthenticate for token file {token_file_idx}")
        print(f"cache file directory: {innertube._cache_dir}")
        cache_file = os.path.join(innertube._cache_dir, f'tokens_{token_file_idx}.json')
        if os.path.exists(cache_file):
            os.remove(cache_file)
        # Authenticate pytube
        authenticate_pytube(token_file_idx=token_file_idx)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='youtube video downloader')
    parser.add_argument('--workdir', default='cache/download_cache',type=str, help='Working Directory for temporary files')
    parser.add_argument("--out_dir", default="dataset/videos", type=str, help="output directory for downloaded videos")
    parser.add_argument('-r', '--reset_auth', action='store_true')

    parser.add_argument('--bsz', type=int, default=96, help='batch size')
    parser.add_argument("--process_num", type=int, default=None, help="Number of process to use")
    parser.add_argument("--resolution", type=str, default="720p", help="The lowest resolution allowed")
    parser.add_argument("--multiprocess", action='store_true')

    parser.add_argument("--node_num", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_id", type=int, default=0, help="Node ID")
    parser.add_argument("--dataset_key", type=str, default="ytb", choices=["ytb", "hdvg", "test_hdvg", "test_ytb"], help="Dataset key")
    
    parser.add_argument("--add_src_path", action='store_true', help="Add source filepath to the metadata csv files")
    parser.add_argument("--start_from", type=int, default=0, help="Start from which token file")

    parser.add_argument("--test_pytube", action='store_true')

    args = parser.parse_args()

    if args.reset_auth:
        # Follow the printed prompts to authorize with your google accounts
        # You can use ACCOUNT_NUM to specify the number of accounts to use, 
        # this will help you to largely increase the downloading speed and reduce the chance of getting banned by YouTube
        reset_auth(args.start_from)
        exit()
    
    if args.test_pytube:
        authenticate_pytube()
        exit()


    dataset_csv_map = {
        "ytb": "data/ytb_600k_720p.csv",
        "hdvg": "data/hdvg_300k_720p.csv",
        # We do not provide downloading script for webvid data, see explaination in the README.md
    }

    if args.dataset_key == "test_hdvg":
        # create a random sample of 100 videos
        if not os.path.exists("data/test_100.csv"):
            df = pd.read_csv("data/hdvg_300k_720p.csv")
            df = df.sample(n=150, random_state=42)
            df.to_csv("data/test_100.csv", index=False)
        dataset_csv_map["test_hdvg"] = "data/test_100.csv"
    
    if args.dataset_key == "test_ytb":
        # create a random sample of 100 videos
        if not os.path.exists("data/test_100_ytb.csv"):
            df = pd.read_csv("data/ytb_600k_720p.csv")
            df = df.sample(n=150, random_state=42)
            df.to_csv("data/test_100_ytb.csv", index=False)
        dataset_csv_map["test_ytb"] = "data/test_100_ytb.csv"

    if args.add_src_path:
        for dataset_name, csv_file in dataset_csv_map.items():
            df = pd.read_csv(csv_file)
            df["src_filepath"] = df["key"].apply(lambda x: os.path.join(args.out_dir, x + ".mp4"))
            df.to_csv(csv_file, index=False)
        exit()
    
    hdvg = args.dataset_key == "hdvg" or args.dataset_key == "test_hdvg"

    yvdd = DownloadDataset(
        metafiles=[dataset_csv_map[args.dataset_key]],
        work_dir=os.path.join(args.workdir, f"node_{args.node_id}"),
        save_interval=args.bsz,
        out_dir=args.out_dir,
        process_num=args.process_num,
        resolution=args.resolution,
        node_num=args.node_num,
        node_id=args.node_id,
        hdvg=hdvg,
    )

    if args.multiprocess:
        yvdd.download_multinode(multiprocess=True)
    else:
        yvdd.download_multinode(multiprocess=False)
