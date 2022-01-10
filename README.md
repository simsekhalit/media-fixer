# Media Fixer
Have you ever had a film or TV show that your TV wasn't able to play its audio?
Well this program is for you.
Media Fixer is a program which converts given media to a playable format.
It's a kind of wrapper for `ffmpeg` that works with multiple files in a concurrent & memory-buffered fashion.

## Requirements
Python >= 3.8 is required. (CPython and PyPy are both supported)<br>
<br>
Additionally `ffmpeg` and `mediainfo` are required for this program to run. They can be installed on Ubuntu as the following way:
```shell
sudo apt install ffmpeg mediainfo
```

## Installation
Media fixer can be either installed directly via pip:
```shell
pip install media-fixer
```
Or it can be installed from the source:
```shell
git clone https://github.com/simsekhalit/media-fixer.git
python3 -m pip install ./media-fixer
```

## Manual
```
$ python3 -m media_fixer --help
usage: media-fixer [-h] [--audio AUDIO_STREAM] [--input-options INPUT_OPTIONS] [--max-memory MAX_MEMORY] [--output-options OUTPUT_OPTIONS] SOURCE [SOURCE ...] TARGET

A wrapper around ffmpeg to make it work in a concurrent and memory-buffered fashion.

positional arguments:
  SOURCE                source files/directories (works recursively)
  TARGET                target directory

optional arguments:
  -h, --help            show this help message and exit
  --audio AUDIO_STREAM  select index of audio stream to be used (zero-indexed). defaults to 0
  --input-options INPUT_OPTIONS
                        specify custom input file options for ffmpeg (overrides default ones)
  --max-memory MAX_MEMORY
                        specify allowed max memory usage as percent
  --output-options OUTPUT_OPTIONS
                        specify custom output file options for ffmpeg (overrides default ones)

For more information: https://github.com/simsekhalit/media-fixer
```
Media Fixer can be used in one of the two modes, namely Auto and Custom.

#### Auto Mode
If no output options are given, Media Fixer works in Auto mode. Initially, an audio stream is selected with `--audio`.
For each given file, existing audio codec is detected and if it's in DTS format, it's converted to EAC-3.
Otherwise, it's just copied to the target file as it is.
While doing that only the selected audio stream is processed. Rest of the audio streams are stripped out.
This is useful for saving space since generally there are many audio streams in a film (English, French, Spanish, etc.),
but only one of them is needed.

#### Custom Mode
If output options are given with `--output-options`, Media Fixer works in custom mode.
All the given files are converted according to given output options.

## Examples
`SOURCE` and `TARGET` arguments are interpreted in the same way in both modes.<br>
Single or multiple source paths can be given. For any given source path:<br> 
If it's a file, its corresponding target is written under the target path.<br>
If it's a directory, then it's traversed recursively and all the files under it are processed.
Source directory tree is generated as the same way in the target path.
Corresponding target files are written accordingly.

<br>

> :information_source: `MediaInfo` is a helpful tool for identifying audio streams and their formats.
It can be downloaded from https://mediaarea.net/en/MediaInfo

<br>

### 1. There is a film and its 3rd audio (which is the audio stream with index 2) is chosen to be used:

```shell
python3 -m media_fixer --audio 2 '/mnt/HDD/Matrix 4.mkv' '/mnt/MyUSB'
```

`Matrix 4.mkv` is processed and the resulting file is written to `/mnt/USB/Matrix 4.mkv`

<br>

### 2. There is a folder which contains a season of a TV show: 

```
/mnt/HDD/Brooklyn Nine-Nine Season 1
/mnt/HDD/Brooklyn Nine-Nine Season 1/S01E01.mkv
/mnt/HDD/Brooklyn Nine-Nine Season 1/S01E02.mkv
/mnt/HDD/Brooklyn Nine-Nine Season 1/S01E03.mkv
...
```

Following command is executed:

```shell
python3 -m media_fixer '/mnt/HDD/Brooklyn Nine-Nine Season 1' '/mnt/USB'
```

All the episodes are processed and written to `/mnt/USB/Brooklyn Nine-Nine Season 1`

<br>

### 3. Custom ffmpeg options wanted to be used:

```shell
python3 -m media_fixer --output-options '-c:v copy -c:a aac -c:s copy -f matroska -map 0:v -map 0:a:2 -map 0:s' '/mnt/HDD/Brooklyn Nine-Nine Season 1' '/mnt/USB'
```
Following ffmpeg command is run for each source file:
```shell
ffmpeg -i $source -c:v copy -c:a aac -c:s copy -f matroska -map 0:v -map 0:a:2 -map 0:s -y -
```
`$source` is replaced with the source file's path and output is captured from stdout,
buffered there and then written to the corresponding target path.
