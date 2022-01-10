#!/usr/bin/env python3
import os
import signal
import sys
from argparse import ArgumentParser, Namespace
from typing import List

from .core import MediaFixer

__all__ = [
    "main",
    "parse_args",
    "setup_signals",
]


def setup_signals(media_fixer: MediaFixer) -> None:
    def handler(_, __):
        media_fixer.terminate()

    for s in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(s, handler)


def _validate_args(args) -> None:
    for path in (*args.sources, args.target):
        if not os.path.exists(path):
            print(f"'{path}' does not exist!", file=sys.stderr)
            sys.exit(2)


def parse_args(args: List[str]) -> Namespace:
    description = "A wrapper around ffmpeg to make it work in a concurrent and memory-buffered fashion."
    epilog = "For more information: https://github.com/simsekhalit/media-fixer"
    parser = ArgumentParser("media-fixer", description=description, epilog=epilog)
    parser.add_argument("--audio",
                        help="select index of audio stream to be used (zero-indexed). defaults to 0",
                        default=0, type=int, dest="audio_stream")
    parser.add_argument("--input-options", help="specify custom input file options for ffmpeg (overrides default ones)")
    parser.add_argument("--max-memory", type=int, help="specify allowed max memory usage as percent")
    parser.add_argument("--output-options",
                        help="specify custom output file options for ffmpeg (overrides default ones)")
    parser.add_argument("sources", nargs="+", help="source files/directories (works recursively)", metavar="SOURCE")
    parser.add_argument("target", help="target directory", metavar="TARGET")
    args = parser.parse_args(args)
    _validate_args(args)

    return args


def main(args: List[str]) -> None:
    args = parse_args(args)
    media_fixer = MediaFixer(**vars(args))
    setup_signals(media_fixer)
    media_fixer.run()


if __name__ == '__main__':
    main(sys.argv[1:])
