#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from abc import ABCMeta
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from enum import Enum
from io import BufferedWriter
from itertools import chain
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Tuple, Union

import psutil

__all__ = [
    "BaseTask",
    "CopyTask",
    "FixTask",
    "Logger",
    "MediaFixer",
    "Monitor",
    "Stats",
    "Status",
    "WriteTask"
]

BUFFER_SIZE = 1024 ** 2
CONCURRENCY = max(int(os.cpu_count() * 0.8), 1)
POLL_TIME = 0.1


class Logger:
    _instance: Logger = None
    _lock: Lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls, *args, **kwargs)

        return cls._instance

    def __init__(self):
        self.info_queue: deque[str] = deque()
        self.output_queue: deque[str] = deque()

    def get_infos(self) -> List[str]:
        infos = []
        while True:
            try:
                infos.append(self.info_queue.popleft())
            except IndexError:
                return infos

    def get_outputs(self) -> List[str]:
        outputs = []
        while True:
            try:
                outputs.append(self.output_queue.popleft())
            except IndexError:
                return outputs

    def log_info(self, info: Union[bytes, str]) -> None:
        if not info:
            return
        if isinstance(info, bytes):
            info = info.decode()
        self.info_queue.append(info)

    def log_output(self, output: Union[bytes, str]) -> None:
        if not output:
            return
        if isinstance(output, bytes):
            output = output.decode()
        self.output_queue.append(output)


class Stats:
    def __init__(self):
        self.data: Dict[int, List[int, int]] = {}
        self.last_time: int = time.monotonic_ns()
        self.total_size: int = 0
        self.total_time: int = 0

    def add_stat(self, size: int) -> None:
        now = time.monotonic_ns()
        took = now - self.last_time
        timestamp = now // 1_000_000_000
        if timestamp not in self.data:
            self.data[timestamp] = [0, 0]

        self.data[timestamp][0] += size
        self.data[timestamp][1] += took
        self.total_size += size
        self.total_time += took
        self.last_time = now

    def get_current_speed(self) -> float:
        now = int(time.monotonic())
        size = 0
        took = 0

        for timestamp in range(now - 4, now):
            if d := self.data.get(timestamp):
                size += d[0]
                took += d[1]

        return size / (took / 1_000_000_000) if took else 0

    def get_overall_speed(self) -> float:
        return self.total_size / (self.total_time / 1_000_000_000) if self.total_time else 0

    def reset(self) -> None:
        self.data.clear()
        self.last_time = time.monotonic_ns()
        self.total_size = 0
        self.total_time = 0


class Status(Enum):
    FAILED = "failed"
    FINISHED = "finished"
    PENDING = "pending"
    RUNNING = "running"


class BaseTask(metaclass=ABCMeta):
    def __init__(self, canceled: Event):
        self.canceled: Event = canceled
        self.status: Status = Status.PENDING
        self.terminated: Event = Event()
        self.terminating: Event = Event()

    def is_terminated(self, block: bool = False, timeout: float = None) -> bool:
        if block:
            return self.terminated.wait(timeout)
        else:
            return self.terminated.is_set()

    def is_terminating(self, block: bool = False, timeout: float = None) -> bool:
        if block:
            return self.canceled.is_set() or self.terminated.is_set() or self.terminating.wait(timeout) \
                   or self.canceled.is_set() or self.terminated.is_set()
        else:
            return self.canceled.is_set() or self.terminated.is_set() or self.terminating.is_set()


class CopyTask(BaseTask):
    def __init__(self, source: str, target: str, *, canceled: Event):
        super().__init__(canceled)
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(1)
        self.logger: Logger = Logger()
        self.source: str = source
        self.source_size: int = os.stat(source).st_size
        self.stats: Stats = Stats()
        self.target: str = target

    def _copy(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.target), exist_ok=True)
            shutil.copy2(self.source, self.target, follow_symlinks=False)
        except Exception:
            self.status = Status.FAILED
            self.logger.log_info(f"Failed to copy '{self.target}':\n{traceback.format_exc()}")
        else:
            self.status = Status.FINISHED

    def run(self) -> None:
        try:
            if self.is_terminating():
                self.status = Status.FAILED
                return
            self.stats.reset()
            self.status = Status.RUNNING
            self.logger.log_info(f"[C] {self.target}")
            future = self.executor.submit(self._copy)
            prev_size = 0

            while not (future.done() or self.is_terminating()):
                try:
                    future.result(POLL_TIME)
                except TimeoutError:
                    pass
                finally:
                    try:
                        current_size = os.stat(self.target).st_size
                    except OSError:
                        current_size = 0
                    self.stats.add_stat(current_size - prev_size)
                    prev_size = current_size

        except Exception:
            self.status = Status.FAILED
            self.logger.log_info(traceback.format_exc())
        finally:
            self.terminating.set()
            self.terminated.set()

    def terminate(self) -> None:
        self.terminating.set()


class FixTask(BaseTask):
    def __init__(self, source: str, target: str, *, command: list, canceled: Event):
        super().__init__(canceled)
        self.buffer: Queue[bytes] = Queue()
        self.buffer_size: int = BUFFER_SIZE
        self.command: List[str] = command
        self.file: Optional[BufferedWriter] = None
        self.logger: Logger = Logger()
        self.output_logger: Optional[Thread] = None
        self.process: Optional[subprocess.Popen] = None
        self.source: str = source
        self.source_size: int = os.stat(source).st_size
        self.stats: Stats = Stats()
        self.target: str = target

    def _terminate(self) -> None:
        self.terminating.set()
        self.buffer = Queue()
        if self.process is None:
            self.status = Status.FAILED
            return

        self.process.terminate()
        self.output_logger.join(3)
        self.process.kill()
        self.output_logger.join()

        if self.canceled.is_set():
            self.status = Status.FAILED
        elif self.status not in (Status.FAILED, Status.FINISHED):
            self.status = self.process.returncode == 0

        if self.status == Status.FAILED:
            self.logger.log_info(f"Failed to fix '{self.source}'.")

        self.terminated.set()

    def _put_data_to_buffer(self, data: bytes):
        while not self.is_terminating():
            try:
                self.buffer.put(data, timeout=POLL_TIME)
            except Full:
                pass
            else:
                break

    def _log_output(self) -> None:
        try:
            for line in self.process.stderr:
                if not line:
                    return
                self.logger.log_output(line.strip())
        except Exception:
            self.logger.log_info(traceback.format_exc())
        finally:
            self.process.wait()

    def run(self) -> None:
        try:
            if self.is_terminating():
                return
            self.stats.reset()
            self.status = Status.RUNNING
            self.logger.log_info(f"[F] {self.target}")
            self.process = subprocess.Popen(self.command, bufsize=BUFFER_SIZE, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.output_logger = Thread(target=self._log_output, daemon=True)
            self.output_logger.start()

            while not self.is_terminating():
                data = self.process.stdout.read(self.buffer_size)
                self._put_data_to_buffer(data)
                if not data:
                    break
                self.stats.add_stat(len(data))
                del data

            while self.buffer.qsize() > 0 and not self.is_terminating(True, POLL_TIME):
                pass

        except Exception:
            self.status = Status.FAILED
            self.logger.log_info(traceback.format_exc())
        finally:
            self._terminate()

    def terminate(self) -> None:
        self.terminating.set()


class WriteTask(BaseTask, Thread):
    def __init__(self, fix_tasks: List[FixTask], canceled: Event):
        super().__init__(canceled)
        super(BaseTask, self).__init__()
        self.fix_tasks: Dict[str, FixTask] = {t.target: t for t in fix_tasks}
        self.logger: Logger = Logger()
        self.stats: Stats = Stats()

    def _process_task(self, task: FixTask) -> bool:
        if task.file is None:
            os.makedirs(os.path.dirname(task.target), exist_ok=True)
            task.file = open(task.target, "bw", buffering=BUFFER_SIZE)

        while not self.is_terminating():
            try:
                data = task.buffer.get_nowait()
            except Empty:
                return False
            if data:
                task.file.write(data)
                self.stats.add_stat(len(data))
                del data
            else:
                task.file.close()
                return True

        return True

    def run(self) -> None:
        self.stats.reset()
        self.status = Status.RUNNING
        fix_tasks = self.fix_tasks.copy()
        failed = False
        try:
            while fix_tasks and not self.is_terminating():
                for target, task in [*fix_tasks.items()]:
                    if task.is_terminating():
                        del fix_tasks[target]
                        continue

                    try:
                        if task.status == Status.RUNNING:
                            done = self._process_task(task)
                            if done:
                                del fix_tasks[target]

                    except Exception:
                        failed = True
                        self.logger.log_info(f"Failed to write '{target}':\n{traceback.format_exc()}")
                        task.terminate()
                        del fix_tasks[target]

                self.is_terminating(True, POLL_TIME)
        finally:
            self.status = Status.FAILED if failed else Status.FINISHED
            self.terminating.set()
            self.terminated.set()

    def terminate(self) -> None:
        self.terminating.set()


class Monitor(Thread):
    def __init__(self, canceled: Event, max_memory: int = None):
        super().__init__(name="Monitor")
        self.canceled: Event = canceled
        self.copy_tasks: List[CopyTask] = []
        self.fix_tasks: List[FixTask] = []
        self.logger: Logger = Logger()
        self.max_memory: int = min(max_memory, 80) if max_memory is not None else 50
        self.start_time: float = 0
        self.write_tasks: List[WriteTask] = []

    def add_task(self, task: BaseTask) -> None:
        if isinstance(task, CopyTask):
            self.copy_tasks.append(task)
        elif isinstance(task, FixTask):
            self.fix_tasks.append(task)
        elif isinstance(task, WriteTask):
            self.write_tasks.append(task)
        else:
            raise ValueError(f"Task type '{type(task)}' is not supported.")

    @staticmethod
    def get_pretty_duration(duration: float) -> str:
        result = []
        hours = int(duration / 3600)
        if hours >= 1:
            result.append(f"{hours}h")
        minutes = int((duration // 60) % 60)
        if minutes >= 1:
            result.append(f"{minutes}m")
        seconds = int(duration % 60)
        if seconds >= 1:
            result.append(f"{seconds}s")

        if result:
            return " ".join(result)
        else:
            return f"{duration:.1f}"

    def get_overall_write_speed(self, pretty: bool = False) -> Union[float, str]:
        took = time.monotonic() - self.start_time
        speed = int(self.get_written_size() // took)
        if pretty:
            return self.get_pretty_size(speed)
        else:
            return speed

    def get_overall_process_speed(self, pretty: bool = False) -> Union[float, str]:
        took = time.monotonic() - self.start_time
        speed = int(self.get_total_size() // took)
        if pretty:
            return self.get_pretty_size(speed)
        else:
            return speed

    def print_results(self) -> None:
        if os.isatty(sys.stderr.fileno()):
            print(f"\r{' ' * os.get_terminal_size(sys.stderr.fileno()).columns}\r")

        if self.canceled.is_set():
            status = "Canceled."
        elif all(t.status == Status.FINISHED for t in chain(self.copy_tasks, self.fix_tasks, self.write_tasks)):
            status = "Done."
        else:
            status = "Failed."

        print(f"{status}\n"
              f"Processed: {self.get_total_size(True)} ({self.get_overall_process_speed(True)}/s)\n"
              f"Output:    {self.get_written_size(True)} ({self.get_overall_write_speed(True)}/s)\n"
              f"Duration:  {self.get_pretty_duration(time.monotonic() - self.start_time)}", file=sys.stderr)

    def get_written_size(self, pretty: bool = False) -> Union[int, str]:
        sizes = (t.stats.total_size for t in chain(self.copy_tasks, self.write_tasks))
        size = sum(sizes) if sizes else 0
        if pretty:
            return self.get_pretty_size(size)
        else:
            return size

    def get_current_write_speed(self, pretty: bool = False) -> Union[float, str]:
        speeds = (t.stats.get_current_speed() for t in chain(self.copy_tasks, self.write_tasks)
                  if t.status == Status.RUNNING)
        speed = sum(speeds) if speeds else 0
        if pretty:
            return self.get_pretty_size(speed)
        else:
            return speed

    def get_buffered_size(self, pretty: bool = False) -> Union[int, str]:
        sizes = (t.stats.total_size for t in chain(self.copy_tasks, self.fix_tasks))
        size = sum(sizes) if sizes else 0
        if pretty:
            return self.get_pretty_size(size)
        else:
            return size

    def get_total_size(self, pretty: bool = False) -> Union[int, str]:
        sizes = (t.source_size for t in chain(self.copy_tasks, self.fix_tasks))
        size = sum(sizes) if sizes else 0
        if pretty:
            return self.get_pretty_size(size)
        else:
            return size

    @staticmethod
    def get_pretty_size(size: Union[float, int]) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1000:
                break
            else:
                size /= 1024

        if size < 100:
            return f"{size:.1f} {unit}"
        else:
            return f"{size:.0f} {unit}"

    def get_current_buffer_speed(self, pretty: bool = False) -> Union[float, str]:
        speeds = (t.stats.get_current_speed() for t in chain(self.copy_tasks, self.fix_tasks)
                  if t.status == Status.RUNNING)
        speed = sum(speeds) if speeds else 0
        if pretty:
            return self.get_pretty_size(speed)
        else:
            return speed

    def print_stats(self) -> None:
        if not os.isatty(sys.stderr.fileno()):
            return

        b_speed = int(self.get_current_buffer_speed())
        b_speed = f"{self.get_pretty_size(b_speed)}/s" if b_speed else "💤"
        t_size = self.get_total_size()
        b_status = f"{((100 * self.get_buffered_size()) // t_size)}%" if t_size else "N/A"
        w_speed = int(self.get_current_write_speed())
        w_speed = f"{self.get_pretty_size(w_speed)}/s" if w_speed else "💤"
        w_status = f"{(100 * self.get_written_size()) // t_size}%" if t_size else "N/A"

        t_width = os.get_terminal_size(sys.stderr.fileno()).columns
        output = f"Buffering: {b_speed} ({b_status}) | Writing: {w_speed} ({w_status})"[:t_width]
        output = output[:t_width - output.count("💤")]
        print(f"\r{' ' * t_width}\r"
              f"{output:<{t_width - output.count('💤')}}", end="", file=sys.stderr, flush=True)

    def print_logs(self) -> None:
        if outputs := self.logger.get_outputs():
            if os.isatty(sys.stdout.fileno()):
                print("\r" + " " * os.get_terminal_size().columns, end="\r")
            print("\n".join(outputs))

        if infos := self.logger.get_infos():
            if os.isatty(sys.stderr.fileno()):
                print("\r" + " " * os.get_terminal_size(sys.stderr.fileno()).columns, end="\r", file=sys.stderr)
            print("\n".join(infos), file=sys.stderr)

    def _check_memory(self) -> None:
        tasks = [t for t in self.fix_tasks if t.status == Status.RUNNING]
        if not tasks:
            return

        if psutil.virtual_memory().percent < self.max_memory:
            for task in tasks:
                task.buffer.maxsize = 0
        else:
            for task in tasks:
                task.buffer.maxsize = 16

        for writer in self.write_tasks:
            w_speed = writer.stats.get_overall_speed()
            for task in writer.fix_tasks.values():
                speed = min(task.stats.get_overall_speed(), w_speed)
                task.buffer_size = max(int(speed * POLL_TIME * 2), BUFFER_SIZE)

    def _is_finished(self) -> bool:
        for task in chain(self.copy_tasks, self.fix_tasks, self.write_tasks):
            if not task.is_terminated(True, POLL_TIME):
                return False
        return True

    def run(self) -> None:
        self.start_time = time.monotonic()
        finished = False
        while not finished:
            finished = self._is_finished()
            self._check_memory()
            self.print_logs()
            self.print_stats()

        self.print_results()


class MediaFixer:
    def __init__(self, sources: List[str], target: str, *, audio_stream: int = 0, input_options: str = None,
                 max_memory: int = None, output_options: str = None):
        for prog in ("ffmpeg", "mediainfo"):
            if not shutil.which(prog):
                raise ValueError(f"'{prog}' must be installed and existing in $PATH!")

        self.audio_stream: int = audio_stream
        self.input_options: List[str] = self._parse_options(input_options)
        self.output_options: List[str] = self._parse_options(output_options)
        self.sources = [os.path.abspath(s) for s in sources]
        self.target = os.path.abspath(target)

        self.canceled: Event = Event()
        self.copy_executor: ThreadPoolExecutor = ThreadPoolExecutor(1)
        self.fix_executor: ThreadPoolExecutor = ThreadPoolExecutor(CONCURRENCY)
        self.monitor: Monitor = Monitor(self.canceled, max_memory)

    @staticmethod
    def _parse_options(options: str) -> List[str]:
        return shlex.split(options) if options else []

    def create_write_task(self, fix_tasks: List[FixTask]) -> WriteTask:
        writer = WriteTask(fix_tasks, self.canceled)
        return writer

    def create_fix_task(self, source: str, target: str, info: List[dict]) -> FixTask:
        if self.output_options:
            output_options = self.output_options
        else:
            audios = sorted((a for a in info if a["@type"] == "Audio"), key=lambda a: a["StreamOrder"])
            if "dts" in audios[self.audio_stream]["Format"].lower():
                codec = "eac3"
            else:
                codec = "copy"
            output_options = ["-c:v", "copy", "-c:a", codec, "-c:s", "copy", "-f", "matroska", "-loglevel", "warning",
                              "-nostats", "-map", "0:v", "-map", f"0:a:{self.audio_stream}", "-map", "0:s"]

        command = ["ffmpeg", *self.input_options, "-i", source, *output_options, "-y", "-"]
        return FixTask(source, target, command=command, canceled=self.canceled)

    def create_copy_task(self, source: str, target: str) -> CopyTask:
        return CopyTask(source, target, canceled=self.canceled)

    def is_dts_fix_needed(self, info: list) -> bool:
        audios = sorted((a for a in info if a["@type"] == "Audio"), key=lambda a: a["StreamOrder"])
        if not audios:
            return False
        if self.audio_stream >= len(audios):
            raise ValueError(f"Media does not have an audio stream on index {self.audio_stream}")
        if len(audios) == 1 and "dts" not in audios[0]["Format"].lower():
            return False
        else:
            return True

    @staticmethod
    def is_media(info: List[dict]) -> bool:
        return any(a for a in info if a["@type"] == "Audio")

    @staticmethod
    def get_info(path: str) -> List[Dict]:
        info = subprocess.check_output(["mediainfo", "--Output=JSON", path])
        return json.loads(info)["media"]["track"]

    def _prepare_tasks(self) -> Tuple[List[CopyTask], List[FixTask], WriteTask]:
        copy_tasks = []
        fix_tasks = []

        paths = [os.path.split(src) for src in self.sources]
        reversed(paths)
        while paths:
            src_root, rel_path = paths.pop()
            src_path = os.path.join(src_root, rel_path)
            target_path = os.path.join(self.target, rel_path)
            if os.path.isdir(src_path):
                paths += sorted(((src_root, os.path.join(rel_path, p)) for p in os.listdir(src_path)), reverse=True)
            else:
                info = self.get_info(src_path)
                if self.is_media(info):
                    if self.output_options or self.is_dts_fix_needed(info):
                        fix_tasks.append(self.create_fix_task(src_path, target_path, info))
                    else:
                        copy_tasks.append(self.create_copy_task(src_path, target_path))
                else:
                    copy_tasks.append(self.create_copy_task(src_path, target_path))

        write_task = self.create_write_task(fix_tasks)

        return copy_tasks, fix_tasks, write_task

    def _prepare_and_register_tasks(self) -> None:
        copy_tasks, fix_tasks, write_task = self._prepare_tasks()
        for task in (*copy_tasks, *fix_tasks, write_task):
            self.monitor.add_task(task)

        self.monitor.start()
        for task in copy_tasks:
            self.copy_executor.submit(task.run)
        for task in fix_tasks:
            self.fix_executor.submit(task.run)
        for task in copy_tasks:
            task.is_terminating(True)
        write_task.start()

    def run(self) -> None:
        self._prepare_and_register_tasks()
        self.monitor.join()

    def terminate(self) -> None:
        self.canceled.set()
