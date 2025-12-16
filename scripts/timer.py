import json
import os
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4


def _env_to_int(keys, default: int = 0) -> int:
    for key in keys:
        raw = os.getenv(key)
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return default


class DistributedTimer:
    def __init__(
        self,
        name: str,
        root_dir: Optional[Path | str] = None,
        wait_interval: float = 0.5,
        verbose: bool = True,
    ):
        self.name = name
        self.root_dir = Path(root_dir).expanduser() if root_dir else Path.cwd()
        self.wait_interval = wait_interval
        self.verbose = verbose
        self.rank = max(0, _env_to_int(("RANK", "LOCAL_RANK")))
        self.world_size = max(1, _env_to_int(("WORLD_SIZE", "SLURM_NTASKS")))
        self.timer_dir = self.root_dir / ".lcm_timers"
        self.timer_dir.mkdir(parents=True, exist_ok=True)
        self.start_file = self.timer_dir / f"{name}.start.json"
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._stopped = False

    def start(self) -> float:
        if self._start_time is not None:
            return self._start_time

        now = time.time()
        run_id = f"{int(now)}-{uuid4().hex[:8]}"
        payload = None
        try:
            with self.start_file.open("x") as fp:
                payload = {"start_time": now, "run_id": run_id}
                json.dump(payload, fp)
        except FileExistsError:
            payload = self._safe_load(self.start_file)

        if payload is None:
            payload = {"start_time": now, "run_id": run_id}
            with self.start_file.open("w") as fp:
                json.dump(payload, fp)

        self._start_time = payload.get("start_time", now)
        self.run_id = payload.get("run_id", run_id)
        self.run_dir = self.timer_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose and self.rank == 0:
            print(
                f"⏱️ 计时器 {self.name} 启动 run_id={self.run_id} "
                f"(rank {self.rank}/{self.world_size}) "
                f"start={self._format_ts(self._start_time)}"
            )
        return self._start_time

    def stop(self, label: Optional[str] = None) -> Optional[float]:
        if self._stopped:
            return None

        if self._start_time is None:
            self.start()

        end_time = time.time()
        if self.run_dir is None:
            return None

        rank_file = self.run_dir / f"rank_{self.rank}.json"
        try:
            with rank_file.open("w") as fp:
                json.dump({"rank": self.rank, "end_time": end_time}, fp)
        except OSError:
            pass

        self._stopped = True
        if self.rank == 0:
            return self._finalize(label)
        return None

    def _finalize(self, label: Optional[str]) -> float:
        expected_files = {f"rank_{i}.json" for i in range(self.world_size)}
        if self.run_dir is None:
            return 0.0

        while True:
            available = {p.name for p in self.run_dir.glob("rank_*.json")}
            if expected_files.issubset(available):
                break
            time.sleep(self.wait_interval)

        end_times = []
        for rank in range(self.world_size):
            data = self._safe_load(self.run_dir / f"rank_{rank}.json")
            if data is None:
                continue
            end_times.append(data.get("end_time", self._start_time))

        final_end = max(end_times) if end_times else (self._start_time or time.time())
        start_meta = self._safe_load(self.start_file) or {}
        start_time = self._start_time or start_meta.get("start_time", final_end)
        elapsed = max(final_end - start_time, 0.0)

        if self.verbose:
            self._print_summary(start_time, final_end, elapsed, label or self.name)

        self._cleanup()
        return elapsed

    def _cleanup(self) -> None:
        try:
            if self.start_file.exists():
                self.start_file.unlink()
        except OSError:
            pass

        if not self.run_dir or not self.run_dir.exists():
            return

        for child in self.run_dir.iterdir():
            try:
                child.unlink()
            except OSError:
                pass
        try:
            self.run_dir.rmdir()
        except OSError:
            pass

    def _print_summary(self, start_time: float, end_time: float, elapsed: float, label: str) -> None:
        print(
            f"⏱️ {label} 完成 (world_size={self.world_size}) "
            f"in {elapsed:.2f}s "
            f"[{self._format_ts(start_time)} - {self._format_ts(end_time)}]"
        )

    @staticmethod
    def _format_ts(timestamp: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

    @staticmethod
    def _safe_load(path: Path) -> Optional[dict]:
        try:
            with path.open("r") as fp:
                return json.load(fp)
        except (FileNotFoundError, ValueError, OSError):
            return None

