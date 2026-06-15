import re
import time
from datetime import datetime, timedelta

def format_duration(seconds: float) -> str:
    """Formats a duration in seconds into a human-readable string (e.g. 2m 11s or 45s)."""
    if seconds is None or seconds < 0:
        return "Pending"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}m {secs}s"

class GitCloneParser:
    """
    Parses git clone --progress output to extract phase, percentages,
    download sizes, speed, and estimates remaining download size.
    """
    def __init__(self):
        self.phase = "Initializing"
        self.percent = 0
        self.downloaded_mb = 0.0
        self.speed_mbs = 0.0
        self.remaining_mb = 0.0

    def parse_line(self, line: str) -> bool:
        """
        Parses a single line of git clone output.
        Returns True if progress values changed, False otherwise.
        """
        changed = False
        
        # Detect phase
        if "Counting objects" in line:
            self.phase = "Counting Objects"
            changed = True
        elif "Compressing objects" in line:
            self.phase = "Compressing Objects"
            changed = True
        elif "Receiving objects" in line:
            self.phase = "Receiving Objects"
            changed = True
        elif "Resolving deltas" in line:
            self.phase = "Resolving Deltas"
            changed = True
            
        # Parse percentage (e.g. 72%)
        pct_match = re.search(r"(\d+)%", line)
        if pct_match:
            self.percent = int(pct_match.group(1))
            changed = True
            
        # For Receiving Objects phase, extract size and speed
        if self.phase == "Receiving Objects":
            parts = line.split("|")
            if len(parts) > 0:
                size_part = parts[0]
                # Look for downloaded size e.g. "348.00 MiB" or "12.50 MB"
                size_match = re.search(r"([\d\.]+)\s*([KMGT]i?B)", size_part)
                if size_match:
                    val = float(size_match.group(1))
                    unit = size_match.group(2).lower()
                    if "g" in unit:
                        self.downloaded_mb = val * 1024.0
                    elif "m" in unit:
                        self.downloaded_mb = val
                    elif "k" in unit:
                        self.downloaded_mb = val / 1024.0
                    else:
                        self.downloaded_mb = val / (1024.0 * 1024.0)
                    changed = True
                    
            if len(parts) > 1:
                speed_part = parts[1]
                speed_match = re.search(r"([\d\.]+)\s*([KMGT]i?B/s)", speed_part)
                if speed_match:
                    val = float(speed_match.group(1))
                    unit = speed_match.group(2).lower()
                    if "g" in unit:
                        self.speed_mbs = val * 1024.0
                    elif "m" in unit:
                        self.speed_mbs = val
                    elif "k" in unit:
                        self.speed_mbs = val / 1024.0
                    else:
                        self.speed_mbs = val / (1024.0 * 1024.0)
                    changed = True
                    
            # Estimate remaining MBs
            if self.percent > 0 and self.percent < 100:
                total_est = self.downloaded_mb / (self.percent / 100.0)
                self.remaining_mb = max(0.0, total_est - self.downloaded_mb)
            elif self.percent == 100:
                self.remaining_mb = 0.0
                
        return changed


class PipelineTracker:
    """
    Manages the overall stage-based progress system, timing metrics,
    and weighted progress values.
    """
    STAGE_WEIGHTS = {
        1: 0.20,  # Clone Repository
        2: 0.35,  # Mine Commit History
        3: 0.10,  # Mine Contributors
        4: 0.05,  # Mine File Modifications
        5: 0.20,  # Compute Quality Metrics
        6: 0.05,  # Feature Engineering
        7: 0.03,  # Risk Prediction
        8: 0.02   # Generate Report
    }

    STAGE_NAMES = {
        1: "Clone Repository",
        2: "Mine Commit History",
        3: "Mine Contributors",
        4: "Mine File Modifications",
        5: "Compute Quality Metrics",
        6: "Feature Engineering",
        7: "Risk Prediction",
        8: "Generate Report"
    }

    def __init__(self, estimated_total_runtime: float):
        self.estimated_total_runtime = estimated_total_runtime
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.current_stage = 1
        
        # Stage states: "Pending", "Running", "Complete", "Failed"
        self.stage_status = {i: "Pending" for i in range(1, 9)}
        self.stage_start_times = {i: None for i in range(1, 9)}
        self.stage_end_times = {i: None for i in range(1, 9)}
        self.stage_progress = {i: 0.0 for i in range(1, 9)}  # 0.0 to 1.0 within stage

    def start_stage(self, stage_idx: int):
        """Marks a stage as running and records its start time."""
        if stage_idx < 1 or stage_idx > 8:
            return
        
        # Complete previous stages if not already done
        for prev in range(1, stage_idx):
            if self.stage_status[prev] in ("Pending", "Running"):
                self.complete_stage(prev)
                
        self.current_stage = stage_idx
        self.stage_status[stage_idx] = "Running"
        self.stage_start_times[stage_idx] = time.time()
        self.stage_progress[stage_idx] = 0.0
        self.update_activity()

    def update_stage_progress(self, stage_idx: int, progress: float):
        """Updates internal progress percentage of a stage (0.0 to 1.0)."""
        if stage_idx in self.stage_progress:
            self.stage_progress[stage_idx] = min(1.0, max(0.0, progress))
            self.update_activity()

    def complete_stage(self, stage_idx: int):
        """Marks a stage as complete and records its end time."""
        if stage_idx in self.stage_status:
            self.stage_status[stage_idx] = "Complete"
            self.stage_progress[stage_idx] = 1.0
            if self.stage_start_times[stage_idx] is None:
                self.stage_start_times[stage_idx] = time.time()
            self.stage_end_times[stage_idx] = time.time()
            self.update_activity()

    def fail_stage(self, stage_idx: int):
        """Marks a stage as failed."""
        if stage_idx in self.stage_status:
            self.stage_status[stage_idx] = "Failed"
            self.stage_end_times[stage_idx] = time.time()
            self.update_activity()

    def update_activity(self):
        """Resets the stall watch timer."""
        self.last_update_time = time.time()

    def check_stall(self) -> bool:
        """Returns True if no progress update has occurred for > 60 seconds."""
        # Only check stalls if the pipeline is actively running
        active = any(status == "Running" for status in self.stage_status.values())
        if active and (time.time() - self.last_update_time > 60.0):
            return True
        return False

    def get_elapsed_and_remaining(self, stage_idx: int) -> tuple:
        """
        Calculates elapsed time and estimated remaining time for a stage.
        Returns (elapsed_str, remaining_str)
        """
        status = self.stage_status[stage_idx]
        start = self.stage_start_times[stage_idx]
        end = self.stage_end_times[stage_idx]
        
        if status == "Pending":
            return "", "Pending"
            
        if status == "Complete":
            elapsed = end - start if (end and start) else 0.0
            return format_duration(elapsed), ""
            
        if status == "Failed":
            elapsed = end - start if (end and start) else 0.0
            return f"{format_duration(elapsed)} (Failed)", ""
            
        # Running stage calculations
        elapsed = time.time() - start if start else 0.0
        progress = self.stage_progress[stage_idx]
        
        # Calculate ETA based on progress rate or predicted budget weight
        weight = self.STAGE_WEIGHTS[stage_idx]
        allocated_time = self.estimated_total_runtime * weight
        
        if progress > 0.01:
            est_total_for_stage = elapsed / progress
            remaining = max(0.0, est_total_for_stage - elapsed)
        else:
            remaining = max(0.0, allocated_time - elapsed)
            
        return format_duration(elapsed) + " elapsed", format_duration(remaining) + " remaining"

    def get_overall_progress(self) -> float:
        """Computes the weighted overall progress fraction (0.0 to 1.0)."""
        overall = 0.0
        for idx, weight in self.STAGE_WEIGHTS.items():
            overall += self.stage_progress[idx] * weight
        return min(1.0, max(0.0, overall))

    def get_overall_elapsed_and_remaining(self) -> tuple:
        """
        Computes overall elapsed and remaining time in seconds.
        Returns (elapsed_seconds, remaining_seconds, est_completion_time)
        """
        elapsed = time.time() - self.start_time
        progress = self.get_overall_progress()
        
        if progress > 0.05:
            total_est = elapsed / progress
            remaining = max(0.0, total_est - elapsed)
        else:
            remaining = max(0.0, self.estimated_total_runtime - elapsed)
            
        completion_dt = datetime.now() + timedelta(seconds=remaining)
        completion_str = completion_dt.strftime("%I:%M %p")
        
        return elapsed, remaining, completion_str

def render_pipeline_ui(tracker: PipelineTracker, warning_msg: str = None) -> str:
    # Build list of stages
    stages_html = []
    for idx in range(1, 9):
        name = tracker.STAGE_NAMES[idx]
        status = tracker.stage_status[idx]
        elapsed, remaining = tracker.get_elapsed_and_remaining(idx)
        
        # Determine icon and styling based on status
        if status == "Pending":
            icon = '<span style="color:#475569; font-size:1.15rem; margin-right:0.5rem;">⏳</span>'
            status_style = 'color: #475569;'
            time_info = '<span style="color:#475569;">Pending</span>'
        elif status == "Running":
            icon = '<span style="color:#60a5fa; font-size:1.15rem; margin-right:0.5rem; display:inline-block; animation: spin 2s linear infinite;">🔄</span>'
            status_style = 'color: #f8fafc; font-weight: 600;'
            time_info = f'<span style="color:#60a5fa;">{elapsed}</span><br><span style="color:#94a3b8; font-size: 0.72rem;">{remaining}</span>'
        elif status == "Complete":
            icon = '<span style="color:#4ade80; font-size:1.15rem; margin-right:0.5rem;">✓</span>'
            status_style = 'color: #94a3b8;'
            time_info = f'<span style="color:#4ade80;">{elapsed}</span>'
        else: # Failed
            icon = '<span style="color:#f87171; font-size:1.15rem; margin-right:0.5rem;">❌</span>'
            status_style = 'color: #f87171; font-weight: 600;'
            time_info = f'<span style="color:#f87171;">{elapsed}</span>'
            
        stages_html.append(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.6rem 0; border-bottom:1px solid #1e293b;">
          <div style="display:flex; align-items:center; {status_style}">
            {icon} <span>{name}</span>
          </div>
          <div style="font-size:0.8rem; text-align:right; font-family:monospace; line-height:1.2;">
            {time_info}
          </div>
        </div>
        """)
        
    warning_banner = ""
    if warning_msg:
        warning_banner = f"""
        <div style="background:rgba(249,115,22,0.1); border:1px solid rgba(249,115,22,0.3); border-radius:8px; padding:0.8rem; margin-bottom:1rem; color:#fb923c; font-size:0.8rem; display:flex; align-items:center; gap:0.5rem;">
          ⚠️ <strong>Warning:</strong> {warning_msg}
        </div>
        """
        
    # Get overall stats
    overall_pct = tracker.get_overall_progress()
    _, _, comp_time = tracker.get_overall_elapsed_and_remaining()
    
    html = f"""
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    <div style="background:#111318; border:1px solid #1e293b; border-radius:16px; padding:1.5rem; margin-top:1rem; box-shadow:0 4px 20px rgba(0,0,0,0.25);">
      {warning_banner}
      <div style="font-weight:700; color:#f8fafc; margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.9rem; text-transform:uppercase; letter-spacing:0.05em; color:#6366f1;">Pipeline Stages</span>
        <span style="font-size:0.95rem; font-weight:800; color:#a5b4fc;">ETA Completion: {comp_time}</span>
      </div>
      
      <div style="margin-bottom:1.5rem;">
        {"".join(stages_html)}
      </div>
    </div>
    """
    return html

def render_git_progress_ui(parser: GitCloneParser) -> str:
    pct = parser.percent
    bar_width = 30
    filled = int(pct / 100.0 * bar_width)
    empty = bar_width - filled
    bar_str = "█" * filled + "░" * empty
    
    # Format size
    downloaded_str = f"{parser.downloaded_mb:.1f} MB" if parser.downloaded_mb >= 1.0 else f"{parser.downloaded_mb * 1024:.0f} KB"
    remaining_str = f"{parser.remaining_mb:.1f} MB" if parser.remaining_mb >= 1.0 else f"{parser.remaining_mb * 1024:.0f} KB"
    speed_str = f"{parser.speed_mbs:.2f} MB/s" if parser.speed_mbs >= 1.0 else f"{parser.speed_mbs * 1024:.1f} KB/s"
    
    html = f"""
    <div style="background:#181b22; border:1px solid #1e293b; border-radius:12px; padding:1.2rem; margin-top:1rem;">
      <div style="font-size:0.75rem; font-weight:700; color:#fb923c; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;">
        📥 Repository Download Progress ({parser.phase})
      </div>
      <div style="font-family:monospace; font-size:1.1rem; color:#f8fafc; letter-spacing:0.1em; margin-bottom:0.5rem;">
        {bar_str} &nbsp; {pct}%
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; font-size:0.75rem; color:#94a3b8; font-family:monospace;">
        <div>Downloaded:<br><strong style="color:#f8fafc; font-size:0.85rem;">{downloaded_str}</strong></div>
        <div>Remaining:<br><strong style="color:#f8fafc; font-size:0.85rem;">{remaining_str}</strong></div>
        <div>Speed:<br><strong style="color:#f8fafc; font-size:0.85rem;">{speed_str}</strong></div>
      </div>
    </div>
    """
    return html

def render_commit_progress_ui(processed: int, total: int, pct: int, speed: float, remaining_str: str) -> str:
    bar_width = 30
    filled = int(pct / 100.0 * bar_width)
    empty = bar_width - filled
    bar_str = "█" * filled + "░" * empty
    
    html = f"""
    <div style="background:#181b22; border:1px solid #1e293b; border-radius:12px; padding:1.2rem; margin-top:1rem;">
      <div style="font-size:0.75rem; font-weight:700; color:#60a5fa; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;">
        📝 Commits Processed (Stage 2)
      </div>
      <div style="font-family:monospace; font-size:1.1rem; color:#f8fafc; letter-spacing:0.1em; margin-bottom:0.5rem;">
        {bar_str} &nbsp; {pct}%
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; font-size:0.75rem; color:#94a3b8; font-family:monospace;">
        <div>Processed:<br><strong style="color:#f8fafc; font-size:0.85rem;">{processed:,} / {total:,}</strong></div>
        <div>Speed:<br><strong style="color:#f8fafc; font-size:0.85rem;">{speed:.1f} commits/sec</strong></div>
        <div>ETA:<br><strong style="color:#f8fafc; font-size:0.85rem;">{remaining_str}</strong></div>
      </div>
    </div>
    """
    return html

def render_file_progress_ui(processed: int, total: int, current_file: str, speed: float, remaining_str: str) -> str:
    pct = int(processed / total * 100) if total > 0 else 0
    bar_width = 30
    filled = int(pct / 100.0 * bar_width)
    empty = bar_width - filled
    bar_str = "█" * filled + "░" * empty
    
    if len(current_file) > 40:
        current_file = "..." + current_file[-37:]
        
    html = f"""
    <div style="background:#181b22; border:1px solid #1e293b; border-radius:12px; padding:1.2rem; margin-top:1rem;">
      <div style="font-size:0.75rem; font-weight:700; color:#4ade80; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;">
        🔬 Files Processed (Stage 5)
      </div>
      <div style="font-family:monospace; font-size:1.1rem; color:#f8fafc; letter-spacing:0.1em; margin-bottom:0.5rem;">
        {bar_str} &nbsp; {pct}%
      </div>
      <div style="margin-bottom:0.5rem;">
        <span style="font-size:0.72rem; color:#64748b; font-family:monospace;">Current File:</span><br>
        <strong style="color:#f8fafc; font-size:0.8rem; font-family:monospace; word-break:break-all;">{current_file}</strong>
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; font-size:0.75rem; color:#94a3b8; font-family:monospace;">
        <div>Processed:<br><strong style="color:#f8fafc; font-size:0.85rem;">{processed:,} / {total:,}</strong></div>
        <div>Speed:<br><strong style="color:#f8fafc; font-size:0.85rem;">{speed:.1f} files/sec</strong></div>
        <div>ETA:<br><strong style="color:#f8fafc; font-size:0.85rem;">{remaining_str}</strong></div>
      </div>
    </div>
    """
    return html
